## Resonance training — SineGate MLP stack on Shakespeare.
##
## Architecture:
##   token → drive → pad to dim → [RMSNorm → W → SineGate → + residual] × L
##   → RMSNorm → W_out → cross-entropy
##
## All per-position (no recurrence in this version — fast on GPU).
## Damped rotation recurrence will be added once this baseline converges.

import std/[math, os, strformat, strutils, times, random]
import gpu, model

const
  opNN  = 0  # C = A @ B
  opNT  = 2  # C = A @ B^T
  opTNA = 5  # C += A^T @ B

proc loadText(path: string): seq[int32] =
  let data = readFile(path)
  result = newSeq[int32](data.len)
  for i, c in data: result[i] = c.int32

proc train(m: var Model, data: seq[int32], steps, seqLen, batchSize: int, lr: float32) =
  let dim = m.dim
  let nOsc = m.nOsc
  let vocab = m.vocabSize
  let BT = batchSize * seqLen
  let dataLen = data.len
  let nL = m.nLayers

  # GPU buffers
  let tokBuf = gpuCreate(BT)
  let tgtBuf = gpuCreate(BT)
  let driveBuf = gpuCreate(BT * nOsc)
  # State + per-layer caches for backward
  var states = newSeq[GpuBuf](nL + 1)     # state before each layer + final
  var normed = newSeq[GpuBuf](nL + 1)     # normed at each layer + final
  var mixed = newSeq[GpuBuf](nL)          # W output at each layer
  for i in 0..nL:
    states[i] = gpuCreate(BT * dim)
    normed[i] = gpuCreate(BT * dim)
  for i in 0..<nL:
    mixed[i] = gpuCreate(BT * dim)

  let gammaBuf = gpuCreate(BT * nOsc)
  let betaBuf = gpuCreate(BT * nOsc)
  let senseBuf = gpuCreate(BT * nOsc)
  let oscOutBuf = gpuCreate(BT * dim)
  let actBuf = gpuCreate(BT * dim)
  let logitBuf = gpuCreate(BT * vocab)
  let lossBuf = gpuCreate(BT)
  let dLogitBuf = gpuCreate(BT * vocab)
  let dStateBuf = gpuCreate(BT * dim)
  let dNormBuf = gpuCreate(BT * dim)
  let dMixBuf = gpuCreate(BT * dim)
  let dOscOutBuf = gpuCreate(BT * dim)
  let dGammaBuf = gpuCreate(BT * nOsc)   # pre-sigmoid gradient
  let dBetaBuf = gpuCreate(BT * nOsc)
  let dSenseBuf = gpuCreate(BT * nOsc)
  let dBankBuf = gpuCreate(BT * dim)     # gradient for bank output
  let dProjBuf = gpuCreate(BT * nOsc)    # scratch for sigmoid backward

  var tokData = newSeq[int32](BT)
  var tgtData = newSeq[int32](BT)

  # Pre-allocate FFT bank buffers (outside loop to avoid OOM)
  let fftLen = 2 * seqLen
  let fftCLen = fftLen div 2 + 1
  let totalOsc = batchSize * nOsc
  let columnsBuf = gpuCreate(totalOsc * fftLen)
  let colsFft = gpuCreate(totalOsc * fftCLen * 2)
  let prodPosFft = gpuCreate(totalOsc * fftCLen * 2)
  let prodVelFft = gpuCreate(totalOsc * fftCLen * 2)
  let convPosBuf = gpuCreate(totalOsc * fftLen)
  let convVelBuf = gpuCreate(totalOsc * fftLen)

  # Bank backward buffers (pre-allocated)
  let dBankPosBuf = gpuCreate(totalOsc * fftLen)
  let dBankVelBuf = gpuCreate(totalOsc * fftLen)
  let dPosFft = gpuCreate(totalOsc * fftCLen * 2)
  let dVelFft = gpuCreate(totalOsc * fftCLen * 2)
  let dDrivePosFft = gpuCreate(totalOsc * fftCLen * 2)
  let dDriveVelFft = gpuCreate(totalOsc * fftCLen * 2)
  let dDriveFft = gpuCreate(totalOsc * fftCLen * 2)
  let dDriveConv = gpuCreate(totalOsc * fftLen)
  let dPosTmp = gpuCreate(BT * nOsc)
  let dVelTmp = gpuCreate(BT * nOsc)

  # Spectral mixing buffers
  let specLen = dim div 2 + 1
  let gateBuf = gpuCreate(BT * specLen)
  let specFftBuf = gpuCreate(BT * specLen * 2)  # complex
  let specProdBuf = gpuCreate(BT * specLen * 2)  # complex

  let t0 = epochTime()

  for step in 0..<steps:
    # Sample
    for b in 0..<batchSize:
      let start = rand(dataLen - seqLen - 2)
      for i in 0..<seqLen:
        tokData[b * seqLen + i] = data[start + i]
        tgtData[b * seqLen + i] = data[start + i + 1]
    tokBuf.gpuUploadInts(tokData)
    tgtBuf.gpuUploadInts(tgtData)

    # OneCycleLR-style: 5% warmup, then cosine decay to near zero
    let warmup = steps div 20  # 5%
    let s = if step < warmup: step.float32 / max(1, warmup).float32
            else:
              let progress = (step - warmup).float32 / max(1, steps - warmup).float32
              0.5 * (1.0 + cos(PI * progress))
    let curLr = lr * s

    # === Forward ===

    # Embed drives
    gpu_embed_fwd(tokBuf.data, m.drive.w.data, driveBuf.data, nOsc.cint, BT.cint)

    # FFT bank: batched convolution — 3 kernel launches instead of 6144
    states[0].gpuZero()
    columnsBuf.gpuZero()
    gpu_bank_gather(driveBuf.data, columnsBuf.data,
      batchSize.cint, seqLen.cint, nOsc.cint, fftLen.cint)

    # 2. Batched FFT: all columns at once
    discard gpu_fft_r2c_batched(columnsBuf.data, colsFft.data, fftLen.cint, totalOsc.cint)

    # 3. Multiply by kernel FFTs (broadcast: each batch uses same kernel per oscillator)
    gpu_complex_mul_broadcast(colsFft.data, m.hPosFft.data, prodPosFft.data,
      (nOsc * fftCLen).cint, (totalOsc * fftCLen).cint)
    gpu_complex_mul_broadcast(colsFft.data, m.hVelFft.data, prodVelFft.data,
      (nOsc * fftCLen).cint, (totalOsc * fftCLen).cint)

    # 4. Batched IFFT
    discard gpu_fft_c2r_batched(prodPosFft.data, convPosBuf.data, fftLen.cint, totalOsc.cint)
    discard gpu_fft_c2r_batched(prodVelFft.data, convVelBuf.data, fftLen.cint, totalOsc.cint)

    # 5. Scatter to state
    let invFft = 1.0 / fftLen.float32
    gpu_bank_scatter(convPosBuf.data, states[0].data,
      batchSize.cint, seqLen.cint, nOsc.cint, dim.cint, 0.cint, invFft.cfloat)
    gpu_bank_scatter(convVelBuf.data, states[0].data,
      batchSize.cint, seqLen.cint, nOsc.cint, dim.cint, nOsc.cint, invFft.cfloat)

    # Layer stack: norm → project controls → damped rotation → W → SineGate → residual
    for l in 0..<nL:
      gpu_rmsnorm(states[l].data, normed[l].data, dim.cint, BT.cint)

      # Project controls with bias: gamma, beta, sense (dim → n_osc each)
      gpuSgemm(opNN, BT, nOsc, dim, normed[l], m.layers[l].projGamma.w, gammaBuf)
      gpu_add_bias(gammaBuf.data, m.layers[l].projGammaBias.w.data, nOsc.cint, (BT * nOsc).cint)
      gpu_sigmoid(gammaBuf.data, gammaBuf.data, (BT * nOsc).cint)
      gpuSgemm(opNN, BT, nOsc, dim, normed[l], m.layers[l].projBeta.w, betaBuf)
      gpu_add_bias(betaBuf.data, m.layers[l].projBetaBias.w.data, nOsc.cint, (BT * nOsc).cint)
      gpu_sigmoid(betaBuf.data, betaBuf.data, (BT * nOsc).cint)
      gpuSgemm(opNN, BT, nOsc, dim, normed[l], m.layers[l].projSense.w, senseBuf)
      gpu_add_bias(senseBuf.data, m.layers[l].projSenseBias.w.data, nOsc.cint, (BT * nOsc).cint)
      gpu_sigmoid(senseBuf.data, senseBuf.data, (BT * nOsc).cint)

      # FM rotation scan: damped rotation + cross-oscillator frequency modulation
      gpu_rotation_scan(gammaBuf.data, betaBuf.data, senseBuf.data,
        states[0].data, oscOutBuf.data,
        m.cosW.data, m.sinW.data, m.freqs.data, m.layers[l].fmDepth.w.data,
        batchSize.cint, seqLen.cint, nOsc.cint, m.layers[l].foldOffset.cint)

      # Interference: dense W computes all pairwise cross-terms between oscillators
      # spectralRe is repurposed as wMix (dim × dim)
      gpuSgemm(opNN, BT, dim, dim, oscOutBuf, m.layers[l].spectralRe.w, mixed[l])
      gpu_sinegate_fwd(mixed[l].data, actBuf.data, (BT * dim).cint)
      gpu_add(states[l].data, actBuf.data, states[l+1].data, (BT * dim).cint)

    # Output: norm → W_out → logits
    gpu_rmsnorm(states[nL].data, normed[nL].data, dim.cint, BT.cint)
    gpuSgemm(opNN, BT, vocab, dim, normed[nL], m.wOut.w, logitBuf)

    # Loss
    gpu_ce_loss(logitBuf.data, tgtBuf.data, lossBuf.data, vocab.cint, BT.cint)
    let ld = gpuDownload(lossBuf)
    var totalLoss: float32 = 0
    for i in 0..<BT: totalLoss += ld[i]
    totalLoss /= BT.float32

    # === Backward ===

    gpu_ce_backward(logitBuf.data, tgtBuf.data, dLogitBuf.data, vocab.cint, BT.cint)

    # Through W_out
    gpuSgemm(opNT, BT, dim, vocab, dLogitBuf, m.wOut.w, dNormBuf)  # dNorm
    gpuSgemm(opTNA, dim, vocab, BT, normed[nL], dLogitBuf, m.wOut.g)  # dW_out

    # Through final RMSNorm
    gpu_rmsnorm_bwd(states[nL].data, dNormBuf.data, dStateBuf.data, dim.cint, BT.cint)

    # Through layers in reverse
    for l in countdown(nL - 1, 0):
      # Through SineGate: dMix = dState * sinegate'(mixed)
      gpu_sinegate_bwd(mixed[l].data, dStateBuf.data, dMixBuf.data, (BT * dim).cint)

      # Through interference (dense W): dW += oscOut^T @ dMix, dOscOut = dMix @ W^T
      gpuSgemm(opTNA, dim, dim, BT, oscOutBuf, dMixBuf, m.layers[l].spectralRe.g)
      gpuSgemm(opNT, BT, dim, dim, dMixBuf, m.layers[l].spectralRe.w, dOscOutBuf)

      # Through rotation scan backward
      gpu_rotation_scan_bwd(
        gammaBuf.data, betaBuf.data, senseBuf.data,
        states[0].data, oscOutBuf.data, dOscOutBuf.data,
        dGammaBuf.data, dBetaBuf.data, dSenseBuf.data, dBankBuf.data,
        m.cosW.data, m.sinW.data, m.freqs.data,
        batchSize.cint, seqLen.cint, nOsc.cint)

      # Through sigmoid + projection weight/bias grad + dNormed accumulation
      # Do each projection once: sigmoid_bwd → weight grad → bias grad → normed grad
      let opNTA = 3  # opNT + accumulate

      # gamma
      gpu_sigmoid_bwd(gammaBuf.data, dGammaBuf.data, dProjBuf.data, (BT * nOsc).cint)
      gpuSgemm(opTNA, dim, nOsc, BT, normed[l], dProjBuf, m.layers[l].projGamma.g)
      gpu_bias_grad(dProjBuf.data, m.layers[l].projGammaBias.g.data, nOsc.cint, BT.cint)
      gpuSgemm(opNTA, BT, dim, nOsc, dProjBuf, m.layers[l].projGamma.w, dNormBuf)

      # beta
      gpu_sigmoid_bwd(betaBuf.data, dBetaBuf.data, dProjBuf.data, (BT * nOsc).cint)
      gpuSgemm(opTNA, dim, nOsc, BT, normed[l], dProjBuf, m.layers[l].projBeta.g)
      gpu_bias_grad(dProjBuf.data, m.layers[l].projBetaBias.g.data, nOsc.cint, BT.cint)
      gpuSgemm(opNTA, BT, dim, nOsc, dProjBuf, m.layers[l].projBeta.w, dNormBuf)

      # sense
      gpu_sigmoid_bwd(senseBuf.data, dSenseBuf.data, dProjBuf.data, (BT * nOsc).cint)
      gpuSgemm(opTNA, dim, nOsc, BT, normed[l], dProjBuf, m.layers[l].projSense.g)
      gpu_bias_grad(dProjBuf.data, m.layers[l].projSenseBias.g.data, nOsc.cint, BT.cint)
      gpuSgemm(opNTA, BT, dim, nOsc, dProjBuf, m.layers[l].projSense.w, dNormBuf)

      # Through RMSNorm with complete dNormed (W_mix + all projections)
      gpu_rmsnorm_bwd(states[l].data, dNormBuf.data, dNormBuf.data, dim.cint, BT.cint)
      gpu_add_inplace(dStateBuf.data, dNormBuf.data, (BT * dim).cint)

    # Through FFT bank (adjoint): dState → dDrives via conj(H) convolution
    dBankPosBuf.gpuZero()
    dBankVelBuf.gpuZero()
    # Extract pos/vel from dState (BT, dim) → contiguous (BT, nOsc) → transposed (batch*nOsc, fftLen)
    gpu_strided_gather(dStateBuf.data, dPosTmp.data, BT.cint, dim.cint, nOsc.cint, 0.cint)
    gpu_strided_gather(dStateBuf.data, dVelTmp.data, BT.cint, dim.cint, nOsc.cint, nOsc.cint)
    dBankPosBuf.gpuZero()
    dBankVelBuf.gpuZero()
    gpu_bank_gather(dPosTmp.data, dBankPosBuf.data, batchSize.cint, seqLen.cint, nOsc.cint, fftLen.cint)
    gpu_bank_gather(dVelTmp.data, dBankVelBuf.data, batchSize.cint, seqLen.cint, nOsc.cint, fftLen.cint)
    # FFT, multiply by conj(H), IFFT
    discard gpu_fft_r2c_batched(dBankPosBuf.data, dPosFft.data, fftLen.cint, totalOsc.cint)
    discard gpu_fft_r2c_batched(dBankVelBuf.data, dVelFft.data, fftLen.cint, totalOsc.cint)
    gpu_complex_mul_conj_broadcast(dPosFft.data, m.hPosFft.data, dDrivePosFft.data,
      (nOsc * fftCLen).cint, (totalOsc * fftCLen).cint)
    gpu_complex_mul_conj_broadcast(dVelFft.data, m.hVelFft.data, dDriveVelFft.data,
      (nOsc * fftCLen).cint, (totalOsc * fftCLen).cint)
    # Sum pos and vel contributions
    gpu_add(dDrivePosFft.data, dDriveVelFft.data, dDriveFft.data, (totalOsc * fftCLen * 2).cint)
    # IFFT back to time domain
    discard gpu_fft_c2r_batched(dDriveFft.data, dDriveConv.data, fftLen.cint, totalOsc.cint)
    # Scatter back to (BT, n_osc) format — reuse driveBuf as dDrives
    driveBuf.gpuZero()
    gpu_bank_scatter(dDriveConv.data, driveBuf.data,
      batchSize.cint, seqLen.cint, nOsc.cint, nOsc.cint, 0.cint, invFft.cfloat)

    # Through embedding using the properly back-propagated drive gradients
    gpu_embed_bwd(tokBuf.data, driveBuf.data, m.drive.g.data, nOsc.cint, BT.cint)

    # Adam
    let b1: float32 = 0.9
    let b2: float32 = 0.999
    let wd: float32 = 0.01
    let t = (step + 1).float32
    let bc1 = 1.0 / (1.0 - pow(b1, t))
    let bc2 = 1.0 / (1.0 - pow(b2, t))

    m.drive.adamStep(curLr * 3.0, b1, b2, 0, bc1, bc2)
    m.wOut.adamStep(curLr, b1, b2, wd, bc1, bc2)
    for l in 0..<nL:
      m.layers[l].spectralRe.adamStep(curLr, b1, b2, wd, bc1, bc2)  # wMix
      m.layers[l].fmDepth.adamStep(curLr, b1, b2, 0, bc1, bc2)
      m.layers[l].projGammaBias.adamStep(curLr, b1, b2, 0, bc1, bc2)
      m.layers[l].projBetaBias.adamStep(curLr, b1, b2, 0, bc1, bc2)
      m.layers[l].projSenseBias.adamStep(curLr, b1, b2, 0, bc1, bc2)
      m.layers[l].projGamma.adamStep(curLr, b1, b2, wd, bc1, bc2)
      m.layers[l].projBeta.adamStep(curLr, b1, b2, wd, bc1, bc2)
      m.layers[l].projSense.adamStep(curLr, b1, b2, wd, bc1, bc2)

    sync()

    if step mod 100 == 0:
      let elapsed = epochTime() - t0
      let bpc = totalLoss / ln(2.0)
      let sps = if elapsed > 0: (step + 1).float / elapsed else: 0.0
      echo &"step {step:5d}  loss {totalLoss:.3f}  bpc {bpc:.3f}  lr {curLr:.1e}  [{sps:.1f} s/s]"

  echo "\nDone."

proc main() =
  randomize()
  gpuInit()

  let dataPath = if paramCount() >= 1: paramStr(1) else: "data/shakespeare.txt"
  let nOsc = if paramCount() >= 2: parseInt(paramStr(2)) else: 96
  let nLayers = if paramCount() >= 3: parseInt(paramStr(3)) else: 6
  let steps = if paramCount() >= 4: parseInt(paramStr(4)) else: 5000

  let data = loadText(dataPath)
  echo &"Data: {data.len} bytes from {dataPath}"

  var m = createModel(nOsc, nLayers, vocabSize = 256, seqLen = 128)
  train(m, data, steps, seqLen = 128, batchSize = 64, lr = 3e-4)

main()
