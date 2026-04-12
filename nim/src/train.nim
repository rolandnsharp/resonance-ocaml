## Resonance training — SineGate MLP stack on Shakespeare.
##
## Architecture:
##   token → drive → pad to dim → [RMSNorm → W → SineGate → + residual] × L
##   → RMSNorm → W_out → cross-entropy
##
## All per-position (no recurrence in this version — fast on GPU).
## Damped rotation recurrence will be added once this baseline converges.

import std/[math, os, strformat, strutils, times, random, algorithm]
import gpu, model, dsp

const
  opNN  = 0  # C = A @ B
  opNT  = 2  # C = A @ B^T
  opTNA = 5  # C += A^T @ B

proc loadText(path: string): seq[int32] =
  ## Load raw text as byte tokens (vocab 256)
  let data = readFile(path)
  result = newSeq[int32](data.len)
  for i, c in data: result[i] = c.int32

proc loadShard(path: string): seq[int32] =
  ## Load a FineWeb binary shard (header + uint16/uint32 tokens)
  let f = open(path, fmRead)
  var header: array[256, int32]
  discard f.readBuffer(addr header[0], 256 * 4)
  assert header[0] == 20240520, "Bad shard magic: " & $header[0]
  let ntok = header[2]
  let version = header[1]
  if version == 1:
    # uint16 tokens
    var raw = newSeq[uint16](ntok)
    discard f.readBuffer(addr raw[0], ntok * 2)
    result = newSeq[int32](ntok)
    for i in 0..<ntok: result[i] = raw[i].int32
  elif version == 7:
    # uint32 tokens
    var raw = newSeq[uint32](ntok)
    discard f.readBuffer(addr raw[0], ntok * 4)
    result = newSeq[int32](ntok)
    for i in 0..<ntok: result[i] = raw[i].int32
  else:
    quit "Unknown shard version: " & $version
  f.close()

proc loadData(path: string): seq[int32] =
  ## Auto-detect: raw text (.txt) or FineWeb shard (.bin)
  if path.endsWith(".txt"):
    loadText(path)
  else:
    loadShard(path)

proc findShards(dir: string): seq[string] =
  ## Find all training shards in a directory
  result = @[]
  for kind, path in walkDir(dir):
    if kind == pcFile and path.contains("train") and path.endsWith(".bin"):
      result.add(path)
  result.sort()

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

    # === Forward: strike → propagate → [resonate → interfere → harmonics → superpose] × L → listen ===

    # Strike: token → oscillator excitation
    tokBuf.strike(m.drive.w, driveBuf, nOsc, BT)

    # Propagate: FFT convolve drives with oscillator impulse responses
    driveBuf.propagate(columnsBuf, colsFft, prodPosFft, prodVelFft, convPosBuf, convVelBuf,
                       m.hPosFft, m.hVelFft, states[0],
                       batchSize, seqLen, nOsc, dim, fftLen, fftCLen, totalOsc)

    # Layer stack: normalize → gate → resonate → interfere → harmonics → superpose
    for l in 0..<nL:
      let ly = m.layers[l]

      # Normalize: scale to unit energy
      states[l].normalize(normed[l], dim, BT)

      # Gate: project state → per-oscillator controls (how to remember, absorb, read)
      normed[l].project(ly.projGamma.w, gammaBuf, BT, nOsc, dim)
      gammaBuf.addBias(ly.projGammaBias.w, nOsc, BT * nOsc)
      gammaBuf.activate(BT * nOsc)

      normed[l].project(ly.projBeta.w, betaBuf, BT, nOsc, dim)
      betaBuf.addBias(ly.projBetaBias.w, nOsc, BT * nOsc)
      betaBuf.activate(BT * nOsc)

      normed[l].project(ly.projSense.w, senseBuf, BT, nOsc, dim)
      senseBuf.addBias(ly.projSenseBias.w, nOsc, BT * nOsc)
      senseBuf.activate(BT * nOsc)

      # Resonate: damped rotation — the oscillator equation
      gammaBuf.resonate(betaBuf, senseBuf, states[0], oscOutBuf,
                        m.cosW, m.sinW, m.freqs, batchSize, seqLen, nOsc)

      # Interfere: pairwise cross-terms between oscillators
      oscOutBuf.interfere(ly.spectralRe.w, mixed[l], BT, dim)

      # Harmonics: x × sin(x) — overtone generation
      mixed[l].harmonics(actBuf, BT * dim)

      # Superpose: residual — waves add
      states[l].superpose(actBuf, states[l+1], BT * dim)

    # Listen: normalize → project → dot with drive signatures
    states[nL].normalize(normed[nL], dim, BT)
    normed[nL].interfere(m.wOut.w, logitBuf, BT, vocab)  # W_out as final interference

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
  let vocabSize = if paramCount() >= 5: parseInt(paramStr(5)) else: 0  # 0 = auto
  let seqLen = if paramCount() >= 6: parseInt(paramStr(6)) else: 128
  let batchSize = if paramCount() >= 7: parseInt(paramStr(7)) else: 64

  # Load data — either single file or cycle through shards
  var data: seq[int32]
  var vocab: int

  if dataPath.endsWith(".txt"):
    data = loadText(dataPath)
    vocab = if vocabSize > 0: vocabSize else: 256
    echo &"Data: {data.len} bytes from {dataPath} (vocab {vocab})"
  elif dirExists(dataPath):
    # Directory of shards — load first shard to start, cycle in training
    let shards = findShards(dataPath)
    echo &"Found {shards.len} training shards in {dataPath}"
    # Concatenate all shards (simple approach for now)
    data = @[]
    for s in shards:
      let shard = loadShard(s)
      data.add(shard)
      if data.len > 100_000_000: break  # cap at 100M tokens
    vocab = if vocabSize > 0: vocabSize else: 1024
    echo &"Data: {data.len} tokens (vocab {vocab})"
  else:
    data = loadData(dataPath)
    vocab = if vocabSize > 0: vocabSize else: 1024
    echo &"Data: {data.len} tokens from {dataPath} (vocab {vocab})"

  var m = createModel(nOsc, nLayers, vocabSize = vocab, seqLen = seqLen)
  train(m, data, steps, seqLen = seqLen, batchSize = batchSize, lr = 3e-4)

main()
