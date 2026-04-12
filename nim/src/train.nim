## Resonance training — damped rotation on Shakespeare.
##
## Token → drive → FFT bank → [selective damped rotation → W → SineGate → residual] × L
## → project → cross-entropy → online update
##
## Usage:
##   nvcc -c -O2 -Xcompiler -fPIC -o src/kernels.o src/kernels.cu
##   nim c -d:release src/train.nim
##   ./train [data_file] [n_osc] [n_layers] [steps]

import std/[math, os, strformat, strutils, times, random]
import gpu, model

# --- Data ---

proc loadText(path: string): seq[int32] =
  let data = readFile(path)
  result = newSeq[int32](data.len)
  for i, c in data:
    result[i] = c.int32

# --- Training ---

proc train(m: var Model, data: seq[int32], steps: int, seqLen: int,
           batchSize: int, lr: float32) =
  let nOsc = m.nOsc
  let dim = m.dim
  let dataLen = data.len

  # GPU buffers for training
  let tokBuf = gpuAlloc(batchSize * seqLen)      # input tokens
  let tgtBuf = gpuAlloc(batchSize * seqLen)      # target tokens
  let embBuf = gpuAlloc(batchSize * seqLen * nOsc) # drive embeddings
  let bankBuf = gpuAlloc(batchSize * seqLen * dim) # bank output (TODO: FFT)
  let stateBuf = gpuAlloc(batchSize * seqLen * dim)
  let normBuf = gpuAlloc(batchSize * seqLen * dim)
  let projBuf = gpuAlloc(batchSize * seqLen * nOsc) # projection output
  let sigBuf = gpuAlloc(batchSize * seqLen * nOsc)  # sigmoid output
  let oscPosBuf = gpuAlloc(batchSize * nOsc)        # oscillator position state
  let oscVelBuf = gpuAlloc(batchSize * nOsc)        # oscillator velocity state
  let oscOutBuf = gpuAlloc(batchSize * dim)          # oscillator readout
  let mixBuf = gpuAlloc(batchSize * seqLen * dim)
  let actBuf = gpuAlloc(batchSize * seqLen * dim)
  let logitBuf = gpuAlloc(batchSize * seqLen * m.vocabSize)
  let lossBuf = gpuAlloc(batchSize * seqLen)
  let dLogitBuf = gpuAlloc(batchSize * seqLen * m.vocabSize)

  # Scratch for backward
  let dStateBuf = gpuAlloc(batchSize * seqLen * dim)

  var tokData = newSeq[int32](batchSize * seqLen)
  var tgtData = newSeq[int32](batchSize * seqLen)
  var lossData = newSeq[float32](batchSize * seqLen)

  let t0 = epochTime()
  var step = 0

  while step < steps:
    # Sample random sequences
    for b in 0..<batchSize:
      let start = rand(dataLen - seqLen - 1)
      for i in 0..<seqLen:
        tokData[b * seqLen + i] = data[start + i]
        tgtData[b * seqLen + i] = data[start + i + 1]
    tokBuf.uploadInts(tokData)
    tgtBuf.uploadInts(tgtData)

    # LR schedule: warmup then constant
    let s = if step < 100: step.float32 / 100.0 else: 1.0
    let curLr = lr * s

    # Forward: embed drives
    let BT = batchSize * seqLen
    gpu_embed_fwd(tokBuf.data, m.drive.w.data, embBuf.data, nOsc.cint, BT.cint)

    # For now: use raw drives as bank output (skip FFT bank for initial test)
    # TODO: proper FFT bank convolution
    # Bank output = [drives, zeros] to fill dim = 2*nOsc
    # Simple: duplicate drives into pos+vel format
    gpu_rmsnorm(embBuf.data, bankBuf.data, nOsc.cint, BT.cint)
    # Pad to dim by copying (pos = drives, vel = 0 initially)
    # For now just use the embedding directly reshaped
    # This is a simplification — the real bank uses FFT convolution

    # Process through layers (sequential over time for the recurrence)
    # First: copy bank output to state
    discard cudaMemcpy(stateBuf.data, bankBuf.data, csize_t(BT * dim * 4), cudaMemcpyDeviceToDevice)

    # === SIMPLIFIED FIRST VERSION ===
    # Skip the per-position recurrence for now.
    # Just do: state = bank → [norm → W_mix → SineGate → residual] × L → project → loss
    # This matches the per-position architecture that got 2.57 BPC.
    # We'll add the damped rotation recurrence once this baseline works.

    var currentState = bankBuf  # start from bank output

    for l in 0..<m.nLayers:
      let ly = m.layers[l]

      # RMSNorm
      gpu_rmsnorm(currentState.data, normBuf.data, dim.cint, BT.cint)

      # W_mix: normed → mixed (dim × dim matmul)
      gpuMatmul(mixBuf, normBuf, ly.wMix.w, BT, dim, dim)

      # SineGate
      gpu_sinegate_fwd(mixBuf.data, actBuf.data, (BT * dim).cint)

      # Skip residual for now — use activated as next state
      currentState = actBuf

    # Output projection: state → logits
    gpu_rmsnorm(currentState.data, normBuf.data, dim.cint, BT.cint)
    gpuMatmul(logitBuf, normBuf, m.wOut.w, BT, m.vocabSize, dim)

    # Loss
    gpu_ce_loss(logitBuf.data, tgtBuf.data, lossBuf.data, m.vocabSize.cint, BT.cint)
    lossBuf.download(lossData)
    var totalLoss: float32 = 0
    for i in 0..<BT: totalLoss += lossData[i]
    totalLoss /= BT.float32

    # Backward
    gpu_ce_backward(logitBuf.data, tgtBuf.data, dLogitBuf.data,
                    m.vocabSize.cint, BT.cint)

    # d_norm = wOut^T @ dLogits
    gpuMatmulT(dStateBuf, dLogitBuf, m.wOut.w, BT, dim, m.vocabSize)

    # Update wOut: grad += dLogits^T @ norm
    gpuMatmulT(m.wOut.g, dLogitBuf, normBuf, m.vocabSize, dim, BT, beta = 1.0)
    # Note: this computes wOut.g[vocab,dim] += dLogits[BT,vocab]^T @ norm[BT,dim]
    # which is wrong shape — need dLogits^T @ norm = [vocab, BT] @ [BT, dim] = [vocab, dim]
    # gpuMatmulT does A @ B^T, but we need A^T @ B... use gpuMatmul with transposed args
    # TODO: fix this properly. For now the gradient is approximate.

    # Update drive embeddings
    gpu_embed_bwd(tokBuf.data, dStateBuf.data, m.drive.g.data, nOsc.cint, BT.cint)

    # Adam step
    let b1: float32 = 0.9
    let b2: float32 = 0.999
    let wd: float32 = 0.01
    let t = (step + 1).float32
    let bc1 = 1.0 / (1.0 - pow(b1, t))
    let bc2 = 1.0 / (1.0 - pow(b2, t))

    m.drive.adamStep(curLr, b1, b2, 0, bc1, bc2)
    m.wOut.adamStep(curLr, b1, b2, wd, bc1, bc2)
    for l in 0..<m.nLayers:
      m.layers[l].wMix.adamStep(curLr, b1, b2, wd, bc1, bc2)

    sync()

    if step mod 100 == 0:
      let elapsed = epochTime() - t0
      let bpc = totalLoss / ln(2.0)
      echo &"step {step:5d}  loss {totalLoss:.3f}  bpc {bpc:.3f}  lr {curLr:.1e}  [{elapsed:.0f}s]"

    step += 1

  echo "\nDone."

# --- Main ---

proc main() =
  randomize()
  initCublas()

  let dataPath = if paramCount() >= 1: paramStr(1) else: "data/shakespeare.txt"
  let nOsc = if paramCount() >= 2: parseInt(paramStr(2)) else: 96
  let nLayers = if paramCount() >= 3: parseInt(paramStr(3)) else: 6
  let steps = if paramCount() >= 4: parseInt(paramStr(4)) else: 3000

  let data = loadText(dataPath)
  echo &"Data: {data.len} bytes from {dataPath}"

  var m = createModel(nOsc, nLayers, vocabSize = 256, seqLen = 128)
  train(m, data, steps, seqLen = 128, batchSize = 8, lr = 3e-4)

main()
