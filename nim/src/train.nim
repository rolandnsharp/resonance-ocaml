## Resonance training — damped rotation on Shakespeare.
##
## Architecture:
##   token → drive → [RMSNorm → W_mix → SineGate → residual] × L → RMSNorm → W_out → loss
##
## Per-position online updates. Byte-level vocab (256).
## First version: per-position MLP stack (no recurrence yet).
## Next: add damped rotation recurrence using the CUDA kernel.
##
## Usage:
##   cd nim && make train

import std/[math, os, strformat, strutils, times, random]
import gpu, model

const
  ## gpuSgemm op bits: 4=transA, 2=transB, 1=accumulate
  opNN  = 0  # C = A @ B
  opNT  = 2  # C = A @ B^T
  opTN  = 4  # C = A^T @ B
  opTNA = 5  # C += A^T @ B (accumulate)

proc loadText(path: string): seq[int32] =
  let data = readFile(path)
  result = newSeq[int32](data.len)
  for i, c in data:
    result[i] = c.int32

proc train(m: var Model, data: seq[int32], steps: int, seqLen: int,
           batchSize: int, lr: float32) =
  let dim = m.dim
  let nOsc = m.nOsc
  let vocab = m.vocabSize
  let BT = batchSize * seqLen
  let dataLen = data.len

  # Pre-allocate all GPU buffers
  let tokBuf = gpuCreate(BT)          # input tokens
  let tgtBuf = gpuCreate(BT)          # targets
  let driveBuf = gpuCreate(BT * nOsc) # drive embeddings
  let stateBuf = gpuCreate(BT * dim)  # current state
  let normBuf = gpuCreate(BT * dim)   # after RMSNorm
  let mixBuf = gpuCreate(BT * dim)    # after W_mix
  let actBuf = gpuCreate(BT * dim)    # after SineGate
  let logitBuf = gpuCreate(BT * vocab)
  let lossBuf = gpuCreate(BT)
  let dLogitBuf = gpuCreate(BT * vocab)
  let dStateBuf = gpuCreate(BT * dim)
  let dMixBuf = gpuCreate(BT * dim)

  # CPU-side arrays
  var tokData = newSeq[int32](BT)
  var tgtData = newSeq[int32](BT)
  var lossData = newSeq[float32](BT)

  let t0 = epochTime()

  for step in 0..<steps:
    # Sample random sequences
    for b in 0..<batchSize:
      let start = rand(dataLen - seqLen - 2)
      for i in 0..<seqLen:
        tokData[b * seqLen + i] = data[start + i]
        tgtData[b * seqLen + i] = data[start + i + 1]
    tokBuf.gpuUploadInts(tokData)
    tgtBuf.gpuUploadInts(tgtData)

    # LR schedule: linear warmup
    let s = if step < 100: step.float32 / 100.0 else: 1.0
    let curLr = lr * s

    # === Forward ===

    # Embed: tokens → drives (BT × nOsc)
    gpu_embed_fwd(tokBuf.data, m.drive.w.data, driveBuf.data, nOsc.cint, BT.cint)

    # Initialize state: pad drives to dim (pos=drives, vel=0)
    # For now: just zero-pad. TODO: proper FFT bank encoding
    stateBuf.gpuZero()
    gpuCopy(driveBuf, stateBuf, BT * nOsc)  # copy drives into first nOsc dims

    # Per-position layers (no recurrence in this version)
    for l in 0..<m.nLayers:
      # RMSNorm
      gpu_rmsnorm(stateBuf.data, normBuf.data, dim.cint, BT.cint)
      # W_mix: (BT, dim) @ (dim, dim) → (BT, dim)
      gpuSgemm(opNN, BT, dim, dim, normBuf, m.layers[l].wMix.w, mixBuf)
      # SineGate
      gpu_sinegate_fwd(mixBuf.data, actBuf.data, (BT * dim).cint)
      # Residual: state += activated (using gpuCopy + we overwrite for now)
      # TODO: proper add kernel. For now just use actBuf as next state.
      gpuCopy(actBuf, stateBuf, BT * dim)

    # Output: RMSNorm → W_out → logits
    gpu_rmsnorm(stateBuf.data, normBuf.data, dim.cint, BT.cint)
    # logits(BT, vocab) = norm(BT, dim) @ W_out(dim, vocab)
    gpuSgemm(opNN, BT, vocab, dim, normBuf, m.wOut.w, logitBuf)

    # Loss
    gpu_ce_loss(logitBuf.data, tgtBuf.data, lossBuf.data, vocab.cint, BT.cint)
    let ld = gpuDownload(lossBuf)
    var totalLoss: float32 = 0
    for i in 0..<BT: totalLoss += ld[i]
    totalLoss /= BT.float32

    # === Backward ===

    # dLogits from CE loss
    gpu_ce_backward(logitBuf.data, tgtBuf.data, dLogitBuf.data, vocab.cint, BT.cint)

    # dNorm = dLogits @ W_out^T : (BT, vocab) @ (vocab, dim) = (BT, dim)
    gpuSgemm(opNT, BT, dim, vocab, dLogitBuf, m.wOut.w, dStateBuf)

    # dW_out += norm^T @ dLogits : (dim, BT) @ (BT, vocab) = (dim, vocab)
    gpuSgemm(opTNA, dim, vocab, BT, normBuf, dLogitBuf, m.wOut.g)

    # Backward through layers (reverse)
    for l in countdown(m.nLayers - 1, 0):
      # dSineGate: d(x*sin(x))/dx = sin(x) + x*cos(x)
      gpu_sinegate_bwd(mixBuf.data, dStateBuf.data, dMixBuf.data, (BT * dim).cint)

      # dW_mix += norm^T @ dMix : (dim, BT) @ (BT, dim) = (dim, dim)
      gpuSgemm(opTNA, dim, dim, BT, normBuf, dMixBuf, m.layers[l].wMix.g)

      # dNorm = dMix @ W_mix^T : (BT, dim) @ (dim, dim) = (BT, dim)
      gpuSgemm(opNT, BT, dim, dim, dMixBuf, m.layers[l].wMix.w, dStateBuf)

      # TODO: backward through RMSNorm (skip for now — approximate as identity)

    # dDrive from embedding backward
    gpu_embed_bwd(tokBuf.data, dStateBuf.data, m.drive.g.data, nOsc.cint, BT.cint)

    # Adam step
    let b1: float32 = 0.9
    let b2: float32 = 0.999
    let wd: float32 = 0.01
    let t = (step + 1).float32
    let bc1 = 1.0 / (1.0 - pow(b1, t))
    let bc2 = 1.0 / (1.0 - pow(b2, t))

    m.drive.adamStep(curLr * 3.0, b1, b2, 0, bc1, bc2)  # higher LR for embeddings
    m.wOut.adamStep(curLr, b1, b2, wd, bc1, bc2)
    for l in 0..<m.nLayers:
      m.layers[l].wMix.adamStep(curLr, b1, b2, wd, bc1, bc2)

    sync()

    if step mod 100 == 0:
      let elapsed = epochTime() - t0
      let bpc = totalLoss / ln(2.0)
      let stepsPerSec = if elapsed > 0: (step + 1).float / elapsed else: 0.0
      echo &"step {step:5d}  loss {totalLoss:.3f}  bpc {bpc:.3f}  lr {curLr:.1e}  [{stepsPerSec:.1f} steps/s]"

  echo "\nDone."

proc main() =
  randomize()
  gpuInit()

  let dataPath = if paramCount() >= 1: paramStr(1) else: "data/shakespeare.txt"
  let nOsc = if paramCount() >= 2: parseInt(paramStr(2)) else: 96
  let nLayers = if paramCount() >= 3: parseInt(paramStr(3)) else: 6
  let steps = if paramCount() >= 4: parseInt(paramStr(4)) else: 3000

  let data = loadText(dataPath)
  echo &"Data: {data.len} bytes from {dataPath}"

  var m = createModel(nOsc, nLayers, vocabSize = 256, seqLen = 128)
  train(m, data, steps, seqLen = 128, batchSize = 32, lr = 3e-4)

main()
