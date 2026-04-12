## dsp.nim — Signal processing for Resonance.
##
## Inspired by aitherNim: every operation is a named signal transform.
## Uses Nim's UFCS for left-to-right signal chains:
##
##   drives.propagate(bank, state)
##   state.normalize(normed)
##   normed.project(w, bias, gate).activate(gate)
##   bank.resonate(gamma, beta, sense, oscOut)
##   oscOut.interfere(w, mixed).harmonics(activated)
##   state.superpose(activated, nextState)
##
## No hidden allocations. Every buffer is pre-allocated.
## This is the aither way: explicit state, pipeline flow.

import std/macros
import gpu

# ── Pipe operator (from aitherNim) ──

macro `|>`*(lhs: typed, rhs: untyped): untyped =
  ## Insert lhs as the first argument of rhs.
  ## `x |> f(y, z)` becomes `f(x, y, z)`
  result = rhs.copyNimTree()
  result.insert(1, lhs)

type
  Buf* = GpuBuf  ## Alias — a signal is just a GPU buffer with a name

# ── Propagation: FFT bank encoding ──

proc propagate*(drives, columns, colsFft, prodPos, prodVel, convPos, convVel: Buf,
                hPosFft, hVelFft: Buf, state: Buf,
                batchSize, seqLen, nOsc, dim, fftLen, fftCLen, totalOsc: int) =
  ## Token drives → FFT convolve with oscillator impulse responses → state
  ## The bank IS the physics: h(t) = e^{-γt} sin(ωd·t) / ωd
  let inv = 1.0 / fftLen.float32
  state.gpuZero()
  columns.gpuZero()
  gpu_bank_gather(drives.data, columns.data,
    batchSize.cint, seqLen.cint, nOsc.cint, fftLen.cint)
  discard gpu_fft_r2c_batched(columns.data, colsFft.data, fftLen.cint, totalOsc.cint)
  gpu_complex_mul_broadcast(colsFft.data, hPosFft.data, prodPos.data,
    (nOsc * fftCLen).cint, (totalOsc * fftCLen).cint)
  gpu_complex_mul_broadcast(colsFft.data, hVelFft.data, prodVel.data,
    (nOsc * fftCLen).cint, (totalOsc * fftCLen).cint)
  discard gpu_fft_c2r_batched(prodPos.data, convPos.data, fftLen.cint, totalOsc.cint)
  discard gpu_fft_c2r_batched(prodVel.data, convVel.data, fftLen.cint, totalOsc.cint)
  gpu_bank_scatter(convPos.data, state.data,
    batchSize.cint, seqLen.cint, nOsc.cint, dim.cint, 0.cint, inv.cfloat)
  gpu_bank_scatter(convVel.data, state.data,
    batchSize.cint, seqLen.cint, nOsc.cint, dim.cint, nOsc.cint, inv.cfloat)

# ── Normalization ──

proc normalize*(input, output: Buf, dim, rows: int) =
  ## RMSNorm: scale to unit energy
  gpu_rmsnorm(input.data, output.data, dim.cint, rows.cint)

# ── Projection + activation ──

proc project*(input, weights, output: Buf, rows, outDim, inDim: int) =
  ## Linear projection: output = input @ weights
  gpuSgemm(0, rows, outDim, inDim, input, weights, output)

proc addBias*(signal: Buf, bias: Buf, cols, n: int): Buf {.discardable.} =
  ## Add per-column bias
  gpu_add_bias(signal.data, bias.data, cols.cint, n.cint)
  result = signal

proc activate*(signal: Buf, n: int): Buf {.discardable.} =
  ## Sigmoid: squash to (0, 1) — the gate opens/closes
  gpu_sigmoid(signal.data, signal.data, n.cint)
  result = signal

# ── Resonance: damped rotation ──

proc resonate*(gamma, beta, sense, bankOut, oscOut: Buf,
               cosW, sinW, freqs: Buf,
               batchSize, seqLen, nOsc: int) =
  ## Damped rotation scan: the oscillator equation
  ## pos' = γ·(pos·cos(ω) + vel·sin(ω)/ω) + (1-γ)·β·drive
  ## vel' = γ·(vel·cos(ω) - pos·ω·sin(ω)) + (1-γ)·β·drive
  gpu_rotation_scan(gamma.data, beta.data, sense.data,
    bankOut.data, oscOut.data,
    cosW.data, sinW.data, freqs.data,
    batchSize.cint, seqLen.cint, nOsc.cint)

# ── Interference: pairwise cross-terms ──

proc interfere*(input, weights, output: Buf, rows, dim: int) =
  ## Dense W: |A + B|² = |A|² + |B|² + 2·Re(A·conj(B))
  ## The cross-terms ARE the pairwise interactions
  gpuSgemm(0, rows, dim, dim, input, weights, output)

# ── Harmonics: wave nonlinearity ──

proc harmonics*(input, output: Buf, n: int) =
  ## SineGate: x × sin(x) — generates overtones like an overdriven oscillator
  gpu_sinegate_fwd(input.data, output.data, n.cint)

# ── Superposition: residual add ──

proc superpose*(a, b, output: Buf, n: int) =
  ## Waves add: the principle of superposition
  gpu_add(a.data, b.data, output.data, n.cint)

# ── Strike: token → oscillator excitation ──

proc strike*(tokens, driveTable, drives: Buf, dim, n: int) =
  ## Each token strikes the oscillator bank like a hammer hitting bells
  gpu_embed_fwd(tokens.data, driveTable.data, drives.data, dim.cint, n.cint)

# ── Listen: oscillator state → prediction ──

proc listen*(state, weights, logits: Buf, rows, vocabSize, dim: int) =
  ## Dot product with drive signatures — weight-tied readout
  ## The same resonance patterns used to strike are used to listen
  gpuSgemm(0, rows, vocabSize, dim, state, weights, logits)
