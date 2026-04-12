## dsp.nim — Signal processing primitives for Resonance.
##
## Inspired by aitherNim: every operation is a signal transform.
## Chain with |> for left-to-right data flow:
##
##   normed |> project(layer.gamma) |> sigmoid |> gate
##
## Each function takes a Signal (GPU buffer + shape) and returns a Signal.
## Allocation is explicit — no hidden mallocs in the signal chain.

import gpu

type
  Signal* = object
    ## A signal on the GPU: a buffer with known shape.
    buf*: GpuBuf
    rows*: int    ## batch × seq_len (or batch for per-step ops)
    cols*: int    ## feature dimension

  Projection* = object
    ## A learned linear projection: weight + optional bias.
    w*: GpuBuf
    bias*: GpuBuf
    hasBias*: bool
    inDim*, outDim*: int

  Interference* = object
    ## Dense W for pairwise cross-terms.
    w*: GpuBuf
    dim*: int

# --- Signal constructors ---

proc signal*(buf: GpuBuf, rows, cols: int): Signal =
  Signal(buf: buf, rows: rows, cols: cols)

proc numel*(s: Signal): int = s.rows * s.cols

# --- Core operations ---

proc rmsNorm*(input: Signal, output: Signal): Signal =
  ## RMSNorm across feature dimension.
  gpu_rmsnorm(input.buf.data, output.buf.data, input.cols.cint, input.rows.cint)
  result = output

proc project*(input: Signal, proj: Projection, output: Signal): Signal =
  ## Linear projection: output = input @ W + bias
  gpuSgemm(0, input.rows, proj.outDim, proj.inDim, input.buf, proj.w, output.buf)
  if proj.hasBias:
    gpu_add_bias(output.buf.data, proj.bias.data, proj.outDim.cint, (input.rows * proj.outDim).cint)
  result = output

proc sigmoid*(input: Signal): Signal =
  ## In-place sigmoid.
  gpu_sigmoid(input.buf.data, input.buf.data, input.numel.cint)
  result = input

proc sineGate*(input: Signal, output: Signal): Signal =
  ## x * sin(x) — harmonic nonlinearity.
  gpu_sinegate_fwd(input.buf.data, output.buf.data, input.numel.cint)
  result = output

proc interfere*(input: Signal, w: Interference, output: Signal): Signal =
  ## Dense W: compute all pairwise interference cross-terms.
  gpuSgemm(0, input.rows, w.dim, w.dim, input.buf, w.w, output.buf)
  result = output

proc superpose*(a, b: Signal, output: Signal): Signal =
  ## Residual add: superposition of two signals.
  gpu_add(a.buf.data, b.buf.data, output.buf.data, a.numel.cint)
  result = output

proc scale*(input: Signal, factor: float32): Signal =
  ## Scale signal amplitude.
  gpu_scale_array(input.buf.data, factor, input.numel.cint)
  result = input

# --- Pipe operator ---

template `|>`*(input: Signal, call: untyped): Signal =
  ## Pipe operator for signal chains.
  call
