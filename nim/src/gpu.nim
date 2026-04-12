## GPU memory management and CUDA bindings for Resonance.
## Borrowed patterns from nimllm, stripped to what we need.

{.passL: "src/kernels.o -lcudart -lcublas -lcufft -lstdc++".}

# --- CUDA runtime ---
proc cudaMalloc(p: ptr pointer, size: csize_t): cint {.importc, header: "<cuda_runtime.h>".}
proc cudaFree(p: pointer): cint {.importc, header: "<cuda_runtime.h>".}
proc cudaMemcpy*(dst, src: pointer, size: csize_t, kind: cint): cint {.importc, header: "<cuda_runtime.h>".}
proc cudaMemset*(p: pointer, value: cint, size: csize_t): cint {.importc, header: "<cuda_runtime.h>".}
proc cudaDeviceSynchronize*(): cint {.importc, header: "<cuda_runtime.h>".}

const cudaMemcpyDeviceToDevice* = 3.cint

const
  cudaMemcpyHostToDevice = 1.cint
  cudaMemcpyDeviceToHost = 2.cint

# --- cuBLAS ---
type CublasHandle = pointer
proc cublasCreate_v2(h: ptr CublasHandle): cint {.importc, header: "<cublas_v2.h>".}
proc cublasSgemm_v2(h: CublasHandle, ta, tb: cint,
    m, n, k: cint, alpha: ptr cfloat, A: pointer, lda: cint,
    B: pointer, ldb: cint, beta: ptr cfloat, C: pointer, ldc: cint): cint {.importc, header: "<cublas_v2.h>".}

const
  CUBLAS_OP_N = 0.cint
  CUBLAS_OP_T = 1.cint

var cublasH*: CublasHandle

proc initCublas*() =
  discard cublasCreate_v2(addr cublasH)

# --- GPU buffer ---
type GpuBuf* = object
  data*: pointer
  len*: int  # number of float32s

proc gpuAlloc*(n: int): GpuBuf =
  result.len = n
  discard cudaMalloc(addr result.data, csize_t(n * 4))
  discard cudaMemset(result.data, 0, csize_t(n * 4))

proc gpuFree*(b: GpuBuf) =
  discard cudaFree(b.data)

proc upload*(b: GpuBuf, data: openArray[float32]) =
  assert data.len <= b.len
  discard cudaMemcpy(b.data, unsafeAddr data[0], csize_t(data.len * 4), cudaMemcpyHostToDevice)

proc download*(b: GpuBuf, data: var openArray[float32]) =
  assert data.len <= b.len
  discard cudaMemcpy(addr data[0], b.data, csize_t(data.len * 4), cudaMemcpyDeviceToHost)

proc uploadInts*(b: GpuBuf, data: openArray[int32]) =
  assert data.len <= b.len
  discard cudaMemcpy(b.data, unsafeAddr data[0], csize_t(data.len * 4), cudaMemcpyHostToDevice)

proc sync*() =
  discard cudaDeviceSynchronize()

# --- cuBLAS wrappers ---

proc gpuMatmul*(C, A, B: GpuBuf, m, n, k: int, beta: float32 = 0.0) =
  ## C[m,n] = A[m,k] @ B[k,n]  (row-major, but cuBLAS is col-major so we transpose)
  var alpha: cfloat = 1.0
  var betaC: cfloat = beta
  discard cublasSgemm_v2(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
    n.cint, m.cint, k.cint, addr alpha,
    B.data, n.cint, A.data, k.cint, addr betaC, C.data, n.cint)

proc gpuMatmulT*(C, A, B: GpuBuf, m, n, k: int, beta: float32 = 0.0) =
  ## C[m,n] = A[m,k] @ B^T[n,k]  (B transposed)
  var alpha: cfloat = 1.0
  var betaC: cfloat = beta
  discard cublasSgemm_v2(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
    n.cint, m.cint, k.cint, addr alpha,
    B.data, k.cint, A.data, k.cint, addr betaC, C.data, n.cint)

# --- Kernel bindings ---

proc gpu_embed_fwd*(ids: pointer, table, output: pointer, dim, n: cint) {.importc, cdecl.}
proc gpu_embed_bwd*(ids: pointer, dout, dtable: pointer, dim, n: cint) {.importc, cdecl.}
proc gpu_rmsnorm*(x, output: pointer, dim, n: cint) {.importc, cdecl.}
proc gpu_sinegate_fwd*(x, output: pointer, n: cint) {.importc, cdecl.}
proc gpu_sinegate_bwd*(x, dout, dx: pointer, n: cint) {.importc, cdecl.}
proc gpu_rotation_step*(pos, vel, gamma, beta, drive: pointer,
    cos_w, sin_w, freqs: pointer, B, n_osc: cint) {.importc, cdecl.}
proc gpu_rotation_step_bwd*(dpos_out, dvel_out, pos_prev, vel_prev: pointer,
    gamma, beta, drive: pointer, cos_w, sin_w, freqs: pointer,
    dpos_prev, dvel_prev, dgamma, dbeta, ddrive: pointer,
    B, n_osc: cint) {.importc, cdecl.}
proc gpu_sigmoid*(x, output: pointer, n: cint) {.importc, cdecl.}
proc gpu_sigmoid_bwd*(sig_out, dout, dx: pointer, n: cint) {.importc, cdecl.}
proc gpu_ce_loss*(logits: pointer, targets: pointer, losses: pointer,
    vocab, n: cint) {.importc, cdecl.}
proc gpu_ce_backward*(logits: pointer, targets: pointer, dlogits: pointer,
    vocab, n: cint) {.importc, cdecl.}
proc gpu_adamw*(param, grad, m, v: pointer,
    lr, b1, b2, wd, bc1, bc2: cfloat, n: cint) {.importc, cdecl.}
proc gpu_grad_norm*(grad: pointer, output: pointer, n: cint) {.importc, cdecl.}
proc gpu_scale*(x: pointer, s: cfloat, n: cint) {.importc, cdecl.}
