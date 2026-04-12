## gpu.nim — CUDA bindings for Resonance.
## Based on nimllm/gpu.nim patterns.

{.passL: "src/kernels.o -lcudart -lcublas -lcufft -lstdc++".}

# ── CUDA types ────────────────────────────────────────────────────

type
  CudaError* {.importc: "cudaError_t", header: "<cuda_runtime.h>".} = cint
  CublasHandle* {.importc: "cublasHandle_t", header: "<cublas_v2.h>".} = pointer
  CublasStatus* {.importc: "cublasStatus_t", header: "<cublas_v2.h>".} = cint
  CublasOperation* {.importc: "cublasOperation_t", header: "<cublas_v2.h>".} = cint

const
  CudaSuccess* = 0.CudaError
  CublasOpN* = 0.CublasOperation
  CublasOpT* = 1.CublasOperation

# ── CUDA runtime ──────────────────────────────────────────────────

proc cudaMalloc*(devPtr: ptr pointer, size: csize_t): CudaError
  {.importc, header: "<cuda_runtime.h>".}
proc cudaFree*(devPtr: pointer): CudaError
  {.importc, header: "<cuda_runtime.h>".}
proc cudaMemcpy*(dst, src: pointer, count: csize_t, kind: cint): CudaError
  {.importc, header: "<cuda_runtime.h>".}
proc cudaMemset*(devPtr: pointer, value: cint, count: csize_t): CudaError
  {.importc, header: "<cuda_runtime.h>".}
proc cudaDeviceSynchronize*(): CudaError
  {.importc, header: "<cuda_runtime.h>".}

const
  CudaMemcpyHostToDevice* = 1.cint
  CudaMemcpyDeviceToHost* = 2.cint
  CudaMemcpyDeviceToDevice* = 3.cint

# ── cuBLAS ────────────────────────────────────────────────────────

proc cublasCreate*(handle: ptr CublasHandle): CublasStatus
  {.importc: "cublasCreate_v2", header: "<cublas_v2.h>".}
proc cublasSgemm*(handle: CublasHandle, transa, transb: CublasOperation,
    m, n, k: cint, alpha: ptr cfloat, a: pointer, lda: cint,
    b: pointer, ldb: cint, beta: ptr cfloat, c: pointer, ldc: cint): CublasStatus
  {.importc: "cublasSgemm_v2", header: "<cublas_v2.h>".}
proc cublasSgemmStridedBatched*(handle: CublasHandle,
    transa, transb: CublasOperation,
    m, n, k: cint, alpha: ptr cfloat,
    a: pointer, lda: cint, strideA: clonglong,
    b: pointer, ldb: cint, strideB: clonglong,
    beta: ptr cfloat, c: pointer, ldc: cint, strideC: clonglong,
    batchCount: cint): CublasStatus
  {.importc: "cublasSgemmStridedBatched", header: "<cublas_v2.h>".}

# ── GPU Buffer ────────────────────────────────────────────────────

type
  GpuBuf* = object
    data*: pointer
    numel*: int

proc gpuCreate*(n: int): GpuBuf =
  result.numel = n
  let err = cudaMalloc(addr result.data, csize_t(n * sizeof(cfloat)))
  assert err == CudaSuccess, "cudaMalloc failed"
  discard cudaMemset(result.data, 0, csize_t(n * sizeof(cfloat)))

proc gpuFree*(buf: var GpuBuf) =
  if buf.data != nil:
    discard cudaFree(buf.data)
    buf.data = nil; buf.numel = 0

proc gpuUpload*(buf: GpuBuf, hostData: openArray[float32]) =
  assert hostData.len == buf.numel
  discard cudaMemcpy(buf.data, unsafeAddr hostData[0],
                     csize_t(buf.numel * sizeof(cfloat)), CudaMemcpyHostToDevice)

proc gpuDownload*(buf: GpuBuf): seq[float32] =
  result = newSeq[float32](buf.numel)
  discard cudaMemcpy(addr result[0], buf.data,
                     csize_t(buf.numel * sizeof(cfloat)), CudaMemcpyDeviceToHost)

proc gpuCopy*(src, dst: GpuBuf, n: int) =
  discard cudaMemcpy(dst.data, src.data,
                     csize_t(n * sizeof(cfloat)), CudaMemcpyDeviceToDevice)

proc gpuZero*(buf: GpuBuf) =
  discard cudaMemset(buf.data, 0, csize_t(buf.numel * sizeof(cfloat)))

proc toGpu*(data: openArray[float32]): GpuBuf =
  result = gpuCreate(data.len)
  gpuUpload(result, data)

proc gpuUploadInts*(buf: GpuBuf, data: openArray[int32]) =
  assert data.len <= buf.numel
  discard cudaMemcpy(buf.data, unsafeAddr data[0],
                     csize_t(data.len * sizeof(cint)), CudaMemcpyHostToDevice)

# ── Global cuBLAS handle ─────────────────────────────────────────

var gCublas: CublasHandle

proc gpuInit*() =
  let err = cublasCreate(addr gCublas)
  assert err == 0, "cublasCreate failed"

# ── GEMM wrapper (from nimllm) ───────────────────────────────────
## 3-bit op encoding:
##   bit 2 (4): transpose A
##   bit 1 (2): transpose B
##   bit 0 (1): accumulate (beta=1) vs overwrite (beta=0)

proc gpuSgemm*(op: int, m, n, k: int, a, b, c: GpuBuf) =
  var alpha: cfloat = 1.0
  var beta: cfloat = if (op and 1) != 0: 1.0 else: 0.0
  let ta = (op and 4) != 0
  let tb = (op and 2) != 0
  let opB = if tb: CublasOpT else: CublasOpN
  let opA = if ta: CublasOpT else: CublasOpN
  let lda = if ta: cint(m) else: cint(k)
  let ldb = if tb: cint(k) else: cint(n)
  let ldc = cint(n)
  discard cublasSgemm(gCublas, opB, opA, cint(n), cint(m), cint(k),
                      addr alpha, b.data, ldb, a.data, lda,
                      addr beta, c.data, ldc)

proc sync*() =
  discard cudaDeviceSynchronize()

proc gpuMonarchGrad*(dOutput: GpuBuf, input: GpuBuf, dFactor: GpuBuf,
                     nBlocks, blockSize, batchRows: int) =
  ## Accumulate block-diagonal gradient: dFactor[block] += sum_row dOutput[row,block_slice]^T @ input[row,block_slice]
  ## dFactor: (nBlocks, blockSize, blockSize)
  ## dOutput, input: (batchRows, nBlocks * blockSize)
  var alpha: cfloat = 1.0
  var beta: cfloat = 1.0  # accumulate
  let s = blockSize.cint
  let n = batchRows.cint
  let stride = (nBlocks * blockSize).cint
  # For each block b: dFactor[b] += dOutput[:, b*s:(b+1)*s]^T @ input[:, b*s:(b+1)*s]
  # Col-major: dFactor[b] = input_chunk^T @ dOutput_chunk (swapped for row-major)
  discard cublasSgemmStridedBatched(gCublas,
    CublasOpN, CublasOpT,  # col-major: compute input^T @ dOutput
    s, s, n,                # m=blockSize, n=blockSize, k=batchRows
    addr alpha,
    dOutput.data, stride, clonglong(s),  # A = dOutput chunks
    input.data, stride, clonglong(s),    # B = input chunks
    addr beta,
    dFactor.data, s, clonglong(s * s),   # C = dFactor blocks (accumulated)
    nBlocks.cint)

proc gpuMonarchMul*(factor: GpuBuf, input: GpuBuf, output: GpuBuf,
                    nBlocks, blockSize, batchRows: int, accumulate: bool = false) =
  ## Block-diagonal matmul: output[row, block*blockSize + j] = sum_k factor[block, j, k] * input[row, block*blockSize + k]
  ## factor: (nBlocks, blockSize, blockSize) — the block-diagonal weights
  ## input: (batchRows, nBlocks * blockSize) — viewed as (batchRows, nBlocks, blockSize)
  ## output: same shape
  ## Uses cublasSgemmStridedBatched for one-call execution.
  var alpha: cfloat = 1.0
  var beta: cfloat = if accumulate: 1.0 else: 0.0
  let s = blockSize.cint
  let n = batchRows.cint
  # Row-major → col-major trick: swap A/B
  # We want: for each block b: out[:, b*s:(b+1)*s] = input[:, b*s:(b+1)*s] @ factor[b]^T
  # Col-major: C = B^T @ A where B = factor[b], A = input chunk
  discard cublasSgemmStridedBatched(gCublas,
    CublasOpT, CublasOpN,  # transA=T (factor transposed), transB=N
    s, n, s,                 # m=blockSize, n=batchRows, k=blockSize
    addr alpha,
    factor.data, s, clonglong(s * s),     # A = factor blocks, stride = blockSize²
    input.data, (nBlocks * blockSize).cint, clonglong(s),  # B = input, stride between blocks = blockSize
    addr beta,
    output.data, (nBlocks * blockSize).cint, clonglong(s), # C = output
    nBlocks.cint)

# ── Kernel bindings ───────────────────────────────────────────────

proc gpu_embed_fwd*(ids, table, output: pointer, dim, n: cint) {.importc, cdecl.}
proc gpu_embed_bwd*(ids, dout, dtable: pointer, dim, n: cint) {.importc, cdecl.}
proc gpu_rmsnorm*(x, output: pointer, dim, n: cint) {.importc, cdecl.}
proc gpu_rmsnorm_bwd*(x, dout, dx: pointer, dim, n: cint) {.importc, cdecl.}
proc gpu_add*(a, b, y: pointer, n: cint) {.importc, cdecl.}
proc gpu_add_inplace*(a, b: pointer, n: cint) {.importc, cdecl.}
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
proc gpu_ce_loss*(logits, targets, losses: pointer, vocab, n: cint) {.importc, cdecl.}
proc gpu_ce_backward*(logits, targets, dlogits: pointer, vocab, n: cint) {.importc, cdecl.}
proc gpu_adamw*(param, grad, m, v: pointer,
    lr, b1, b2, wd, bc1, bc2: cfloat, n: cint) {.importc, cdecl.}
proc gpu_grad_norm*(grad, output: pointer, n: cint) {.importc, cdecl.}
proc gpu_scale*(x: pointer, s: cfloat, n: cint) {.importc, cdecl.}
proc gpu_complex_mul*(a, b, output: pointer, n: cint) {.importc, cdecl.}
proc gpu_complex_mul_conj*(a, b, output: pointer, n: cint) {.importc, cdecl.}
proc gpu_extract_column*(src, dst: pointer, col, n_cols, n_rows: cint) {.importc, cdecl.}
proc gpu_scatter_to_state*(col, state: pointer, c, dim, n_rows: cint) {.importc, cdecl.}
proc gpu_scale_array*(x: pointer, s: cfloat, n: cint) {.importc, cdecl.}
proc gpu_fft_r2c_batched*(input, output: pointer, fft_len, batch: cint): cint {.importc, cdecl.}
proc gpu_fft_c2r_batched*(input, output: pointer, fft_len, batch: cint): cint {.importc, cdecl.}
proc gpu_complex_mul_broadcast*(a, b, output: pointer, stride, n: cint) {.importc, cdecl.}
proc gpu_complex_mul_conj_broadcast*(a, b, output: pointer, stride, n: cint) {.importc, cdecl.}
proc gpu_bank_scatter*(conv, state: pointer,
    batchSize, seqLen, n_osc, dim, col_offset: cint, inv_fft: cfloat) {.importc, cdecl.}
proc gpu_bank_gather*(drives, columns: pointer,
    batchSize, seqLen, n_osc, fft_len: cint) {.importc, cdecl.}
proc gpu_strided_gather*(src, dst: pointer,
    rows, stride, n_osc, col_offset: cint) {.importc, cdecl.}
proc gpu_add_bias*(x, bias: pointer, cols, n: cint) {.importc, cdecl.}
proc gpu_bias_grad*(dout, dbias: pointer, cols, rows: cint) {.importc, cdecl.}

proc gpu_spectral_mul_gated*(spec_in, spec_out: pointer,
    wRe, wIm, gate: pointer, specLen, batch: cint) {.importc, cdecl.}
proc gpu_spectral_mul_gated_bwd*(spec_in, d_spec_out: pointer,
    wRe, wIm, gate: pointer,
    d_spec_in, d_wRe, d_wIm, d_gate: pointer,
    specLen, batch: cint) {.importc, cdecl.}

proc gpu_monarch_permute*(input, output: pointer,
    nBlocks, blockSize, rows: cint) {.importc, cdecl.}
proc gpu_monarch_permute_inv*(input, output: pointer,
    nBlocks, blockSize, rows: cint) {.importc, cdecl.}
proc gpu_rotation_scan*(gamma, beta, sense, bank_out, osc_out: pointer,
    cos_w, sin_w, freqs, fm_depth: pointer,
    batch_size, seq_len, n_osc, fold_offset: cint) {.importc, cdecl.}
proc gpu_rotation_scan_bwd*(gamma, beta, sense, bank_out, osc_out, d_osc_out: pointer,
    d_gamma, d_beta, d_sense, d_bank_out: pointer,
    cos_w, sin_w, freqs: pointer,
    batch_size, seq_len, n_osc: cint) {.importc, cdecl.}
