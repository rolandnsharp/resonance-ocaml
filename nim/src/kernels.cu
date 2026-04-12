/*
 * Resonance — CUDA kernels for damped rotation neural network.
 *
 * Core operations:
 *   - Oscillator bank: FFT convolution via cuFFT
 *   - Damped rotation: selective second-order recurrence
 *   - SineGate: x * sin(x) activation
 *   - Standard: RMSNorm, embedding, cross-entropy, AdamW
 */

#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>

#define BLOCK 256

/* ---- Embedding ---- */

__global__ void k_embed_fwd(const int* ids, const float* table,
                            float* out, int dim, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * dim) return;
    int tok = i / dim, d = i % dim;
    out[i] = table[ids[tok] * dim + d];
}

__global__ void k_embed_bwd(const int* ids, const float* dout,
                            float* dtable, int dim, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * dim) return;
    int tok = i / dim, d = i % dim;
    atomicAdd(&dtable[ids[tok] * dim + d], dout[i]);
}

/* ---- RMSNorm ---- */

__global__ void k_rmsnorm_fwd(const float* x, float* out, int dim, int n) {
    int row = blockIdx.x;
    if (row >= n) return;
    const float* xr = x + row * dim;
    float* or_ = out + row * dim;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        ss += xr[i] * xr[i];
    // Warp reduce
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        ss += __shfl_down_sync(0xffffffff, ss, offset);
    __shared__ float shared_ss;
    if (threadIdx.x == 0) shared_ss = ss;
    __syncthreads();
    float inv = rsqrtf(shared_ss / dim + 1e-8f);
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        or_[i] = xr[i] * inv;
}

/* ---- SineGate: x * sin(x) ---- */

__global__ void k_sinegate_fwd(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = x[i] * sinf(x[i]);
}

__global__ void k_sinegate_bwd(const float* x, const float* dout,
                               float* dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // d(x*sin(x))/dx = sin(x) + x*cos(x)
    dx[i] = dout[i] * (sinf(x[i]) + x[i] * cosf(x[i]));
}

/* ---- Damped Rotation Step (forward) ----
   The core equation per oscillator:
     pos' = gamma * (pos*cos_w + vel*sin_w/freq) + (1-gamma)*beta*drive_pos
     vel' = gamma * (vel*cos_w - pos*freq*sin_w) + (1-gamma)*beta*drive_vel

   Layout: pos[B*n_osc], vel[B*n_osc] — one batch of oscillator states
   gamma[B*n_osc], beta[B*n_osc] — input-dependent controls
   drive[B*2*n_osc] — bank output (pos concat vel)
   cos_w[n_osc], sin_w[n_osc], freqs[n_osc] — precomputed rotation constants
*/

__global__ void k_rotation_step(
    float* pos, float* vel,           // in/out: [B, n_osc]
    const float* gamma_,              // [B, n_osc]
    const float* beta,                // [B, n_osc]
    const float* drive,               // [B, 2*n_osc]
    const float* cos_w,               // [n_osc]
    const float* sin_w,               // [n_osc]
    const float* freqs,               // [n_osc]
    int B, int n_osc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * n_osc) return;
    int k = i % n_osc;
    float g = gamma_[i];
    float b = beta[i];
    float p = pos[i], v = vel[i];
    float cw = cos_w[k], sw = sin_w[k], f = freqs[k];
    float dp = drive[i / n_osc * 2 * n_osc + k];          // drive pos
    float dv = drive[i / n_osc * 2 * n_osc + n_osc + k];  // drive vel

    pos[i] = g * (p * cw + v * sw / f) + (1.0f - g) * b * dp;
    vel[i] = g * (v * cw - p * f * sw) + (1.0f - g) * b * dv;
}

/* ---- Damped Rotation Step (backward) ----
   Given dpos', dvel' (upstream gradients), compute:
   - dpos, dvel (gradient w.r.t. previous state)
   - dgamma, dbeta (gradient w.r.t. controls)
   - ddrive (gradient w.r.t. drive input)
*/

__global__ void k_rotation_step_bwd(
    const float* dpos_out, const float* dvel_out,  // upstream: [B, n_osc]
    const float* pos_prev, const float* vel_prev,  // saved state before step
    const float* gamma_, const float* beta,
    const float* drive,
    const float* cos_w, const float* sin_w, const float* freqs,
    float* dpos_prev, float* dvel_prev,            // output: [B, n_osc]
    float* dgamma, float* dbeta,                   // output: [B, n_osc]
    float* ddrive,                                  // output: [B, 2*n_osc]
    int B, int n_osc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * n_osc) return;
    int k = i % n_osc;
    int b_idx = i / n_osc;
    float g = gamma_[i];
    float bt = beta[i];
    float p = pos_prev[i], v = vel_prev[i];
    float cw = cos_w[k], sw = sin_w[k], f = freqs[k];
    float dp_out = dpos_out[i], dv_out = dvel_out[i];
    int drive_base = b_idx * 2 * n_osc;
    float drv_p = drive[drive_base + k];
    float drv_v = drive[drive_base + n_osc + k];

    // d/d(pos_prev): g * cos_w * dp_out + g * (-freq * sin_w) * dv_out
    dpos_prev[i] = g * cw * dp_out - g * f * sw * dv_out;
    // d/d(vel_prev): g * sin_w/freq * dp_out + g * cos_w * dv_out
    dvel_prev[i] = g * sw / f * dp_out + g * cw * dv_out;

    // d/d(gamma): [rotated_pos - (1-g)*beta*drive_pos]/... simplified:
    float rot_p = p * cw + v * sw / f;
    float rot_v = v * cw - p * f * sw;
    dgamma[i] = dp_out * (rot_p - bt * drv_p) + dv_out * (rot_v - bt * drv_v);

    // d/d(beta): (1-g) * drive * d_out
    dbeta[i] = dp_out * (1.0f - g) * drv_p + dv_out * (1.0f - g) * drv_v;

    // d/d(drive): (1-g) * beta * d_out
    ddrive[drive_base + k] = (1.0f - g) * bt * dp_out;
    ddrive[drive_base + n_osc + k] = (1.0f - g) * bt * dv_out;
}

/* ---- FFT Bank: complex multiply in frequency domain ---- */

__global__ void k_complex_mul(const cufftComplex* a, const cufftComplex* b,
                               cufftComplex* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float ar = a[i].x, ai = a[i].y;
    float br = b[i].x, bi = b[i].y;
    out[i].x = ar * br - ai * bi;
    out[i].y = ar * bi + ai * br;
}

__global__ void k_complex_mul_conj(const cufftComplex* a, const cufftComplex* b,
                                    cufftComplex* out, int n) {
    // Multiply a by conj(b) — for backward (FFT adjoint)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float ar = a[i].x, ai = a[i].y;
    float br = b[i].x, bi = -b[i].y;  // conjugate
    out[i].x = ar * br - ai * bi;
    out[i].y = ar * bi + ai * br;
}

__global__ void k_extract_column(const float* src, float* dst,
                                  int col, int n_cols, int n_rows) {
    // Extract column 'col' from row-major matrix (n_rows, n_cols)
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    dst[row] = src[row * n_cols + col];
}

__global__ void k_scatter_to_state(const float* col_data, float* state,
                                    int col, int dim, int n_rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    state[row * dim + col] = col_data[row];
}

__global__ void k_scale_array(float* x, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] *= s;
}

extern "C" void gpu_complex_mul(const void* a, const void* b, void* out, int n) {
    k_complex_mul<<<(n+BLOCK-1)/BLOCK, BLOCK>>>((const cufftComplex*)a,
        (const cufftComplex*)b, (cufftComplex*)out, n);
}
extern "C" void gpu_complex_mul_conj(const void* a, const void* b, void* out, int n) {
    k_complex_mul_conj<<<(n+BLOCK-1)/BLOCK, BLOCK>>>((const cufftComplex*)a,
        (const cufftComplex*)b, (cufftComplex*)out, n);
}
extern "C" void gpu_extract_column(const float* src, float* dst, int col, int n_cols, int n_rows) {
    k_extract_column<<<(n_rows+BLOCK-1)/BLOCK, BLOCK>>>(src, dst, col, n_cols, n_rows);
}
extern "C" void gpu_scatter_to_state(const float* col, float* state, int c, int dim, int n_rows) {
    k_scatter_to_state<<<(n_rows+BLOCK-1)/BLOCK, BLOCK>>>(col, state, c, dim, n_rows);
}
extern "C" void gpu_scale_array(float* x, float s, int n) {
    k_scale_array<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(x, s, n);
}

// Batched cuFFT — one call for all oscillators × batch elements
// Layout: data is (batch, fft_len) contiguous, batch = n_osc * batchSize
extern "C" int gpu_fft_r2c_batched(float* input, void* output, int fft_len, int batch) {
    cufftHandle plan;
    cufftPlan1d(&plan, fft_len, CUFFT_R2C, batch);
    int r = cufftExecR2C(plan, input, (cufftComplex*)output);
    cufftDestroy(plan);
    return r;
}
extern "C" int gpu_fft_c2r_batched(void* input, float* output, int fft_len, int batch) {
    cufftHandle plan;
    cufftPlan1d(&plan, fft_len, CUFFT_C2R, batch);
    int r = cufftExecC2R(plan, (cufftComplex*)input, output);
    cufftDestroy(plan);
    return r;
}

// Batched complex multiply: out[i] = a[i] * b[i % stride] (broadcasts b over batches)
__global__ void k_complex_mul_broadcast(const cufftComplex* a, const cufftComplex* b,
                                         cufftComplex* out, int stride, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int j = i % stride;  // which oscillator's kernel
    float ar = a[i].x, ai = a[i].y;
    float br = b[j].x, bi = b[j].y;
    out[i].x = ar * br - ai * bi;
    out[i].y = ar * bi + ai * br;
}

extern "C" void gpu_complex_mul_broadcast(const void* a, const void* b, void* out,
                                           int stride, int n) {
    k_complex_mul_broadcast<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(
        (const cufftComplex*)a, (const cufftComplex*)b, (cufftComplex*)out, stride, n);
}

// Broadcast multiply with conjugate: out[i] = a[i] * conj(b[i % stride])
__global__ void k_complex_mul_conj_broadcast(const cufftComplex* a, const cufftComplex* b,
                                              cufftComplex* out, int stride, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int j = i % stride;
    float ar = a[i].x, ai = a[i].y;
    float br = b[j].x, bi = -b[j].y;  // conjugate
    out[i].x = ar * br - ai * bi;
    out[i].y = ar * bi + ai * br;
}

extern "C" void gpu_complex_mul_conj_broadcast(const void* a, const void* b, void* out,
                                                int stride, int n) {
    k_complex_mul_conj_broadcast<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(
        (const cufftComplex*)a, (const cufftComplex*)b, (cufftComplex*)out, stride, n);
}

// Transpose + scatter: from (batch*n_osc, seqLen) to (batch, seqLen, dim)
// Each oscillator k's convolution result goes to column k (pos) or k+n_osc (vel)
__global__ void k_bank_scatter(const float* conv_results, float* state,
                                int batchSize, int seqLen, int n_osc, int dim,
                                int col_offset, float inv_fft) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * n_osc * seqLen;
    if (idx >= total) return;
    int b = idx / (n_osc * seqLen);
    int rem = idx % (n_osc * seqLen);
    int k = rem / seqLen;
    int t = rem % seqLen;
    // conv_results layout: (batchSize * n_osc, fft_len), take first seqLen
    int src_idx = (b * n_osc + k) * (seqLen * 2) + t;  // fft_len = 2*seqLen
    int dst_idx = (b * seqLen + t) * dim + k + col_offset;
    state[dst_idx] = conv_results[src_idx] * inv_fft;
}

extern "C" void gpu_bank_scatter(const float* conv, float* state,
                                  int batchSize, int seqLen, int n_osc, int dim,
                                  int col_offset, float inv_fft) {
    int n = batchSize * n_osc * seqLen;
    k_bank_scatter<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(conv, state,
        batchSize, seqLen, n_osc, dim, col_offset, inv_fft);
}

// Gather drives into per-oscillator columns: (batch, seqLen, n_osc) → (batch*n_osc, fft_len)
__global__ void k_bank_gather(const float* drives, float* columns,
                               int batchSize, int seqLen, int n_osc, int fft_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * n_osc * seqLen;
    if (idx >= total) return;
    int b = idx / (n_osc * seqLen);
    int rem = idx % (n_osc * seqLen);
    int k = rem / seqLen;
    int t = rem % seqLen;
    int src_idx = (b * seqLen + t) * n_osc + k;
    int dst_idx = (b * n_osc + k) * fft_len + t;
    columns[dst_idx] = drives[src_idx];
}

extern "C" void gpu_bank_gather(const float* drives, float* columns,
                                 int batchSize, int seqLen, int n_osc, int fft_len) {
    int n = batchSize * n_osc * seqLen;
    k_bank_gather<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(drives, columns,
        batchSize, seqLen, n_osc, fft_len);
}

// Strided gather: extract n_osc columns starting at col_offset from (rows, stride) matrix
__global__ void k_strided_gather(const float* src, float* dst,
                                  int rows, int stride, int n_osc, int col_offset, int fft_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * n_osc;
    if (idx >= total) return;
    int row = idx / n_osc;
    int k = idx % n_osc;
    // Zero-padded output: dst is (rows, fft_len), we write to position k within each row
    // But we want (n_osc_groups, fft_len) layout for batched FFT
    // Actually: group by oscillator across batch*seq. Need (batch*n_osc, fft_len) layout.
    // This is more complex. Let's just extract (rows, n_osc) contiguously.
    dst[row * n_osc + k] = src[row * stride + col_offset + k];
}

extern "C" void gpu_strided_gather(const float* src, float* dst,
                                    int rows, int stride, int n_osc, int col_offset) {
    int n = rows * n_osc;
    k_strided_gather<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(src, dst, rows, stride, n_osc, col_offset, 0);
}

/* ---- Add bias: each row gets the same bias vector added ---- */
__global__ void k_add_bias(float* x, const float* bias, int cols, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += bias[i % cols];
}

__global__ void k_bias_grad(const float* dout, float* dbias, int cols, int rows) {
    // Sum dout across rows to get dbias
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cols) return;
    float sum = 0.0f;
    for (int r = 0; r < rows; r++)
        sum += dout[r * cols + j];
    atomicAdd(&dbias[j], sum);
}

extern "C" void gpu_add_bias(float* x, const float* bias, int cols, int n) {
    k_add_bias<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(x, bias, cols, n);
}
extern "C" void gpu_bias_grad(const float* dout, float* dbias, int cols, int rows) {
    k_bias_grad<<<(cols+BLOCK-1)/BLOCK, BLOCK>>>(dout, dbias, cols, rows);
}

/* ---- Spectral mixing: batched FFT across dim, pointwise complex multiply, IFFT ---- */

// Complex pointwise multiply with learned weights + content gate
// spec_out[i] = spec_in[i] * (wRe[i] + j*wIm[i]) * gate[i]
__global__ void k_spectral_mul_gated(
    const cufftComplex* spec_in, cufftComplex* spec_out,
    const float* wRe, const float* wIm, const float* gate,
    int specLen, int batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * specLen) return;
    int row = idx / specLen;
    int k = idx % specLen;
    float ir = spec_in[idx].x, ii = spec_in[idx].y;
    float wr = wRe[k], wi = wIm[k];
    float g = gate[row * specLen + k];
    // Complex multiply then gate
    float or_ = (ir * wr - ii * wi) * g;
    float oi = (ir * wi + ii * wr) * g;
    spec_out[idx].x = or_;
    spec_out[idx].y = oi;
}

// Backward: gradient through spectral multiply
// d_spec_in, d_wRe, d_wIm, d_gate from d_spec_out
__global__ void k_spectral_mul_gated_bwd(
    const cufftComplex* spec_in, const cufftComplex* d_spec_out,
    const float* wRe, const float* wIm, const float* gate,
    cufftComplex* d_spec_in, float* d_wRe, float* d_wIm, float* d_gate,
    int specLen, int batch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * specLen) return;
    int row = idx / specLen;
    int k = idx % specLen;
    float ir = spec_in[idx].x, ii = spec_in[idx].y;
    float wr = wRe[k], wi = wIm[k];
    float g = gate[row * specLen + k];
    float dor = d_spec_out[idx].x, doi = d_spec_out[idx].y;

    // Through gate: d(x*g)/dg = x, d(x*g)/dx = g
    float pre_gate_r = ir * wr - ii * wi;
    float pre_gate_i = ir * wi + ii * wr;
    atomicAdd(&d_gate[row * specLen + k], dor * pre_gate_r + doi * pre_gate_i);
    float dgr = dor * g, dgi = doi * g;

    // Through complex multiply: d_spec_in and d_w
    d_spec_in[idx].x = dgr * wr + dgi * wi;
    d_spec_in[idx].y = -dgr * wi + dgi * wr;
    atomicAdd(&d_wRe[k], dgr * ir + dgi * ii);
    atomicAdd(&d_wIm[k], -dgr * ii + dgi * ir);
}

extern "C" void gpu_spectral_mul_gated(
    const void* spec_in, void* spec_out,
    const float* wRe, const float* wIm, const float* gate,
    int specLen, int batch) {
    int n = batch * specLen;
    k_spectral_mul_gated<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(
        (const cufftComplex*)spec_in, (cufftComplex*)spec_out,
        wRe, wIm, gate, specLen, batch);
}

extern "C" void gpu_spectral_mul_gated_bwd(
    const void* spec_in, const void* d_spec_out,
    const float* wRe, const float* wIm, const float* gate,
    void* d_spec_in, float* d_wRe, float* d_wIm, float* d_gate,
    int specLen, int batch) {
    int n = batch * specLen;
    k_spectral_mul_gated_bwd<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(
        (const cufftComplex*)spec_in, (const cufftComplex*)d_spec_out,
        wRe, wIm, gate, (cufftComplex*)d_spec_in, d_wRe, d_wIm, d_gate,
        specLen, batch);
}

/* ---- Monarch permutation: reshape(nBlocks, blockSize) → transpose → reshape(dim) ---- */

__global__ void k_monarch_permute(const float* in, float* out,
                                   int nBlocks, int blockSize, int rows) {
    // For each row (BT), permute dim elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * nBlocks * blockSize;
    if (idx >= total) return;
    int row = idx / (nBlocks * blockSize);
    int d = idx % (nBlocks * blockSize);
    int block = d / blockSize;
    int pos = d % blockSize;
    // Transpose: (block, pos) → (pos, block)
    int new_d = pos * nBlocks + block;
    out[row * nBlocks * blockSize + new_d] = in[idx];
}

__global__ void k_monarch_permute_inv(const float* in, float* out,
                                       int nBlocks, int blockSize, int rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * nBlocks * blockSize;
    if (idx >= total) return;
    int row = idx / (nBlocks * blockSize);
    int d = idx % (nBlocks * blockSize);
    // Inverse transpose: (pos, block) → (block, pos)
    int pos = d / nBlocks;
    int block = d % nBlocks;
    int new_d = block * blockSize + pos;
    out[row * nBlocks * blockSize + new_d] = in[idx];
}

extern "C" void gpu_monarch_permute(const float* in, float* out,
                                     int nBlocks, int blockSize, int rows) {
    int n = rows * nBlocks * blockSize;
    k_monarch_permute<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(in, out, nBlocks, blockSize, rows);
}
extern "C" void gpu_monarch_permute_inv(const float* in, float* out,
                                         int nBlocks, int blockSize, int rows) {
    int n = rows * nBlocks * blockSize;
    k_monarch_permute_inv<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(in, out, nBlocks, blockSize, rows);
}

/* ---- Element-wise add: y = a + b ---- */

__global__ void k_add(const float* a, const float* b, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = a[i] + b[i];
}

__global__ void k_add_inplace(float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] += b[i];
}

/* ---- RMSNorm backward ---- */

__global__ void k_rmsnorm_bwd(const float* x, const float* dout, float* dx,
                              int dim, int n) {
    int row = blockIdx.x;
    if (row >= n) return;
    const float* xr = x + row * dim;
    const float* dor = dout + row * dim;
    float* dxr = dx + row * dim;

    // Compute RMS
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        ss += xr[i] * xr[i];
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        ss += __shfl_down_sync(0xffffffff, ss, offset);
    __shared__ float shared_rms;
    if (threadIdx.x == 0) shared_rms = sqrtf(ss / dim + 1e-8f);
    __syncthreads();
    float rms = shared_rms;
    float inv = 1.0f / rms;

    // Compute dot(dout, x_normed)
    float dot = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        dot += dor[i] * xr[i];
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        dot += __shfl_down_sync(0xffffffff, dot, offset);
    __shared__ float shared_dot;
    if (threadIdx.x == 0) shared_dot = dot;
    __syncthreads();
    dot = shared_dot;

    // dx = inv * (dout - x * dot / (rms^2 * dim))
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        dxr[i] = inv * (dor[i] - xr[i] * dot / (rms * rms * dim));
}

/* ---- Full damped rotation scan over time ----
   One kernel launch for the entire sequence.
   Parallel over (batch, oscillator), sequential over time.
   Layout: gamma/beta/sense are (batch*seqLen, n_osc)
           bank_out is (batch*seqLen, 2*n_osc)
           osc_out is (batch*seqLen, 2*n_osc)
*/

__global__ void k_rotation_scan(
    const float* gamma, const float* beta, const float* sense,
    const float* bank_out, float* osc_out,
    const float* cos_w, const float* sin_w, const float* freqs,
    const float* fm_depth,  // per-oscillator FM modulation depth
    int batch_size, int seq_len, int n_osc, int fold_offset)
{
    // Each thread handles one (batch, oscillator) pair across all timesteps
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * n_osc) return;
    int b = idx / n_osc;
    int k = idx % n_osc;
    int partner = (k + fold_offset) % n_osc;  // FM modulator

    float pos = 0.0f, vel = 0.0f;
    float f_base = freqs[k];
    float fm_beta = fm_depth[k];
    float partner_pos = 0.0f;  // track partner's state for FM

    for (int t = 0; t < seq_len; t++) {
        int row = b * seq_len + t;
        float g = gamma[row * n_osc + k];
        float bt = beta[row * n_osc + k];
        float s = sense[row * n_osc + k];
        float dp = bank_out[row * 2 * n_osc + k];
        float dv = bank_out[row * 2 * n_osc + n_osc + k];

        // Read partner's amplitude for FM modulation
        // Use the PREVIOUS timestep's osc_out (already written by partner's thread)
        // For t=0, partner_pos=0. For t>0, read from osc_out at t-1.
        if (t > 0) {
            int prev_row = b * seq_len + (t - 1);
            partner_pos = osc_out[prev_row * 2 * n_osc + partner];
        }

        // FM: modulate frequency by partner's amplitude
        float f_eff = f_base + fm_beta * partner_pos;
        float cw = cosf(f_eff);
        float sw = sinf(f_eff);
        float f = fmaxf(fabsf(f_eff), 0.01f);  // prevent division by zero

        float new_pos = g * (pos * cw + vel * sw / f) + (1.0f - g) * bt * dp;
        float new_vel = g * (vel * cw - pos * f * sw) + (1.0f - g) * bt * dv;
        pos = new_pos;
        vel = new_vel;

        osc_out[row * 2 * n_osc + k] = s * pos;
        osc_out[row * 2 * n_osc + n_osc + k] = s * vel;
    }
}

extern "C" void gpu_rotation_scan(
    const float* gamma, const float* beta, const float* sense,
    const float* bank_out, float* osc_out,
    const float* cos_w, const float* sin_w, const float* freqs,
    const float* fm_depth,
    int batch_size, int seq_len, int n_osc, int fold_offset)
{
    int n = batch_size * n_osc;
    k_rotation_scan<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(
        gamma, beta, sense, bank_out, osc_out,
        cos_w, sin_w, freqs, fm_depth,
        batch_size, seq_len, n_osc, fold_offset);
}

/* ---- Backward rotation scan ----
   Mirror of k_rotation_scan. One kernel, parallel over (batch × osc).
   Propagates gradients backward through time, accumulates dgamma/dbeta/dsense.

   Inputs: d_osc_out (upstream grad, from W_mix backward)
   Outputs: d_gamma, d_beta (pre-sigmoid), d_sense (pre-sigmoid), d_bank_out
*/

__global__ void k_rotation_scan_bwd(
    const float* gamma, const float* beta, const float* sense,
    const float* bank_out, const float* osc_out,
    const float* d_osc_out,
    float* d_gamma, float* d_beta, float* d_sense,
    float* d_bank_out,
    const float* cos_w, const float* sin_w, const float* freqs,
    int batch_size, int seq_len, int n_osc)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * n_osc) return;
    int b = idx / n_osc;
    int k = idx % n_osc;
    float cw = cos_w[k], sw = sin_w[k], f = freqs[k];

    // First: re-run forward to save all (pos, vel) states
    // We need pos/vel at each timestep for the backward
    // Store in registers — seq_len is typically 128, too large for registers
    // Use shared memory or just recompute. Let's recompute by storing in global.
    // Actually, let's use a simple approach: two passes.

    // Pass 1: Forward, saving prev states into d_bank_out (repurposed as scratch)
    float pos = 0.0f, vel = 0.0f;
    for (int t = 0; t < seq_len; t++) {
        int row = b * seq_len + t;
        d_bank_out[row * 2 * n_osc + k] = pos;
        d_bank_out[row * 2 * n_osc + n_osc + k] = vel;
        float g = gamma[row * n_osc + k];
        float bt = beta[row * n_osc + k];
        float dp = bank_out[row * 2 * n_osc + k];
        float dv = bank_out[row * 2 * n_osc + n_osc + k];
        float old_pos = pos;
        pos = g * (old_pos * cw + vel * sw / f) + (1.0f - g) * bt * dp;
        vel = g * (vel * cw - old_pos * f * sw) + (1.0f - g) * bt * dv;
    }

    // Pass 2: Backward through time
    float d_pos = 0.0f, d_vel = 0.0f;
    for (int t = seq_len - 1; t >= 0; t--) {
        int row = b * seq_len + t;
        float g = gamma[row * n_osc + k];
        float bt = beta[row * n_osc + k];
        float s = sense[row * n_osc + k];
        float dp = bank_out[row * 2 * n_osc + k];
        float dv_bank = bank_out[row * 2 * n_osc + n_osc + k];
        float prev_p = d_bank_out[row * 2 * n_osc + k];
        float prev_v = d_bank_out[row * 2 * n_osc + n_osc + k];

        float cur_p = g * (prev_p * cw + prev_v * sw / f) + (1.0f - g) * bt * dp;
        float cur_v = g * (prev_v * cw - prev_p * f * sw) + (1.0f - g) * bt * dv_bank;

        float d_out_p = d_osc_out[row * 2 * n_osc + k];
        float d_out_v = d_osc_out[row * 2 * n_osc + n_osc + k];

        d_sense[row * n_osc + k] = d_out_p * cur_p + d_out_v * cur_v;
        d_pos += d_out_p * s;
        d_vel += d_out_v * s;

        float rot_p = prev_p * cw + prev_v * sw / f;
        float rot_v = prev_v * cw - prev_p * f * sw;
        d_gamma[row * n_osc + k] = d_pos * (rot_p - bt * dp) + d_vel * (rot_v - bt * dv_bank);
        d_beta[row * n_osc + k] = d_pos * (1.0f - g) * dp + d_vel * (1.0f - g) * dv_bank;

        d_bank_out[row * 2 * n_osc + k] = d_pos * (1.0f - g) * bt;
        d_bank_out[row * 2 * n_osc + n_osc + k] = d_vel * (1.0f - g) * bt;

        float new_d_pos = d_pos * g * cw - d_vel * g * f * sw;
        float new_d_vel = d_pos * g * sw / f + d_vel * g * cw;
        d_pos = new_d_pos;
        d_vel = new_d_vel;
    }
}

extern "C" void gpu_rotation_scan_bwd(
    const float* gamma, const float* beta, const float* sense,
    const float* bank_out, const float* osc_out, const float* d_osc_out,
    float* d_gamma, float* d_beta, float* d_sense, float* d_bank_out,
    const float* cos_w, const float* sin_w, const float* freqs,
    int batch_size, int seq_len, int n_osc)
{
    int n = batch_size * n_osc;
    k_rotation_scan_bwd<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(
        gamma, beta, sense, bank_out, osc_out, d_osc_out,
        d_gamma, d_beta, d_sense, d_bank_out,
        cos_w, sin_w, freqs, batch_size, seq_len, n_osc);
}

/* ---- Sigmoid ---- */

__global__ void k_sigmoid(const float* x, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = 1.0f / (1.0f + expf(-x[i]));
}

__global__ void k_sigmoid_bwd(const float* sig_out, const float* dout,
                               float* dx, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = sig_out[i];
    dx[i] = dout[i] * s * (1.0f - s);
}

/* ---- Cross-entropy loss ---- */

__global__ void k_ce_loss(const float* logits, const int* targets,
                          float* losses, int vocab, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* row = logits + i * vocab;
    float mx = row[0];
    for (int j = 1; j < vocab; j++) mx = fmaxf(mx, row[j]);
    float sum = 0.0f;
    for (int j = 0; j < vocab; j++) sum += expf(row[j] - mx);
    losses[i] = -(row[targets[i]] - mx - logf(sum));
}

__global__ void k_ce_backward(const float* logits, const int* targets,
                               float* dlogits, int vocab, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * vocab) return;
    int i = idx / vocab, j = idx % vocab;
    const float* row = logits + i * vocab;
    float mx = row[0];
    for (int k = 1; k < vocab; k++) mx = fmaxf(mx, row[k]);
    float sum = 0.0f;
    for (int k = 0; k < vocab; k++) sum += expf(row[k] - mx);
    float prob = expf(row[j] - mx) / sum;
    dlogits[idx] = (prob - (j == targets[i] ? 1.0f : 0.0f)) / (float)n;
}

/* ---- AdamW ---- */

__global__ void k_adamw(float* param, float* grad, float* m, float* v,
                        float lr, float b1, float b2, float wd,
                        float bc1, float bc2, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grad[i];
    m[i] = b1 * m[i] + (1.0f - b1) * g;
    v[i] = b2 * v[i] + (1.0f - b2) * g * g;
    param[i] *= (1.0f - lr * wd);
    param[i] -= lr * (m[i] * bc1) / (sqrtf(v[i] * bc2) + 1e-8f);
    grad[i] = 0.0f;
}

/* ---- Gradient norm (for clipping) ---- */

__global__ void k_grad_norm(const float* grad, float* out, int n) {
    __shared__ float sdata[BLOCK];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x)
        sum += grad[i] * grad[i];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

__global__ void k_scale(float* x, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] *= s;
}

/* ---- Host wrappers ---- */

extern "C" {

void gpu_embed_fwd(const int* ids, const float* table, float* out, int dim, int n) {
    k_embed_fwd<<<(n*dim+BLOCK-1)/BLOCK, BLOCK>>>(ids, table, out, dim, n);
}
void gpu_embed_bwd(const int* ids, const float* dout, float* dtable, int dim, int n) {
    k_embed_bwd<<<(n*dim+BLOCK-1)/BLOCK, BLOCK>>>(ids, dout, dtable, dim, n);
}
void gpu_rmsnorm(const float* x, float* out, int dim, int n) {
    k_rmsnorm_fwd<<<n, BLOCK>>>(x, out, dim, n);
}
void gpu_rmsnorm_bwd(const float* x, const float* dout, float* dx, int dim, int n) {
    k_rmsnorm_bwd<<<n, BLOCK>>>(x, dout, dx, dim, n);
}
void gpu_add(const float* a, const float* b, float* y, int n) {
    k_add<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(a, b, y, n);
}
void gpu_add_inplace(float* a, const float* b, int n) {
    k_add_inplace<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(a, b, n);
}
void gpu_sinegate_fwd(const float* x, float* out, int n) {
    k_sinegate_fwd<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(x, out, n);
}
void gpu_sinegate_bwd(const float* x, const float* dout, float* dx, int n) {
    k_sinegate_bwd<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(x, dout, dx, n);
}
void gpu_rotation_step(float* pos, float* vel, const float* gamma, const float* beta,
                       const float* drive, const float* cos_w, const float* sin_w,
                       const float* freqs, int B, int n_osc) {
    int n = B * n_osc;
    k_rotation_step<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(pos, vel, gamma, beta, drive,
                                                    cos_w, sin_w, freqs, B, n_osc);
}
void gpu_rotation_step_bwd(const float* dpos_out, const float* dvel_out,
                           const float* pos_prev, const float* vel_prev,
                           const float* gamma, const float* beta, const float* drive,
                           const float* cos_w, const float* sin_w, const float* freqs,
                           float* dpos_prev, float* dvel_prev,
                           float* dgamma, float* dbeta, float* ddrive,
                           int B, int n_osc) {
    int n = B * n_osc;
    k_rotation_step_bwd<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(dpos_out, dvel_out,
        pos_prev, vel_prev, gamma, beta, drive, cos_w, sin_w, freqs,
        dpos_prev, dvel_prev, dgamma, dbeta, ddrive, B, n_osc);
}
void gpu_sigmoid(const float* x, float* out, int n) {
    k_sigmoid<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(x, out, n);
}
void gpu_sigmoid_bwd(const float* sig_out, const float* dout, float* dx, int n) {
    k_sigmoid_bwd<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(sig_out, dout, dx, n);
}
void gpu_ce_loss(const float* logits, const int* targets, float* losses, int vocab, int n) {
    k_ce_loss<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(logits, targets, losses, vocab, n);
}
void gpu_ce_backward(const float* logits, const int* targets, float* dlogits, int vocab, int n) {
    k_ce_backward<<<(n*vocab+BLOCK-1)/BLOCK, BLOCK>>>(logits, targets, dlogits, vocab, n);
}
void gpu_adamw(float* param, float* grad, float* m, float* v,
               float lr, float b1, float b2, float wd,
               float bc1, float bc2, int n) {
    k_adamw<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(param, grad, m, v, lr, b1, b2, wd, bc1, bc2, n);
}
void gpu_grad_norm(const float* grad, float* out, int n) {
    k_grad_norm<<<1, BLOCK>>>(grad, out, n);
}
void gpu_scale(float* x, float s, int n) {
    k_scale<<<(n+BLOCK-1)/BLOCK, BLOCK>>>(x, s, n);
}

} // extern "C"
