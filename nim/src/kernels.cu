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
