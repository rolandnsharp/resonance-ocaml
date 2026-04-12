# Butterfly Research — State of the Art (2023-2026)

## The headline finding

**ButterflyQuant (2025)** uses learned orthogonal butterfly transforms for 2-bit LLM quantization. On LLaMA-2-7B: perplexity 15.40 vs 36.77 for QuaRot — 2.4x lower perplexity at the same bit-width. Uses O(n log n) learnable parameters (1,228 params for 5120-dim layers — 21,347x reduction vs full matrix). Converges in 5-10 minutes on a single GPU.

**This is exactly what we need.** Our architecture is quantization-friendly by design. ButterflyQuant proves the learned rotation approach works at scale for quantization.

## What works

### Monarch Matrices (Dao & Re, 2022)
- Factorizes W as product of two block-diagonal matrices + permutation: `M = P * L * R`
- Key insight: abandon O(N log N) butterfly for O(N^{3/2}) block-diagonal — maps to GPU batch GEMM
- **2x real speedup** over dense matmul on GPU
- M2-BERT: 80M params, GLUE 79.9 (matches BERT-base with 27% fewer params)
- M2-GPT-2: 360M params, matched GPT-2 perplexity with 2x training speedup
- **360M on The Pile — first demonstration that Transformer quality is achievable without attention or MLPs**

### ButterflyMoE (2024) — most relevant to us
- Parameterize MoE experts as butterfly rotations of a shared ternary substrate
- `Wi = B(phi_i) * W_base * B(theta_i)^T`
- **150x compression** at 256 experts (256 MB → 1.9 MB)
- ButterflyViT: **354x compression** at 64 experts
- Enables 10,540 experts on Jetson Nano (vs 31 for standard MoE)
- **This is the RISC-V deployment story we need**

### SPECTRE (Feb 2025) — FFT attention replacement
- Replaces each attention head with FFT + content-adaptive spectral gate + IFFT
- **7x faster than FlashAttention-2 on 128K tokens**
- Matches FA2 on PG-19 perplexity
- The "content-adaptive spectral gate" is exactly our interference gate concept

### MatMul-free LM (2024)
- 2.7B params, matches Transformer++ quality
- 61% less memory, 10x inference speedup
- Uses ternary weights — connects to our quantization story

## What hasn't worked

- **Nobody has trained >1B butterfly LLM from scratch** matching dense Transformers
- O(N log N) butterfly has terrible GPU utilization — irregular memory access
- Butterfly is sensitive to weight decay (one path from input to output)
- Power-of-two dimension constraints are annoying
- ReLU inside butterfly destroys information

## Key insight for our architecture

**The GPU vs RISC-V split is confirmed by the research:**

> "Classical butterfly matrices have O(N log N) FLOPs but terrible GPU utilization due to irregular memory access. Monarch sidesteps this by using dense block-diagonal matrices + permutation."

This means:
- For GPU training: use Monarch-style block-diagonal (O(N^{3/2})) — it works with cuBLAS
- For RISC-V inference: use true butterfly (O(N log N)) — memory access doesn't matter on simple cores

**Train with Monarch blocks, deploy with butterfly.** Same math, different factorization for different hardware.

## Implementation plan

### Phase 1: Replace W_mix with Monarch block-diagonal (GPU training)
- Factor dim × dim into sqrt(dim) × sqrt(dim) blocks × permutation
- Use torch.bmm / cuBLAS batched GEMM — fast on GPU
- At dim=192: ~14 blocks of ~14×14 = ~2,744 params per Monarch factor, two factors = ~5,488 vs 36,864 dense

### Phase 2: Distill to butterfly (RISC-V inference)
- Decompose trained Monarch matrices into O(N log N) butterfly factors
- Use ButterflyQuant's learned rotation approach for 4-bit quantization
- Deploy on RISC-V with INT8 fixed-point butterfly rotations

### Phase 3: ButterflyMoE for scaling
- Share a base W_mix, add per-expert butterfly rotations
- 64 experts at 354x compression = massive scaling with tiny memory
- Each expert = different "view" of the oscillator bank through different rotation angles

## Sources
- [ButterflyQuant](https://arxiv.org/abs/2509.09679) — 2-bit quantization via learned butterfly
- [Monarch Matrices](https://proceedings.mlr.press/v162/dao22a) — the practical breakthrough
- [Monarch Mixer (M2)](https://arxiv.org/abs/2310.12109) — 360M GPT matching dense
- [ButterflyMoE](https://arxiv.org/html/2601.13563) — 150x compression for MoE
- [SPECTRE](https://arxiv.org/abs/2502.18394) — 7x faster than FlashAttention-2
- [MatMul-free LM](https://arxiv.org/abs/2406.02528) — 2.7B without matmul
- [MonarchAttention](https://arxiv.org/abs/2505.18698) — zero-shot attention replacement
