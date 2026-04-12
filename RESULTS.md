# Resonance — Results Summary

## Architecture

One equation at every layer:
```
pos' = γ(t)·[pos·cos(ω) + vel·sin(ω)/ω] + (1-γ(t))·β(t)·drive_pos
vel' = γ(t)·[vel·cos(ω) - pos·ω·sin(ω)] + (1-γ(t))·β(t)·drive_vel
```

FFT bank → [selective damped rotation → W_mix → SineGate → residual] × L → W_out → listen

## Python (PyTorch) — `resonance.py`

| Config | Params | Steps | BPC | Time | Hardware |
|---|---|---|---|---|---|
| 96 osc, 6 layers, batch=64 | 640K | 2,000 | **2.45** | ~25 min | RTX 3060 |
| 128 osc, 8 layers, batch=64 | 1.4M | 2,000 | **2.36** | ~33 min | RTX 3060 |

- Adam with OneCycleLR (peak 3e-4, 5% warmup, cosine decay)
- Autograd handles all gradients (including FFT bank backward)
- Sequential scan via Python for-loop (~1.3 steps/sec)
- SineGate `x * sin(x)` nonlinearity
- Damped rotation with precomputed cos/sin tables

## Nim (CUDA) — `nim/src/`

| Config | Params | Steps | BPC | Steps/sec | Time | Hardware |
|---|---|---|---|---|---|---|
| 96 osc, 6 layers, batch=64 | 627K | 10,000 | **3.04** | 72 | 2.3 min | RTX 3060 |
| 96 osc, 6 layers, batch=32 | 627K | 20,000 | **2.72** | 69 | 4.8 min | RTX 3060 |

- Custom CUDA kernels: rotation scan fwd/bwd, SineGate, RMSNorm, FFT bank, AdamW
- Batched cuFFT for bank encoding (600x speedup over naive)
- Single-kernel rotation scan (parallel over batch×oscillator, sequential over time)
- Backward recomputes forward states from scratch (no stack arrays)
- **53-72x faster than Python per step**

## Gap Analysis (Nim 3.04 vs Python 2.45)

The 0.59 BPC gap is training optimization, not architecture:
- Python has autograd with exact gradients through everything; Nim has hand-written backward
- Python uses `nn.Linear` with bias for projections; Nim uses bias-free
- Python's Adam has more numerically precise float32 accumulation
- The Nim bank backward may have remaining stride issues in the gather kernels

## OCaml (historical) — `lib/`, `bin/`

The original implementation. Key results from git history:

| Architecture | BPC | Notes |
|---|---|---|
| Bank + dense W (online SGD) | 2.37 | The breakthrough that broke 3.1 wall |
| Bank + W_gate + FFN layers | 2.37 | 6-layer transformer-style |
| Pure wave (Duffing + butterfly fold) | 4.75 | Zero matrix multiply |
| Interference gate + causal EMA | 5.20 | Wave-native gating |

## Key Findings

1. **Online updates break the 3.1 wall** — batch gradient accumulation plateaus; per-position updates give the W immediate feedback
2. **The damped rotation IS the novel contribution** — second-order dynamics with phase, not Mamba's first-order decay
3. **Nim + CUDA is 53-72x faster than Python** — custom kernels eliminate autograd overhead for sequential operations
4. **Architecture is quantization-friendly** — all values bounded (sigmoid, cos/sin), precomputed trig tables, fixed-point feasible
5. **250x less working memory than transformer** — rotation state (1.2KB) vs KV cache (295KB) at same params

## Files

- `resonance.py` — PyTorch reference implementation (standalone, ~270 lines)
- `nim/src/kernels.cu` — Custom CUDA kernels (rotation, SineGate, bank, AdamW)
- `nim/src/gpu.nim` — CUDA bindings and cuBLAS wrappers
- `nim/src/model.nim` — Model type, parameter init
- `nim/src/train.nim` — Training loop with FFT bank and rotation backward
- `lib/` — Original OCaml implementation (historical)
- `bin/` — OCaml training entry point (historical)
