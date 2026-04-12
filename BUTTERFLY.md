# Butterfly W_mix Replacement — Why It Should Work This Time

## Previous failures

### Butterfly prism (OCaml, commit 3b855d2)
- **What:** Pair-mixing layers replacing element-wise scaling in the "prism" transform
- **Result:** Went NaN at step 3100
- **Why it failed:** "Pair mixing amplifies gradients." The butterfly was inside the processing pipeline where gradient explosion could compound across the full sequence.

### Bilinear coupling in recurrence (Nim, early experiment)
- **What:** `scan(k) * scan(partner)` product inside the damped rotation recurrence
- **Result:** NaN immediately
- **Why it failed:** The coupling was inside the sequential scan. Each timestep multiplied the state by a factor involving another oscillator's state. Over 128 timesteps, any amplification factor > 1 grows exponentially: 1.01^128 = 3.6, 1.1^128 = 200K.

## Why it should work now

The architecture has changed fundamentally since those failures:

### 1. The butterfly replaces W_mix, which is PER-POSITION
W_mix operates on the oscillator output at each position independently — it's NOT inside the recurrence. The damped rotation scan runs first (sequential over time), then W_mix processes the output (parallel over positions). Gradient through the butterfly never passes through the recurrence. This is the same position where the dense W_mix currently sits — and the dense W works fine.

### 2. Residual connection absorbs instability
The layer structure is:
```
output = state + scale * SineGate(butterfly(normed_state))
```
Even if the butterfly output is large, the residual means the state is preserved. The butterfly only contributes a delta. With RMSNorm before the butterfly, the input is unit-scale. The SineGate `x*sin(x)` is bounded for small x. The residual makes the butterfly's contribution ADDITIVE, not multiplicative.

### 3. The butterfly is applied to RMSNorm'd input
The previous butterfly prism operated on raw, unnormalized state. The current architecture applies RMSNorm before W_mix. This means the butterfly sees unit-scale input at every layer — no accumulation of magnitude across layers.

### 4. Better optimizer
The original butterfly prism used manual SGD with learning rate scheduling. We now use Adam with proper bias correction. Adam's per-parameter adaptive learning rates are much more stable for learned rotations.

## The butterfly structure

Replace W_mix (dim × dim = O(d²) params) with log₂(n_osc) levels of pairwise rotations:

```
For level = 0 to log₂(n_osc) - 1:
    offset = n_osc / 2^(level+1)
    For each pair (k, k + offset):
        (a, b) = (x[k], x[k+offset])
        x[k]        = a * cos(θ) + b * sin(θ)
        x[k+offset] = b * cos(θ) - a * sin(θ)
```

**Parameters:** n_osc/2 × log₂(n_osc) rotation angles per layer.
At n_osc=96: ~48 × 7 = 336 params (vs 36,864 for dense W_mix).
That's 110x fewer parameters with full O(n log n) mixing.

The rotation is unitary (energy-preserving), which naturally prevents gradient explosion. Each level's rotation has eigenvalues on the unit circle — no amplification possible.

## The backward

The butterfly backward is the transpose butterfly — apply the same rotations in reverse order with negated angles. Each rotation is its own inverse (up to sign). The gradient flows through exactly log₂(n_osc) levels of orthogonal transforms — no accumulation, no explosion.

## Implementation plan

1. Replace `gpuSgemm(opNN, BT, dim, dim, oscOutBuf, wMix, mixed)` with butterfly kernel
2. The kernel processes both pos and vel components (apply same rotation to both)
3. Learned angles stored as n_osc/2 × n_levels floats per layer
4. Backward: reverse butterfly with gradient accumulation for d_angles
5. Keep SineGate + residual unchanged
