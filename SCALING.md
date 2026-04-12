# Scaling the Resonance Architecture

## The scaling breakthrough

Three fixes, each building on the last:

### 1. Logit clamping (fixes NaN at vocab > 256)
The CE loss kernel computes `exp(logit)` which overflows at float32 for large logits.
Clamping logits to [-30, 30] before exp() fixes NaN at vocab=1024.
**Without this, nothing else works.**

### 2. Feedback (breaks the gradient plateau)
`drive(t) = bank_out(t) + fb × osc_out(t-1)`

The previous output feeds back as additional drive. Creates a gradient highway —
gradients can flow through the feedback path instead of decaying through 128 rotation steps.

Like a guitar feeding back through an amp: self-sustaining signal.
Initialized at 0, learns to add sustain.

**96 osc FineWeb: 5.71 → 5.46 (broke the plateau)**

### 3. Full Xavier init for W_mix (unlocks large models)
The W_mix was initialized at 0.01/sqrt(dim) — near zero. The layers didn't contribute.
Gradients through W_mix were proportional to its magnitude: near-zero W → near-zero gradient → W stays near-zero forever.

Full Xavier init: 1.0/sqrt(dim). The logit clamp prevents NaN from larger activations.
The layers contribute from step 1.

**256 osc FineWeb: stuck at 6.93 (random) → 4.69 (learning!)**

## FineWeb results

| Config | Params | Loss | BPC | Steps | What changed |
|---|---|---|---|---|---|
| 96 osc, 6L, no feedback | 849K | 5.71 | 8.24 | 10K | Baseline |
| 96 osc, 6L + feedback | 849K | 5.46 | 7.88 | 10K | + feedback |
| **256 osc, 8L + feedback + Xavier** | **6M** | **4.69** | **6.76** | **5K** | + Xavier init |
| Random baseline (vocab 1024) | — | 6.93 | 10.0 | — | — |

## What we learned about scaling

### Init scale is critical
At dim=192 (96 osc), 0.01/sqrt(dim) = 0.0007 works — it's small but the model learns.
At dim=512 (256 osc), 0.01/sqrt(dim) = 0.0004 is DEAD — gradients are too small.
Full Xavier (1/sqrt(dim)) works at all scales when logit clamping prevents NaN.

**Rule: always use 1/sqrt(dim) for W_mix. Clamp logits for safety.**

### Feedback compounds with scale
At 96 osc, feedback gave 0.25 improvement (5.71 → 5.46).
At 256 osc with feedback + Xavier, the combined effect is 2.24 (6.93 → 4.69).
Feedback's gradient highway matters MORE at larger scale where the rotation has more steps to vanish through.

### More oscillators = better
96 osc (5K steps): loss 5.83
256 osc (5K steps): loss 4.69
The 256-osc model is 1.14 loss better at the same step count. More spectral resolution captures more patterns.

## Next experiments to try

### Nonzero feedback init
Current: `feedback = 0.0` (model discovers feedback from scratch).
Try: `feedback = 0.1` (moderate sustain from the start).
Rationale: the model shouldn't have to discover that feedback helps — give it a head start.

### Higher LR
Current: 3e-4. Golf submission used 3e-3 (10x higher).
The larger model with Xavier init should tolerate higher LR.

### 512 osc
If 256 osc works, does 512 osc work? dim=1024, W_mix = 1M params.
May need batch_size reduction for VRAM.

### Longer sequences
Current: seq_len=256. Try 512, 1024.
More context = more temporal patterns for the bank to capture.
The rotation scan is O(seq_len) — linear, but gradient vanishing gets worse.
Feedback should help: the gradient highway bypasses the rotation.
