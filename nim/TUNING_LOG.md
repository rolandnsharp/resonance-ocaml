# Hyperparameter Tuning Log — FineWeb

## Critical Finding: FineWeb vocab=1024 produces NaN

ALL FineWeb runs with the CORRECTLY compiled binary produce NaN at step 0.
The previous "successful" runs (loss 4.85, 3K steps) were running a STALE binary
from before the DSP refactor — the compile was silently failing due to `invFft` being
undefined, and the old binary was being reused.

### What works:
- Shakespeare (vocab=256, seq=128): loss 5.7 → 2.85 BPC ✓
- Any config with vocab=256 ✓

### What produces NaN:
- FineWeb vocab=1024 at ANY oscillator count (96, 128, 256) ✗
- Even at step 0 with lr=0, batch=1 ✗
- NaN is in the FORWARD, not from gradients ✗

### Likely cause:
The NaN appears during forward computation at vocab=1024.
Initial weights are NaN-free (verified by downloading and checking).
The issue is somewhere in: embed → bank → rotation → W_mix → W_out → CE loss.
With larger vocab, the CE loss kernel processes 1024 logits per position.
The `k_ce_loss` kernel computes max, exp, sum, log — numerical issues with
large logit arrays at float32 are possible.

### Next steps:
1. Add NaN checks after each forward operation to isolate the source
2. Try float64 for the CE loss kernel
3. Try vocab=256 on FineWeb (byte-level, ignore tokenizer)
4. Fix the `invFft` compile error in the DSP forward path

### Results on Shakespeare (VALID):
| Config | BPC | Steps | Status |
|---|---|---|---|
| 96 osc, 6L, batch=64, lr=3e-4 | 2.85 | 20K | ✓ Best |
| 96 osc, 6L, batch=32, lr=3e-4 | 2.72 | 20K | ✓ |
| 96 osc, 6L, batch=64, lr=3e-4 (no bias) | 2.87 | 20K | ✓ |

### FineWeb Results (ALL INVALID — stale binary)
All FineWeb results reported before this log entry were from a stale binary.
They cannot be trusted.
