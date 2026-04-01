# Resonance Research — Experiments to Run

## Proven
- [x] FFT oscillator bank is the key (not one-token-at-a-time)
- [x] Weight-tied drive (strike and listen same signature)
- [x] Dense W: 2.37 BPC, generates real words
- [x] Element-wise prism: 3.1 BPC, converges 6x faster than W, zero matmul
- [x] Gradient descent needed (Hebbian too weak for text)
- [x] OCaml 5 parallel batch training works
- [x] SineGate MLP WORSE than linear (2.60 vs 2.37) — bank is the nonlinearity

## Disproven
- [x] Predictive coding layers add zero (bank + drive does all the work)
- [x] Feedforward init kills learning (erases temporal memory)
- [x] Lateral inhibition doesn't help at this scale
- [x] Complex phase coupling: tiny improvement, not worth complexity
- [x] Butterfly prism: goes NaN, pair mixing amplifies gradients
- [x] More settle steps: no improvement (3 vs 20)
- [x] Hebbian learning: 3.2 BPC ceiling after 1000 passes
- [x] Context ringing hypothesis: same floor on tiny vs full text

## To Test — Architecture
- [ ] **Stacked stages**: bank → prism → bank → prism → ... with residuals
  - Hypothesis: depth gives composition, like S4 stacking layers
  - Test: 2, 4, 6 stages at same total params
  - Why it might work: each stage refines the previous extraction

- [ ] **Wider prism**: more dimensions per layer (expand then contract)
  - state_dim → 2x expansion → prism → contract back
  - More mixing without matrix multiply

- [ ] **Multiple oscillator banks with different frequency ranges**
  - Bank 1: low freq (word context), Bank 2: high freq (character detail)
  - Concatenate states before prism
  - Like dual-stream but simpler

- [ ] **Learnable oscillator frequencies** (retune during training)
  - Currently fixed at init. S4's HiPPO init might be better.
  - Analytical dH/dω gradient from the Python field optimizer

- [ ] **RMSNorm between stages**
  - Prevents activation growth across stages
  - Standard in all deep architectures

- [ ] **Residual connections**
  - output = input + prism(input)
  - Essential for training deep stacks

## To Test — Training
- [ ] **Gradient accumulation across batches**
  - Smoother gradients, higher effective batch size
  - Already have Par.map infrastructure

- [ ] **Learning rate per component**
  - Drive weights, prism params, bank frequencies might need different LRs

- [ ] **Longer sequences** (512, 1024)
  - More context for the oscillators to capture
  - FFT cost is O(n log n), scales well

- [ ] **Gradient clipping**
  - Prevent the NaN issues (butterfly died from gradient explosion)

- [ ] **Adam-style momentum for prism params**
  - Per-parameter adaptive LR, still local
  - Might help prism converge faster

## To Test — Readout
- [ ] **Softmax temperature scheduling**
  - Start soft (high temp), sharpen over training

- [ ] **Drive weight normalization**
  - Normalize drive signatures to unit length
  - Prevents logit magnitude drift

## To Test — Data
- [ ] **Multiple epochs over Shakespeare**
  - Currently one pass. Multiple passes = more gradient signal per pattern.

- [ ] **Byte-pair encoding** (merge common pairs into single tokens)
  - Reduces sequence length, more context per FFT

- [ ] **Curriculum learning** (short sequences first, then longer)

## Big Ideas (Longer Term)
- [ ] **Dual-stream cross-prediction** (MPC paper)
  - Two banks predicting each other's state
  - Self-supervised representation learning

- [ ] **Predictive coding with FFT bank states as input**
  - PC failed when bank was crude. With FFT bank, states are richer.
  - Maybe PC layers can now add value on top of FFT states?

- [ ] **Audio mode** — same architecture, different drive encoding
  - Oscillators ARE sound. Direct application.

- [ ] **Sparse prism** — only top-k elements nonzero per layer
  - Even more efficient, natural feature selection

- [ ] **RISC-V deployment** — cross-compile, benchmark on real hardware

## Architecture Comparison Table

| Architecture | BPC | Params | Ops/pos | Matmul? | Notes |
|---|---|---|---|---|---|
| Raw bank + drive | 3.10 | 16K | O(n) | No | Floor, no processing |
| Bank + dense W | 2.37 | 37K | O(n²) | YES | Best BPC, needs GPU at scale |
| Bank + SineGate MLP | 2.60 | 48K | O(n²) | YES | Worse than linear W |
| Bank + element prism | ~3.1 | 3K | O(n log n) | No | Fast converge, low floor |
| Bank + butterfly prism | NaN | 3K | O(n log n) | No | Gradient explosion |
| Bank + stacked prisms | ? | ~9K | O(n log n) | No | **NEXT TEST** |
