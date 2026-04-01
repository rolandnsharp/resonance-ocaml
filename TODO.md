# Resonance OCaml — TODO

## Done
- [x] Transfer function H(ω) = 1/(ω₀² - ω² + 2iγω)
- [x] Oscillator bank with exact ODE step
- [x] Predictive coding with local Hebbian updates (iPC)
- [x] Resonance coupling (transfer function replaces weight matrices)
- [x] Complex phase coupling (Re + Im, not just |H|)
- [x] Lateral inhibition (divisive normalization, oscillator competition)
- [x] Weight-tied drive (strike and listen same signature)
- [x] Continuous stream (no reset, oscillators ring forever)
- [x] Shakespeare training: loss 5.54 → 3.10 BPC

## Findings
- Feedforward init KILLS learning (erases temporal memory)
- tanh activation works but slightly worse than linear
- Magnitude-only vs complex coupling: no significant difference
- Lateral inhibition: no improvement over baseline
- 20 settle steps vs 3: no improvement
- ODE step vs crude decay: no improvement
- Floor at ~3.1 BPC regardless of: model size (32-64 osc),
  layers (2-3), coupling type, settle depth, inhibition
- The Python version worked because it had learned linear transforms
  (emit, gate, out_proj, feed-forward) — 7×state_dim² free params/layer

## The open question
How to get content-dependent routing without free weight matrices?
The brain does it. We haven't found how yet.

Candidates:
- Phase synchronization (Kuramoto-style)
- Oscillator emit/gate banks (transfer function as learned filter)
- Recurrent coupling (lateral connections within a layer)
- Multiple time scales (fast/slow oscillators for different context)
- Spike-based gating (threshold + reset dynamics)

## Infrastructure
- [ ] OCaml 5 for parallel (Eio structured concurrency)
- [ ] Owl for FFT (causal convolution over full sequence)
- [ ] Push to GitHub
- [ ] RISC-V cross-compilation
