# Resonance OCaml — TODO

## Now
- [x] Transfer function H(ω) = 1/(ω₀² - ω² + 2iγω)
- [x] Oscillator bank (analysis via impulse response convolution)
- [x] Predictive coding layer (local errors, Hebbian updates)
- [x] Basic network (multi-layer, settle + learn)
- [x] Byte-level encoding (byte → drive force → strike bank)
- [x] Text prediction (oscillator states → PC layers → next byte)
- [x] Shakespeare training (loss 5.54 → 4.37, no backprop, CPU)
- [ ] Shakespeare coherent generation (need more training / tuning)
- [ ] Profile and optimize hot path (matrix ops, settle loop)

## Soon
- [ ] OCaml 5 for parallel (Eio structured concurrency)
- [ ] Owl for FFT (replace naive convolution with O(n log n))
- [ ] Kuramoto coupling between oscillator heads
- [ ] Benchmark: predictive coding CPU vs backprop GPU

## Later
- [ ] Cross-compile to RISC-V
- [ ] Pico 2W prototype (one oscillator per board)
- [ ] Distributed predictive coding over UART/SPI
- [ ] Audio mode (same architecture, different encoding)
