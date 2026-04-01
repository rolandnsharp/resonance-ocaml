# Resonance OCaml — TODO

## Now
- [x] Transfer function H(ω) = 1/(ω₀² - ω² + 2iγω)
- [x] Oscillator bank (analysis via impulse response convolution)
- [x] Predictive coding layer (local errors, Hebbian updates)
- [x] Basic network (multi-layer, settle + learn)
- [ ] Character-level text encoding (char → oscillator states)
- [ ] Text prediction loop (predict next char from oscillator states)
- [ ] Shakespeare training + generation

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
