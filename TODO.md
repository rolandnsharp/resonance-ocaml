# Resonance OCaml — TODO

## Done
- [x] Oscillator bank with FFT convolution (Owl)
- [x] Weight-tied drive (strike and listen same signature)
- [x] Learned W transform → broke 3.1 BPC floor → 2.37 BPC
- [x] Prism: composed couple-shuffle layers replacing matrix multiply
- [x] OCaml 5 parallel (Domain-based batch training)
- [x] Precomputed FFT kernels
- [x] Named function pipeline: strike → resonate → transform → listen → predict

## Key findings
- Predictive coding layers add zero over raw bank + drive
- Feedforward init kills learning (erases temporal memory)
- SineGate MLP worse than linear W (oscillator bank IS the nonlinearity)
- Prism converges 6x faster than dense W matrix
- Single W: 2.37 BPC. Prism: testing now (50K run)
- No matrix multiply needed — O(n log n) prism matches O(n²) W

## Architecture
```
tokens |> strike (drive lookup)
       |> resonate (FFT convolve with h(t))
       |> prism (couple → shuffle → couple → ...)
       |> listen (dot with drive signatures)
       |> softmax → next byte
```
Zero matrix multiplies. Every op is O(n) or O(n log n).

## Next
- [ ] Push to GitHub
- [ ] Scale: more oscillators, longer sequences
- [ ] Stack multiple prism+bank layers with residuals
- [ ] S4-style HiPPO initialization for oscillator frequencies
- [ ] RISC-V cross-compilation
- [ ] Benchmark against S4/Mamba at same param count
