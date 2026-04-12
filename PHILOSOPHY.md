# Resonance — What We Learned

## The architecture

One equation, three wave phenomena:

```
Propagation:  bank(FFT) — waves spread through the medium
Resonance:    ẍ + 2γ(t)ωẋ + ω²x = β(t)·F(t) — waves selectively persist
Interference: W × state — waves multiply pairwise, creating cross-terms
```

BPC 2.9 on Shakespeare. 627K params. Trains in 5 minutes on an RTX 3060.

## What we proved

1. **The damped rotation is genuinely novel.** Second-order dynamics where the state ROTATES instead of decaying. Phase encodes temporal distance. Nobody else has published this recurrence.

2. **Interference IS attention.** The dense W is not a compromise — it's the physics of wave interference. `|A + B|²` contains the `A × B` cross-terms that ARE the pairwise associations language needs.

3. **O(n²) mixing is load-bearing.** Every O(n log n) replacement (butterfly, Monarch, spectral convolution) hit the same 4.3 BPC ceiling. The universe pays O(n²) for interference too.

4. **The architecture is quantization-friendly by construction.** All values bounded. Fixed-point feasible. 250x less working memory than transformers.

## What the numbers say

BPC 2.9 is not competitive with transformers (~1.5 at same scale). But the architecture has properties no transformer has: interpretable state, constant memory, no FPU required, physics-grounded.

## What might be missing — the aither insight

The aither audio engine works by **impulse → synthesis → output**:
- An impulse strikes (a drum hit, a note onset)
- The synthesis chain processes it (filters, envelopes, effects)
- The output is a continuous waveform

Language might work the same way:
- A token arrives (impulse)
- The oscillator bank rings (synthesis)
- The prediction emerges from the ringing pattern

But aither's synthesis is richer than what we have. It has:
- **Envelopes**: amplitude shapes that control attack/decay/sustain/release
- **Modulation**: one oscillator controlling another's frequency or amplitude
- **Feedback**: output routed back to input, creating complex timbres
- **Multiple voices**: independent signal chains mixed at the output

Our architecture has the oscillator bank (the "voice") and the rotation (the "envelope"). But it lacks:
- **Cross-modulation**: oscillator k's output modulating oscillator j's frequency. This would be a content-dependent frequency shift — different from the fixed-frequency rotation we have.
- **Feedback routing**: the output of the W_mix feeding back into the next timestep's drive. We have the rotation scan providing memory, but not output-to-input feedback within a layer.
- **Multiple synthesis chains**: each layer is one chain. What if layers were PARALLEL voices, not sequential stages?

## The deeper question

Is intelligence wave interference? The brain runs on oscillations (alpha, beta, gamma, theta). Neural synchronization — where distant brain regions lock their oscillation phases — is how the brain binds information across areas. Phase-locking IS interference. The brain IS computing with waves.

But the brain also has:
- **Spike timing**: discrete impulses, not continuous waves
- **Synaptic plasticity**: connection strengths that change with use
- **Inhibition**: signals that SUPPRESS rather than excite
- **Hierarchy**: cortical columns at multiple spatial scales

Maybe the next step isn't making the oscillators better. Maybe it's making the CONNECTIONS between oscillators more brain-like: sparse, plastic, inhibitory, hierarchical.

## For the future

The code is clean. Three implementations (OCaml, Python, Nim). Custom CUDA kernels. Full documentation. If someone picks this up — or if we come back to it — the starting point is solid.

The truth might be in the synthesis, not the oscillators.
