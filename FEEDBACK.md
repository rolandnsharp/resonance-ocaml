# Feedback Creates Intelligence

## The insight

Feedback is fractal. `z = z² + c` creates the Mandelbrot set. A guitar through an amp creates harmonics that weren't in the pluck. A few neurons in a loop create memory, oscillation, decision.

Our architecture has linear feedback (the rotation state persists). But the nonlinearity (SineGate) is NOT in the feedback loop. It's applied per-position, then discarded. The complexity dies at each step.

## The experiment

Route the SineGate output back into the drive for the next timestep:

```
drive(t) = token_drive(t) + α × SineGate(W × rotation_output(t-1))
```

This creates a nonlinear feedback loop:
- token strikes the bank
- bank rings
- rotation selectively remembers
- W creates interference patterns
- SineGate generates harmonics
- harmonics feed back into the NEXT token's drive
- the cycle repeats, building complexity

The rotation alone gives exponential decay of memory.
Nonlinear feedback gives **persistent complex structure** — attractors, limit cycles, chaos.

## Why this might work

Sound synthesis knows this. Every interesting timbre comes from feedback:
- FM synthesis: oscillator A modulates oscillator B's frequency
- Karplus-Strong: noise → delay → filter → feedback → realistic string sound
- Waveguide: bidirectional delay with nonlinear reflection = physical instrument

Language might need the same: each token doesn't just ADD to the state.
It INTERACTS with the accumulated structure. The structure SHAPES how the
next token is processed. This is what attention does — but attention
rebuilds the structure from scratch at each position (Q×K^T). Feedback
maintains and evolves it continuously.

## The risk

Nonlinear feedback can diverge. The rotation is stable because its eigenvalues
are inside the unit circle (decay < 1). Adding nonlinear feedback breaks this
guarantee. Need careful gain control — like how a guitarist controls feedback
with distance from the amp.

The SineGate output is bounded (x×sin(x) ≈ x² for small x, oscillates for large x).
A small α (0.01-0.1) should keep the feedback stable while adding complexity.

## Connection to biology

Real neurons have feedback everywhere:
- Recurrent connections within a cortical layer
- Thalamocortical loops (thalamus ↔ cortex)
- Basal ganglia loops (action selection)
- Cerebellum loops (timing and prediction)

The brain is NOT feedforward. It's mostly feedback. The feedforward path
(retina → V1 → V2 → ... → IT) is the minority. The majority of connections
are lateral and feedback.

Transformers are pure feedforward. Our rotation adds one feedback loop.
Real intelligence might need MANY.
