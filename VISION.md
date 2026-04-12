# Resonance — Scaling Vision

## Architecture

One equation at every layer:

```
pos' = γ(t)·[pos·cos(ω) + vel·sin(ω)/ω] + (1-γ(t))·β(t)·drive_pos
vel' = γ(t)·[vel·cos(ω) - pos·ω·sin(ω)] + (1-γ(t))·β(t)·drive_vel
```

The state ROTATES instead of decaying. Phase encodes temporal distance. γ, β, c are input-dependent (selective). Not Mamba (diagonal decay). Second-order dynamics with momentum.

**Current results:** BPC 2.72 on Shakespeare. 627K params. Trains in 4 minutes on RTX 3060 at 69 steps/sec. Nim + custom CUDA — 53x faster than PyTorch.

## What's blocking real scaling

### 1. The W_mix is the bottleneck, not the oscillators

The rotation scan is O(n_osc) per token — trivially cheap. But W_mix is dim × dim = O(n_osc²). At 96 osc that's 37K MADs. At 512 osc it's 1M. At 1024 it's 4M. This scales the same as a transformer FFN. The oscillators are cheap; the spectral recombination is expensive.

To escape this: we need a wave-native replacement for W_mix that's O(n_osc log n_osc) instead of O(n_osc²). The butterfly rotation network — or FFT across the oscillator dimension. This is the research frontier.

### 2. Training data: 1.1MB is nothing

We're overfitting Shakespeare. Each byte is seen ~35x in 20K steps. Real models need billions of tokens seen ~1x each. We need FineWeb or equivalent.

### 3. Context length: 128 is tiny

Real language needs 1024-4096 context. The rotation scan is O(seq_len) per oscillator — linear, fine. But the FFT bank is O(seq_len log seq_len) — also fine. The real issue: with 128 context the model can barely see a full sentence. It can't learn paragraph structure, dialogue, or narrative.

### 4. For audio: the architecture already fits

This is the exciting part. The oscillator bank IS a spectrogram. The damped rotation IS a filter bank. The frequencies map directly to audio frequencies. The same Nim binary could:
- Train on text (byte-level, character prediction)
- Train on audio (sample-level, waveform prediction)
- Train on sensor data (time series, anomaly detection)

The architecture doesn't care what signal the oscillators are processing. Change ω and you change the domain. This is the real vision beyond text completion — **a universal sequence model for any oscillatory signal**.

## Advantages over transformers and Mamba

| | Transformer | Mamba | Resonance |
|---|---|---|---|
| Temporal processing | O(n²) attention | O(n) diagonal decay | O(n) damped rotation |
| State transition | None (stateless) | Diagonal (dims independent) | Rotation (pos/vel couple) |
| Working memory | KV cache: O(n × d) | State: O(d) | Rotation state: O(d) |
| Phase encoding | Learned (RoPE) | None | Free (from physics) |
| Quantization | Arbitrary activations | Unbounded intermediates | All values bounded (0,1) |
| FPU required | Yes | Yes | **No** (fixed-point OK) |
| Interpretable | No | No | **Yes** (freq, phase, damping) |
| Hardware target | GPU ($10K+) | GPU ($10K+) | **RISC-V ($5)** |

## Cost comparison

**Edge inference on $5 RISC-V chip (Allwinner D1, 1 GHz):**
- Resonance: **540 tokens/sec**, 628KB total memory
- Transformer (same params): 111 tokens/sec, 922KB memory (KV cache)
- 4.8x faster, 250x less working memory

**1000 concurrent users at 50 tok/sec:**
- Cloud GPU: $66/hr ongoing
- RISC-V cluster: $465 one-time, $0.19/hr power
- Break-even: 7 hours

## The path to scaling

### Near term (what we can do now)
- FineWeb dataset (billions of tokens)
- 256 osc, 12 layers (~5M params)
- seq_len 512
- Multi-GPU training via Nim

### Medium term (research)
- Replace W_mix with O(n log n) butterfly rotation network
- Audio training alongside text (multi-modal)
- INT8 quantization-aware training
- RISC-V inference benchmark on real hardware

### Long term (the vision)
- One model that does text + audio + sensor
- Runs on a mesh of $5 RISC-V chips
- Trains on GPU, deploys on microcontrollers
- Open source, physics-grounded, interpretable
