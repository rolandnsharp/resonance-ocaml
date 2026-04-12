"""Deep Resonance — parametric oscillator neural network.

bank(FFT, once) → [selective_recurrence → W_mix → SineGate → residual] × L → W_out → listen

One equation per oscillator per layer:
  x(t) = γ(t)·x(t-1) + (1-γ(t))·β(t)·drive(t), read through c(t)

γ, β, c all input-dependent via learned projections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time, sys

# --- Oscillator Bank (FFT) ---

class OscillatorBank(nn.Module):
    """Precomputed FFT convolution with damped harmonic oscillator kernels."""
    def __init__(self, n_osc, seq_len):
        super().__init__()
        self.n_osc = n_osc
        self.seq_len = seq_len
        self.fft_len = 2 * seq_len

        # Logarithmic frequency spread, low damping
        freqs = torch.linspace(0.1, math.pi, n_osc)
        gammas = torch.linspace(0.05, 0.20, n_osc)

        # Precompute impulse response FFTs
        t = torch.arange(seq_len, dtype=torch.float32)
        h_pos = []
        h_vel = []
        for k in range(n_osc):
            w0, g = freqs[k].item(), gammas[k].item()
            wd = w0 * math.sqrt(max(1e-6, 1.0 - g*g))
            alpha = g * w0
            decay = torch.exp(-alpha * t)
            pos = decay * torch.sin(wd * t) / max(1e-8, wd)
            vel = decay * (torch.cos(wd * t) - (alpha/max(1e-8, wd)) * torch.sin(wd * t))
            h_pos.append(pos)
            h_vel.append(vel)

        h_pos = torch.stack(h_pos)  # (n_osc, seq_len)
        h_vel = torch.stack(h_vel)

        # Pad and FFT
        h_pos_padded = F.pad(h_pos, (0, seq_len))  # (n_osc, fft_len)
        h_vel_padded = F.pad(h_vel, (0, seq_len))
        self.register_buffer('h_pos_fft', torch.fft.rfft(h_pos_padded))
        self.register_buffer('h_vel_fft', torch.fft.rfft(h_vel_padded))

    def forward(self, drives):
        """drives: (batch, seq_len, n_osc) → states: (batch, seq_len, 2*n_osc)"""
        B, T, K = drives.shape
        # Pad drives to fixed fft_len, convolve in frequency domain
        drives_t = drives.transpose(1, 2)  # (B, K, T)
        drives_padded = F.pad(drives_t, (0, self.fft_len - T))  # (B, K, fft_len)
        drives_fft = torch.fft.rfft(drives_padded, dim=2)  # (B, K, fft_len//2+1)

        pos = torch.fft.irfft(drives_fft * self.h_pos_fft, dim=2)[:, :, :T].transpose(1, 2)
        vel = torch.fft.irfft(drives_fft * self.h_vel_fft, dim=2)[:, :, :T].transpose(1, 2)

        return torch.cat([pos, vel], dim=-1)  # (B, T, 2*K)


# --- Resonance Layer ---

class ResonanceLayer(nn.Module):
    """Selective recurrence → W_mix → SineGate → residual"""
    def __init__(self, dim, n_osc):
        super().__init__()
        self.dim = dim
        self.n_osc = n_osc

        # Parametric oscillator projections: state → per-oscillator control
        self.proj_gamma = nn.Linear(dim, n_osc)  # damping
        self.proj_beta = nn.Linear(dim, n_osc)   # absorption
        self.proj_sense = nn.Linear(dim, n_osc)  # readout sensitivity

        # Spectral recombination
        self.w_mix = nn.Linear(dim, dim, bias=False)

        # Init projections small
        for p in [self.proj_gamma, self.proj_beta, self.proj_sense]:
            nn.init.xavier_uniform_(p.weight, gain=0.1)
            nn.init.zeros_(p.bias)

    def forward(self, state, bank_out):
        """state: (B, T, dim), bank_out: (B, T, 2*n_osc) → new_state: (B, T, dim)"""
        B, T, D = state.shape
        n = self.n_osc

        normed = F.rms_norm(state, (D,))

        # Input-dependent controls
        gamma = torch.sigmoid(self.proj_gamma(normed))  # (B, T, n_osc)
        beta = torch.sigmoid(self.proj_beta(normed))
        sense = torch.sigmoid(self.proj_sense(normed))

        # Precompute rotation matrices from oscillator frequencies
        freqs = torch.linspace(0.1, math.pi, n, device=state.device)
        cos_w = torch.cos(freqs)   # (n_osc,)
        sin_w = torch.sin(freqs)

        # Selective damped rotation over time
        osc_pos = torch.zeros(B, n, device=state.device)
        osc_vel = torch.zeros(B, n, device=state.device)
        outputs = []

        for t in range(T):
            g = gamma[:, t, :]  # (B, n_osc) — controls decay
            b = beta[:, t, :]   # — controls drive absorption
            s = sense[:, t, :]  # — controls readout
            drive = bank_out[:, t, :]  # (B, 2*n_osc)

            # Damped rotation: state = decay * Rotate(ω) × state + drive
            decay = g  # sigmoid output ∈ (0,1) — stable by construction
            new_pos = decay * (osc_pos * cos_w + osc_vel * sin_w / freqs) + (1 - g) * b * drive[:, :n]
            new_vel = decay * (osc_vel * cos_w - osc_pos * freqs * sin_w) + (1 - g) * b * drive[:, n:]
            osc_pos = new_pos
            osc_vel = new_vel

            out = torch.cat([s * osc_pos, s * osc_vel], dim=-1)
            outputs.append(out)

        osc_out = torch.stack(outputs, dim=1)  # (B, T, dim)

        # Spectral recombination + SineGate + residual
        mixed = self.w_mix(osc_out)
        activated = mixed * torch.sin(mixed)
        return state + activated


# --- Full Model ---

class Resonance(nn.Module):
    def __init__(self, n_osc=96, n_layers=6, seq_len=128, vocab_size=256):
        super().__init__()
        self.n_osc = n_osc
        self.dim = 2 * n_osc
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        # Drive table: each byte → oscillator excitation pattern
        self.drive = nn.Embedding(vocab_size, self.dim)
        nn.init.normal_(self.drive.weight, std=0.02)

        # FFT bank: temporal encoding
        self.bank = OscillatorBank(n_osc, seq_len)

        # Processing layers
        self.layers = nn.ModuleList([
            ResonanceLayer(self.dim, n_osc) for _ in range(n_layers)
        ])

        # Output synthesis
        self.w_out = nn.Linear(self.dim, self.dim, bias=False)
        nn.init.xavier_uniform_(self.w_out.weight, gain=0.1)

    def forward(self, tokens):
        """tokens: (B, T) → logits: (B, T, vocab_size)"""
        B, T = tokens.shape

        # Strike: extract drives from embeddings
        emb = self.drive(tokens)  # (B, T, dim)
        drives = emb[:, :, :self.n_osc]  # (B, T, n_osc)

        # Resonate: FFT bank encoding
        bank_out = self.bank(drives)  # (B, T, dim)

        # Process through layers
        state = bank_out
        for layer in self.layers:
            state = layer(state, bank_out)

        # Output: norm → W → listen (dot with drive table)
        normed = F.rms_norm(state, (self.dim,))
        transformed = self.w_out(normed)  # (B, T, dim)

        # Listen: dot product with all drive signatures
        logits = transformed @ self.drive.weight.T  # (B, T, vocab_size)
        return logits

    def generate(self, seed, n_gen=200, temperature=0.8):
        """Autoregressive generation from seed tokens."""
        self.eval()
        context = seed.clone()
        generated = []
        with torch.no_grad():
            for _ in range(n_gen):
                logits = self(context[:, -self.seq_len:])
                next_logit = logits[:, -1, :] / temperature
                probs = F.softmax(next_logit, dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated.append(next_token)
                context = torch.cat([context, next_token], dim=1)
        tokens = torch.cat(generated, dim=1)[0]
        return ''.join(chr(t) if 32 <= t < 127 else '.' for t in tokens.tolist())


# --- Training ---

def train(model, text, steps=10000, batch_size=32, seq_len=128, lr=3e-4, device='cuda'):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    data = torch.tensor([ord(c) for c in text], dtype=torch.long, device=device)
    n = len(data)

    model.train()
    t0 = time.time()

    for step in range(steps):
        # Random batch of sequences
        starts = torch.randint(0, n - seq_len - 1, (batch_size,), device=device)
        tokens = torch.stack([data[s:s+seq_len] for s in starts])
        targets = torch.stack([data[s+1:s+seq_len+1] for s in starts])

        logits = model(tokens)
        loss = F.cross_entropy(logits.view(-1, 256), targets.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            bpc = loss.item() / math.log(2)
            elapsed = time.time() - t0
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            print(f"step {step:5d}  loss {loss.item():.3f}  bpc {bpc:.3f}  "
                  f"lr {scheduler.get_last_lr()[0]:.1e}  [{steps_per_sec:.1f} steps/s]", end='')
            if step % 500 == 0:
                seed = torch.tensor([[ord(c) for c in "First Citizen:\n"]],
                                    dtype=torch.long, device=device)
                gen = model.generate(seed, n_gen=60, temperature=0.8)
                print(f"  | {gen}", end='')
            print(flush=True)

    return model


if __name__ == '__main__':
    # Load Shakespeare
    data_path = 'data/shakespeare.txt'
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    with open(data_path) as f:
        text = f.read()

    n_osc = int(sys.argv[2]) if len(sys.argv) > 2 else 96
    n_layers = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
    seq_len = 128
    batch_size = 64

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Resonance — parametric oscillator ({device})")
    print(f"{n_osc} osc, {n_layers} layers, dim={2*n_osc}, seq={seq_len}, batch={batch_size}")

    model = Resonance(n_osc=n_osc, n_layers=n_layers, seq_len=seq_len)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params:,} parameters\n")

    model = train(model, text, steps=steps, batch_size=batch_size,
                  seq_len=seq_len, device=device)
