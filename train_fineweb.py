"""Resonance on FineWeb — scaled training with damped rotation.

Usage (single GPU):
  python train_fineweb.py --data_dir ./data/datasets/fineweb10B_sp1024

Usage (multi-GPU):
  torchrun --standalone --nproc_per_node=2 train_fineweb.py --data_dir ...

Architecture:
  bank(FFT) → [RMSNorm → project(γ,β,c) → damped_rotation → W_mix → SineGate → residual] × L
  → RMSNorm → W_out → listen(tied weights)
"""

import argparse, glob, math, os, time, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# --- Data loading (FineWeb binary shards) ---

def load_shard(path):
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520
    ntok = header[2]
    dtype = np.uint16 if header[1] == 1 else np.uint32
    return torch.from_numpy(np.fromfile(path, dtype=dtype, offset=256*4, count=ntok).astype(np.int32))

class ShardLoader:
    def __init__(self, pattern, seq_len, rank=0, world_size=1):
        self.files = sorted(glob.glob(pattern))
        assert self.files, f"No files: {pattern}"
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.shard_idx = 0
        self.pos = 0
        self.tokens = load_shard(self.files[0])

    def next_batch(self, batch_size):
        B, T = batch_size, self.seq_len
        need = B * T + 1
        if self.pos + need > len(self.tokens):
            self.shard_idx = (self.shard_idx + 1) % len(self.files)
            self.tokens = load_shard(self.files[self.shard_idx])
            self.pos = 0
        buf = self.tokens[self.pos:self.pos + need]
        self.pos += need * self.world_size
        return buf[:-1].reshape(B, T), buf[1:].reshape(B, T)

# --- Model ---

class OscillatorBank(nn.Module):
    def __init__(self, n_osc, seq_len):
        super().__init__()
        self.n_osc = n_osc
        self.fft_len = 2 * seq_len
        freqs = torch.linspace(0.1, math.pi, n_osc)
        gammas = torch.linspace(0.05, 0.20, n_osc)
        t = torch.arange(seq_len, dtype=torch.float32)
        h_pos, h_vel = [], []
        for k in range(n_osc):
            w0, g = freqs[k].item(), gammas[k].item()
            wd = w0 * math.sqrt(max(1e-6, 1.0 - g*g))
            alpha = g * w0
            decay = torch.exp(-alpha * t)
            h_pos.append(decay * torch.sin(wd * t) / max(1e-8, wd))
            h_vel.append(decay * (torch.cos(wd * t) - alpha/max(1e-8, wd) * torch.sin(wd * t)))
        h_pos = F.pad(torch.stack(h_pos), (0, seq_len))
        h_vel = F.pad(torch.stack(h_vel), (0, seq_len))
        self.register_buffer('h_pos_fft', torch.fft.rfft(h_pos))
        self.register_buffer('h_vel_fft', torch.fft.rfft(h_vel))

    def forward(self, drives):
        B, T, K = drives.shape
        d = drives.float().transpose(1, 2)
        d = F.pad(d, (0, self.fft_len - T))
        d_fft = torch.fft.rfft(d)
        pos = torch.fft.irfft(d_fft * self.h_pos_fft, dim=2)[:, :, :T].transpose(1, 2)
        vel = torch.fft.irfft(d_fft * self.h_vel_fft, dim=2)[:, :, :T].transpose(1, 2)
        return torch.cat([pos, vel], dim=-1)

class ResonanceLayer(nn.Module):
    def __init__(self, dim, n_osc):
        super().__init__()
        self.n_osc = n_osc
        self.proj_gamma = nn.Linear(dim, n_osc)
        self.proj_beta = nn.Linear(dim, n_osc)
        self.proj_sense = nn.Linear(dim, n_osc)
        self.w_mix = nn.Linear(dim, dim, bias=False)
        self.mix_scale = nn.Parameter(torch.ones(dim))
        for p in [self.proj_gamma, self.proj_beta, self.proj_sense]:
            nn.init.xavier_uniform_(p.weight, gain=0.1)
            nn.init.zeros_(p.bias)
        nn.init.zeros_(self.w_mix.weight)
        freqs = torch.linspace(0.1, math.pi, n_osc)
        self.register_buffer('cos_w', torch.cos(freqs))
        self.register_buffer('sin_w', torch.sin(freqs))
        self.register_buffer('freqs', freqs)

    def forward(self, state, bank_out):
        B, T, D = state.shape
        n = self.n_osc
        normed = F.rms_norm(state, (D,))
        gamma = torch.sigmoid(self.proj_gamma(normed))
        beta = torch.sigmoid(self.proj_beta(normed))
        sense = torch.sigmoid(self.proj_sense(normed))
        cos_w = self.cos_w.to(state.dtype)
        sin_w = self.sin_w.to(state.dtype)
        freqs = self.freqs.to(state.dtype)
        osc_pos = state.new_zeros(B, n)
        osc_vel = state.new_zeros(B, n)
        outputs = []
        for t in range(T):
            g, b, s = gamma[:, t], beta[:, t], sense[:, t]
            drv = bank_out[:, t]
            new_pos = g * (osc_pos * cos_w + osc_vel * sin_w / freqs) + (1-g) * b * drv[:, :n]
            new_vel = g * (osc_vel * cos_w - osc_pos * freqs * sin_w) + (1-g) * b * drv[:, n:]
            osc_pos, osc_vel = new_pos, new_vel
            outputs.append(torch.cat([s * osc_pos, s * osc_vel], dim=-1))
        osc_out = torch.stack(outputs, dim=1)
        mixed = self.w_mix(osc_out)
        activated = mixed * torch.sin(mixed)
        return state + self.mix_scale.to(state.dtype)[None, None, :] * activated

class Resonance(nn.Module):
    def __init__(self, n_osc=256, n_layers=12, seq_len=512, vocab_size=1024):
        super().__init__()
        self.n_osc = n_osc
        self.dim = 2 * n_osc
        self.vocab_size = vocab_size
        self.drive = nn.Embedding(vocab_size, self.dim)
        nn.init.normal_(self.drive.weight, std=0.02)
        self.bank = OscillatorBank(n_osc, seq_len)
        self.layers = nn.ModuleList([ResonanceLayer(self.dim, n_osc) for _ in range(n_layers)])
        self.w_out = nn.Linear(self.dim, self.dim, bias=False)
        nn.init.xavier_uniform_(self.w_out.weight, gain=0.1)

    def forward(self, input_ids, target_ids):
        emb = self.drive(input_ids)
        drives = emb[:, :, :self.n_osc]
        bank_out = self.bank(drives)
        state = F.rms_norm(bank_out, (bank_out.size(-1),))
        for layer in self.layers:
            state = layer(state, bank_out)
        x = F.rms_norm(state, (state.size(-1),))
        x = self.w_out(x)
        logits = F.linear(x, self.drive.weight)
        return F.cross_entropy(logits.reshape(-1, self.vocab_size), target_ids.reshape(-1))

# --- Training ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/datasets/fineweb10B_sp1024')
    parser.add_argument('--n_osc', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--vocab_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--steps', type=int, default=10000)
    args = parser.parse_args()

    ddp = 'RANK' in os.environ
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if ddp:
        dist.init_process_group('nccl')
    device = torch.device(f'cuda:{int(os.environ.get("LOCAL_RANK", 0))}')
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')

    if rank == 0:
        print(f"Resonance — damped rotation on FineWeb")
        print(f"{args.n_osc} osc, {args.n_layers} layers, dim={2*args.n_osc}, "
              f"seq={args.seq_len}, vocab={args.vocab_size}, batch={args.batch_size}")

    model = Resonance(args.n_osc, args.n_layers, args.seq_len, args.vocab_size).to(device).bfloat16()
    if ddp:
        model = DDP(model, device_ids=[device.index])
    raw = model.module if ddp else model

    n_params = sum(p.numel() for p in raw.parameters())
    if rank == 0:
        print(f"Parameters: {n_params:,}\n")

    optimizer = torch.optim.AdamW(raw.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                      total_steps=args.steps, pct_start=0.05)

    train_pattern = os.path.join(args.data_dir, 'fineweb_train_*.bin')
    loader = ShardLoader(train_pattern, args.seq_len, rank, world_size)

    t0 = time.time()
    for step in range(args.steps):
        x, y = loader.next_batch(args.batch_size)
        x = x.to(device, dtype=torch.long)
        y = y.to(device, dtype=torch.long)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if rank == 0 and step % 100 == 0:
            elapsed = time.time() - t0
            sps = (step + 1) / elapsed if elapsed > 0 else 0
            bpc = loss.item() / math.log(2)
            print(f"step {step:5d}  loss {loss.item():.3f}  bpc {bpc:.3f}  "
                  f"lr {scheduler.get_last_lr()[0]:.1e}  [{sps:.1f} s/s]", flush=True)

    if rank == 0:
        torch.save(raw.state_dict(), 'resonance_fineweb.pt')
        print(f"\nDone. Saved to resonance_fineweb.pt")

    if ddp:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
