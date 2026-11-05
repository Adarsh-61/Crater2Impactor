"""OT-FMPE: Kaggle training (DDP over every visible GPU, one run per seed) and posterior sampling of the test contexts.

Architecture and recipe as the audited corrected OT-FMPE: AdaLN residual velocity field (hidden 256, 4 blocks, 64-d
sinusoidal time embedding, one dummy class), linear-path flow matching from a uniform base on [-1, 1]^4, RK4 sampling
with 20 steps; AdamW lr 1.5e-3, weight decay 1e-5, cosine schedule, gradient clipping 5, FP16 autocast, early stopping
(patience 25, at least 20 epochs, at most 120), batch 2048.
Training data: research/otfmpe/gen.py (the engine's current generative model). The parameter box is set from the
training data (range plus a margin), because craters up to 5 km from slow, light spacecraft need masses above 1e14 kg.
Inputs (Kaggle dataset): train.npz, val.npz, tests.npz. Outputs per seed: fmpe_seed{S}.pt, history_seed{S}.json,
test_samples_seed{S}.npz (posterior samples of every test case); run_info.json (GPU inventory, timings).
"""
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

SEEDS = [int(v) for v in os.environ.get("C2I_SEEDS", "0,1,2").split(",")]
CFG = dict(hidden_dim=256, n_blocks=4, time_embed_dim=64, batch_size=2048, max_epochs=120, lr=1.5e-3,
           weight_decay=1e-5, patience=25, min_epochs=20, clip=5.0, rk4_steps=20, n_test_samples=2000)
if os.environ.get("C2I_SMOKE"):
    CFG.update(max_epochs=2, min_epochs=1, n_test_samples=50)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t):
        half = self.embed_dim // 2
        w = torch.exp(torch.arange(half, device=t.device, dtype=t.dtype) * -(math.log(10000.0) / float(half - 1)))
        a = t * w.unsqueeze(0)
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)


class AdaLNResidualBlock(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.fc1, self.fc2, self.act = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.GELU()
        self.modulation = nn.Sequential(nn.GELU(), nn.Linear(cond_dim, 3 * dim))
        nn.init.zeros_(self.modulation[1].weight); nn.init.zeros_(self.modulation[1].bias)

    def forward(self, x, cond):
        g, b, a = self.modulation(cond).chunk(3, dim=-1)
        return x + a * self.fc2(self.act(self.fc1(self.norm(x) * (1.0 + g) + b)))


class VelocityFieldNet(nn.Module):
    def __init__(self, dim, context_dim, n_classes=1, hidden_dim=256, n_blocks=4, time_embed_dim=64):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.class_embed = nn.Embedding(n_classes, 16)
        self.cond_net = nn.Sequential(nn.Linear(time_embed_dim + context_dim + 16, hidden_dim), nn.GELU(),
                                      nn.Linear(hidden_dim, hidden_dim))
        self.input_proj = nn.Linear(dim, hidden_dim)
        self.blocks = nn.ModuleList([AdaLNResidualBlock(hidden_dim, hidden_dim) for _ in range(n_blocks)])
        self.output_proj = nn.Linear(hidden_dim, dim)

    def forward(self, u, t, context, class_idx):
        cond = self.cond_net(torch.cat([self.time_embed(t), context, self.class_embed(class_idx)], dim=-1))
        h = F.gelu(self.input_proj(u))
        for blk in self.blocks:
            h = blk(h, cond)
        return self.output_proj(h)


class CorrectedOTFMPE(nn.Module):
    def __init__(self, dim=4, context_dim=8, x_mean=None, x_std=None, box_lo=None, box_hi=None, **kw):
        super().__init__()
        self.dim, self.context_dim = dim, context_dim
        self.velocity_net = VelocityFieldNet(dim, context_dim, **kw)
        self.register_buffer("box_lo", torch.zeros(dim) if box_lo is None else torch.as_tensor(box_lo))
        self.register_buffer("box_hi", torch.ones(dim) if box_hi is None else torch.as_tensor(box_hi))
        self.register_buffer("x_mean", torch.zeros(context_dim) if x_mean is None else torch.as_tensor(x_mean))
        self.register_buffer("x_std", torch.ones(context_dim) if x_std is None else torch.as_tensor(x_std))

    def norm_x(self, x):
        return (x - self.x_mean) / self.x_std

    def to_u(self, th):
        return 2.0 * (th - self.box_lo) / (self.box_hi - self.box_lo) - 1.0

    def loss(self, th, x):
        u1 = self.to_u(th)
        u0 = torch.rand_like(u1) * 2.0 - 1.0
        t = torch.rand((th.shape[0], 1), device=th.device)
        cls = torch.zeros(th.shape[0], dtype=torch.long, device=th.device)
        pred = self.velocity_net((1.0 - t) * u0 + t * u1, t, self.norm_x(x), cls)
        return F.mse_loss(pred.float(), (u1 - u0).float())

    def forward(self, th, x):
        return self.loss(th, x)

    @torch.no_grad()
    def sample(self, x, n, steps=20, gen=None):
        b = x.shape[0]
        xf = self.norm_x(x).repeat_interleave(n, dim=0)
        cls = torch.zeros(b * n, dtype=torch.long, device=x.device)
        u = torch.rand((b * n, self.dim), device=x.device, generator=gen) * 2.0 - 1.0
        dt = 1.0 / steps
        f = lambda uu, tt: self.velocity_net(uu, torch.full((b * n, 1), tt, device=x.device), xf, cls)  # noqa: E731
        for k in range(steps):
            t0 = k * dt
            v1 = f(u, t0); v2 = f(u + 0.5 * dt * v1, t0 + 0.5 * dt)
            v3 = f(u + 0.5 * dt * v2, t0 + 0.5 * dt); v4 = f(u + dt * v3, t0 + dt)
            u = u + dt / 6.0 * (v1 + 2 * v2 + 2 * v3 + v4)
        u = u.clamp(-1.0, 1.0)
        th = self.box_lo + (u + 1.0) / 2.0 * (self.box_hi - self.box_lo)
        return th.view(b, n, self.dim)


def gpu_inventory():
    info = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    try:
        info["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,utilization.gpu",
                                             "--format=csv"], capture_output=True, text=True, timeout=30).stdout
    except Exception as e:  # noqa: BLE001
        info["nvidia_smi"] = f"unavailable: {e}"
    return info


def data_box(theta):
    """Parameter box from the training targets: range plus 0.5 (log quantities) or the exact [0, 1] for elev/90."""
    lo, hi = theta.min(0) - 0.5, theta.max(0) + 0.5
    lo[2], hi[2] = 0.0, 1.0
    return lo.astype(np.float32), hi.astype(np.float32)


def worker(rank, world, data_dir, out_dir, seed):
    if world > 1:
        os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = str(29533 + seed)
        dist.init_process_group("nccl", rank=rank, world_size=world)
    dev = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    torch.manual_seed(1000 * seed + rank)
    tr, va = np.load(data_dir / "train.npz"), np.load(data_dir / "val.npz")
    xm, xs = tr["x"].mean(0), tr["x"].std(0) + 1e-6
    xt, tt = torch.tensor(tr["x"], device=dev), torch.tensor(tr["theta"], device=dev)
    xv, tv = torch.tensor(va["x"], device=dev), torch.tensor(va["theta"], device=dev)
    blo, bhi = data_box(tr["theta"])
    model = CorrectedOTFMPE(x_mean=xm, x_std=xs, box_lo=blo, box_hi=bhi, hidden_dim=CFG["hidden_dim"],
                            n_blocks=CFG["n_blocks"], time_embed_dim=CFG["time_embed_dim"]).to(dev)
    net = nn.parallel.DistributedDataParallel(model, device_ids=[rank]) if world > 1 else model
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG["max_epochs"], eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    per_rank = CFG["batch_size"] // world
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "epoch_seconds": [], "ranks_used": world}
    best, best_ep, bad, t_start = float("inf"), -1, 0, time.time()
    gval = torch.Generator(device=dev); n = xt.shape[0]
    for ep in range(1, CFG["max_epochs"] + 1):
        te = time.time(); model.train()
        g = torch.Generator(device=dev); g.manual_seed(100_000 * seed + ep)
        perm = torch.randperm(n, device=dev, generator=g)[rank::world]
        tot, nb = 0.0, 0
        for i in range(0, len(perm) - per_rank + 1, per_rank):
            idx = perm[i:i + per_rank]
            with torch.autocast("cuda", enabled=dev.type == "cuda", dtype=torch.float16):
                loss = net(tt[idx], xt[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
            scaler.step(opt); scaler.update()
            tot += float(loss.detach()); nb += 1
        sched.step()
        model.eval(); gval.manual_seed(12345); torch.manual_seed(12345)
        with torch.no_grad():
            vl = float(np.mean([float(model.loss(tv[j:j + 8192], xv[j:j + 8192])) for j in range(0, len(tv), 8192)]))
        tl = torch.tensor([tot / max(nb, 1)], device=dev)
        if world > 1:
            dist.all_reduce(tl); tl /= world
        hist["epoch"].append(ep); hist["train_loss"].append(float(tl)); hist["val_loss"].append(vl)
        hist["epoch_seconds"].append(time.time() - te)
        if vl < best - 1e-5:
            best, best_ep, bad = vl, ep, 0
            if rank == 0:
                torch.save({"state_dict": model.state_dict(), "cfg": CFG, "context_dim": 8, "best_epoch": ep, "seed": seed},
                           out_dir / f"fmpe_seed{seed}.pt")
        else:
            bad += 1
        if rank == 0:
            print(f"epoch {ep:3d} train {float(tl):.4f} val {vl:.4f} best {best:.4f} (ep {best_ep}) {time.time() - te:.1f}s", flush=True)
        if bad >= CFG["patience"] and ep >= CFG["min_epochs"]:
            break
    hist.update(best_epoch=best_ep, best_val=best, train_seconds=time.time() - t_start,
                n_train=int(n), n_val=int(len(tv)), per_rank_batch=per_rank, device=str(dev))
    if world > 1:
        dist.barrier(); dist.destroy_process_group()
    if rank == 0:
        (out_dir / f"history_seed{seed}.json").write_text(json.dumps(hist))


def sample_tests(out_dir, data_dir, seed):
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ck = torch.load(out_dir / f"fmpe_seed{seed}.pt", map_location=dev)
    model = CorrectedOTFMPE(hidden_dim=CFG["hidden_dim"], n_blocks=CFG["n_blocks"], time_embed_dim=CFG["time_embed_dim"]).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    xs = torch.tensor(np.load(data_dir / "tests.npz")["x"], device=dev)
    g = torch.Generator(device=dev); g.manual_seed(2026 + seed)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time(); out = []
    for j in range(0, len(xs), 64):
        out.append(model.sample(xs[j:j + 64], CFG["n_test_samples"], CFG["rk4_steps"], gen=g).cpu().numpy())
    sample_s = time.time() - t0
    np.savez_compressed(out_dir / f"test_samples_seed{seed}.npz", samples=np.concatenate(out).astype(np.float32))
    # single-crater latency on one GPU (batch of one context, 2000 samples, 20 steps), after warm-up
    one = xs[:1]
    model.sample(one, CFG["n_test_samples"], CFG["rk4_steps"], gen=g)
    lat = []
    for j in range(20):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time(); model.sample(xs[j:j + 1], CFG["n_test_samples"], CFG["rk4_steps"], gen=g)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        lat.append(time.time() - t1)
    return {"seed": seed, "best_epoch": ck["best_epoch"], "sampling_seconds_total": sample_s,
            "sampling_ms_per_case_batched64": 1000 * sample_s / len(xs),
            "latency_ms_single_case_median": 1000 * float(np.median(lat)),
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
            "box_lo": model.box_lo.tolist(), "box_hi": model.box_hi.tolist()}


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else next(p.parent for p in Path("/kaggle/input").rglob("train.npz"))
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/kaggle/working")
    out_dir.mkdir(parents=True, exist_ok=True)
    inv = gpu_inventory(); print(json.dumps(inv, indent=1), flush=True)
    world = max(1, torch.cuda.device_count())
    runs = []
    for seed in SEEDS:
        t0 = time.time()
        if world > 1:
            mp.spawn(worker, args=(world, data_dir, out_dir, seed), nprocs=world, join=True)
        else:
            worker(0, 1, data_dir, out_dir, seed)
        hist = json.loads((out_dir / f"history_seed{seed}.json").read_text())
        info = sample_tests(out_dir, data_dir, seed)
        info.update(train_seconds=hist["train_seconds"], best_val=hist["best_val"], wall_seconds=time.time() - t0)
        runs.append(info); print(json.dumps(info), flush=True)
        (out_dir / "run_info.json").write_text(json.dumps({"gpu_inventory": inv, "world_size": world, "cfg": CFG,
                                                            "runs": runs}, indent=1))


if __name__ == "__main__":
    main()
