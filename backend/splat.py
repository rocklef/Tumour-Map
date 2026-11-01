"""Part 2: mask-guided 3D Gaussian fitting of the tumour, exported as a standard 3DGS .ply.

The MRI is already a 3D volume, so the Gaussians are fitted directly to it rather than to
rendered camera views. Each Gaussian has a position, rotation, per-axis scale, a density
amplitude and an MRI intensity. Two fields are fitted together over the voxels in and around
the whole-tumour mask:

  occupancy(x) = 1 - exp(-sum_k a_k G_k(x))                   -> mask (1 inside, 0 in a shell outside)
  intensity(x) = sum_k a_k c_k G_k(x) / sum_k a_k G_k(x)      -> T1ce intensity inside the mask

Gaussians are initialised only inside the mask (mask-guided), and each voxel is evaluated
against its K nearest Gaussian centres so the fit runs on a CPU.

Usage on an already processed study:  python backend/splat.py data/studies/<id>
"""
import gzip
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree

SH_C0 = 0.28209479177387814  # degree-0 spherical-harmonic constant used by 3DGS colour encoding


def quat_to_rot(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(*q.shape[:-1], 3, 3)


class Field(torch.nn.Module):
    def __init__(self, mu, scale, intensity):
        super().__init__()
        n = len(mu)
        self.mu = torch.nn.Parameter(torch.tensor(mu, dtype=torch.float32))
        self.log_s = torch.nn.Parameter(torch.full((n, 3), math.log(scale), dtype=torch.float32))
        self.quat = torch.nn.Parameter(torch.tensor([[1.0, 0, 0, 0]] * n, dtype=torch.float32))
        self.a_raw = torch.nn.Parameter(torch.full((n,), 0.5, dtype=torch.float32))
        c = np.clip(intensity, 0.02, 0.98)
        self.c_raw = torch.nn.Parameter(torch.tensor(np.log(c / (1 - c)), dtype=torch.float32))

    def params(self):
        return self.mu, self.log_s.clamp(-1.5, 2.5).exp(), quat_to_rot(self.quat), torch.nn.functional.softplus(self.a_raw), torch.sigmoid(self.c_raw)

    def evaluate(self, x, nbr):
        """x: (B,3) voxel centres, nbr: (B,K) Gaussian indices. Returns occupancy, intensity."""
        mu, s, R, a, c = self.params()
        d = x[:, None, :] - mu[nbr]                                   # (B,K,3)
        local = torch.einsum("bkji,bkj->bki", R[nbr], d) / s[nbr]     # rotate into each Gaussian's frame
        g = torch.exp(-0.5 * (local * local).sum(-1))                  # (B,K)
        w = a[nbr] * g
        dens = w.sum(-1)
        occ = 1 - torch.exp(-dens)
        inten = (w * c[nbr]).sum(-1) / (dens + 1e-6)
        return occ, inten


def fit(labels, intensity, iters=320, k=10, seed=0, log=lambda s: None):
    """labels: uint8 RAS label map (0 background). intensity: float32 0..1 RAS volume (T1ce).
    Returns dict of Gaussian arrays (voxel coordinates) and fit metrics."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    mask = labels > 0
    nvox = int(mask.sum())
    if nvox < 20:
        return None
    shell = ndimage.binary_dilation(mask, iterations=3) & ~mask
    pin = np.argwhere(mask).astype(np.float32)
    pout = np.argwhere(shell).astype(np.float32)
    inten_in = intensity[mask].astype(np.float32)

    n = int(np.clip(nvox / 6, 400, 40000))
    pick = rng.choice(len(pin), n, replace=False)
    mu0 = pin[pick] + rng.uniform(-0.4, 0.4, (n, 3)).astype(np.float32)
    scale0 = (nvox / n) ** (1 / 3) * 0.55
    field = Field(mu0, scale0, inten_in[pick])
    opt = torch.optim.Adam([
        {"params": [field.mu], "lr": 0.04},
        {"params": [field.log_s], "lr": 0.02},
        {"params": [field.quat], "lr": 0.01},
        {"params": [field.a_raw], "lr": 0.03},
        {"params": [field.c_raw], "lr": 0.03},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=0.002)
    X_in, X_out = torch.from_numpy(pin), torch.from_numpy(pout)
    I_in = torch.from_numpy(inten_in)
    batch = min(32000, len(pin))
    tree = None
    for it in range(iters):
        if it % 40 == 0:
            tree = cKDTree(field.mu.detach().numpy())
            log(f"Fitting Gaussians ({it}/{iters})")
        bi = torch.from_numpy(rng.choice(len(pin), batch, replace=len(pin) < batch))
        bo = torch.from_numpy(rng.choice(len(pout), min(batch // 2, len(pout)), replace=False))
        xb = torch.cat([X_in[bi], X_out[bo]])
        _, nbr = tree.query(xb.numpy(), k=k, workers=-1)
        occ, inten = field.evaluate(xb, torch.from_numpy(nbr))
        target = torch.cat([torch.ones(len(bi)), torch.zeros(len(bo))])
        l_occ = torch.nn.functional.binary_cross_entropy(occ.clamp(1e-5, 1 - 1e-5), target)
        l_int = ((inten[: len(bi)] - I_in[bi]) ** 2).mean()
        _, s, _, a, _ = field.params()
        loss = l_occ + 2.0 * l_int + 1e-4 * (s.max(-1).values / s.min(-1).values).mean()  # mild anisotropy prior
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

    # prune Gaussians that ended up with almost no density
    mu, s, R, a, c = [t.detach() for t in field.params()]
    keep = a > 0.02
    with torch.no_grad():
        field.mu.data, field.log_s.data, field.quat.data = field.mu.data[keep], field.log_s.data[keep], field.quat.data[keep]
        field.a_raw.data, field.c_raw.data = field.a_raw.data[keep], field.c_raw.data[keep]
    mu, s, R, a, c = [t.detach() for t in field.params()]

    # evaluate the fitted field on every voxel of the mask and shell
    tree = cKDTree(mu.numpy())
    allx = np.concatenate([pin, pout])
    occ_all, int_all = [], []
    with torch.no_grad():
        for i in range(0, len(allx), 60000):
            xb = torch.from_numpy(allx[i:i + 60000])
            _, nbr = tree.query(xb.numpy(), k=k, workers=-1)
            o, t = field.evaluate(xb, torch.from_numpy(nbr))
            occ_all.append(o.numpy())
            int_all.append(t.numpy())
    occ_all, int_all = np.concatenate(occ_all), np.concatenate(int_all)
    rec_in = occ_all[: len(pin)] > 0.5
    rec_out = occ_all[len(pin):] > 0.5
    tp, fp = int(rec_in.sum()), int(rec_out.sum())
    dice = 2 * tp / (nvox + tp + fp)
    mse = float(((int_all[: len(pin)] - inten_in) ** 2).mean())
    q = torch.nn.functional.normalize(field.quat.detach(), dim=-1)
    # label each Gaussian by the tumour sub-region at its centre
    ci = np.clip(np.round(mu.numpy()).astype(int), 0, np.array(labels.shape) - 1)
    lab = labels[ci[:, 0], ci[:, 1], ci[:, 2]]
    metrics = {
        "gaussians": int(len(mu)), "iterations": iters, "seconds": round(time.time() - t0, 1),
        "dice_vs_mask": round(dice, 4), "volume_ml_mask": round(nvox / 1000, 2), "volume_ml_splat": round((tp + fp) / 1000, 2),
        "intensity_psnr_db": round(10 * math.log10(1 / max(mse, 1e-8)), 1), "voxels": nvox,
    }
    return {"mu": mu.numpy(), "scale": s.numpy(), "quat": q.numpy(), "a": a.numpy(), "c": c.numpy(), "label": lab}, metrics


def write_ply(path, g, opacity_gain=1.2):
    """Standard 3D Gaussian Splatting PLY (INRIA layout, SH degree 0). Coordinates in voxels (1 mm)."""
    n = len(g["mu"])
    opacity = np.clip(1 - np.exp(-g["a"] * opacity_gain), 1e-4, 1 - 1e-4)
    cols = [
        g["mu"][:, 0], g["mu"][:, 1], g["mu"][:, 2],
        np.zeros(n), np.zeros(n), np.zeros(n),
        *[(g["c"] - 0.5) / SH_C0] * 3,                       # grey MRI intensity as the base colour
        np.log(opacity / (1 - opacity)),                     # opacity stored as a logit
        *np.log(g["scale"]).T,                               # scales stored as logs
        *g["quat"].T,                                        # rotation (w, x, y, z)
    ]
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.stack(cols, 1).astype("<f4")
    header = "ply\nformat binary_little_endian 1.0\ncomment BrainSplat-3D mask-guided tumour Gaussians, units mm (RAS voxel grid)\n"
    header += f"element vertex {n}\n" + "".join(f"property float {nm}\n" for nm in names) + "end_header\n"
    Path(path).write_bytes(header.encode() + data.tobytes())


def write_web(path, g, opacity_gain=1.2):
    """Compact gzip buffer for the viewer: per Gaussian centre(3) scale(3) quat(4) opacity, intensity, label."""
    opacity = 1 - np.exp(-g["a"] * opacity_gain)
    arr = np.concatenate([g["mu"], g["scale"], g["quat"], opacity[:, None], g["c"][:, None], g["label"][:, None].astype(np.float32)], 1).astype("<f4")
    Path(path).write_bytes(gzip.compress(arr.tobytes(), 6))


def run(study_dir: Path, labels, t1ce_u8, log=lambda s: None):
    res = fit(labels, t1ce_u8.astype(np.float32) / 255.0, log=log)
    if res is None:
        return None
    g, metrics = res
    write_ply(study_dir / "splat.ply", g)
    write_web(study_dir / "splat.bin", g)
    metrics["ply_mb"] = round((study_dir / "splat.ply").stat().st_size / 1e6, 2)
    return metrics


if __name__ == "__main__":
    d = Path(sys.argv[1])
    rep = json.loads((d / "report.json").read_text())
    X, Y, Z = rep["shape"]
    load = lambda f: np.frombuffer(gzip.decompress((d / f).read_bytes()), np.uint8).reshape(Z, Y, X).transpose(2, 1, 0)
    m = run(d, load("labels.bin"), load("vol_t1ce.bin"), log=lambda s: print("..", s, flush=True))
    rep["splat"] = m
    (d / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(m, indent=1))
