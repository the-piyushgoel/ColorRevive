"""Train ColorRevive.  --mode gan : cGAN (adversarial + L1)   --mode l1 : same U-Net, L1 only (baseline).

CPU-only laptop?  python train.py --mode gan --size 64 --nf 32 --bs 16 --epochs 20
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from colorrevive.data import get_datasets
from colorrevive.models import UNetGenerator, PatchDiscriminator, depth_for
from colorrevive.utils import lab_tensors_to_rgb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["gan", "l1"], default="gan")
    p.add_argument("--dataset", choices=["auto", "folder", "stl10"], default="auto")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--size", type=int, default=128, help="image side, power of two (64 is faster on CPU)")
    p.add_argument("--nf", type=int, default=64, help="base channel width (32 = ~4x cheaper on CPU)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda_l1", type=float, default=100.0)
    p.add_argument("--n_unlabeled", type=int, default=20000, help="STL-10 only")
    p.add_argument("--max_train", type=int, default=None, help="cap number of training images")
    p.add_argument("--workers", type=int, default=0 if os.name == "nt" else 4)
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out", default="runs")
    a = p.parse_args()

    depth = depth_for(a.size)
    torch.manual_seed(0); np.random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    out = Path(a.out) / a.mode; (out / "samples").mkdir(parents=True, exist_ok=True)
    print(f"device={dev}  mode={a.mode}  size={a.size}  nf={a.nf}")
    if dev == "cpu":
        print("[note] no GPU found - training will be slow. Consider --size 64 --nf 32 --bs 16, or Google Colab (free GPU).")

    train_ds, val_ds, _ = get_datasets(a.dataset, a.data_dir, a.size, a.n_unlabeled, max_train=a.max_train)
    bs = min(a.bs, len(train_ds))
    if bs < 2:
        raise SystemExit("Need at least 2 training images.")
    if bs != a.bs:
        print(f"[note] batch size reduced to {bs} (dataset smaller than requested batch)")
    tl = DataLoader(train_ds, bs, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=dev == "cuda")
    vl = DataLoader(val_ds, 50, num_workers=a.workers)
    print(f"train={len(train_ds)}  val={len(val_ds)}  batches/epoch={len(tl)}")

    G = UNetGenerator(nf=a.nf, depth=depth).to(dev)
    oG = torch.optim.Adam(G.parameters(), a.lr, betas=(0.5, 0.999))
    if a.mode == "gan":
        D = PatchDiscriminator(nf=a.nf).to(dev)
        oD = torch.optim.Adam(D.parameters(), a.lr, betas=(0.5, 0.999))
    bce, l1 = nn.BCEWithLogitsLoss(), nn.L1Loss()
    lam = a.lambda_l1 if a.mode == "gan" else 1.0
    fixed_L, fixed_ab = next(iter(vl)); fixed_L, fixed_ab = fixed_L[:8], fixed_ab[:8]
    hist = []

    def save_samples(ep):
        G.eval()
        with torch.no_grad():
            fake = G(fixed_L.to(dev))
        g = lab_tensors_to_rgb(fixed_L, torch.zeros_like(fixed_ab))
        f = lab_tensors_to_rgb(fixed_L, fake); r = lab_tensors_to_rgb(fixed_L, fixed_ab)
        rows = [np.concatenate(list(x), 1) for x in (g, f, r)]
        Image.fromarray((np.concatenate(rows, 0) * 255).astype(np.uint8)).save(out / "samples" / f"epoch_{ep:03d}.png")
        G.train()

    for ep in range(1, a.epochs + 1):
        G.train(); t0 = time.time(); s = dict(G=0.0, D=0.0, L1=0.0, adv=0.0); n = 0
        for L, ab in tqdm(tl, desc=f"epoch {ep}/{a.epochs}", leave=False):
            L, ab = L.to(dev), ab.to(dev)
            fake = G(L)
            if a.mode == "gan":
                oD.zero_grad()
                pr = D(L, ab)
                lD = 0.5 * (bce(pr, torch.ones_like(pr)) + bce(D(L, fake.detach()), torch.zeros_like(pr)))
                lD.backward(); oD.step(); s["D"] += lD.item()
            oG.zero_grad()
            l_l1 = l1(fake, ab)
            if a.mode == "gan":
                pf = D(L, fake); adv = bce(pf, torch.ones_like(pf))
                lG = adv + lam * l_l1; s["adv"] += adv.item()
            else:
                lG = l_l1
            lG.backward(); oG.step()
            s["G"] += lG.item(); s["L1"] += l_l1.item(); n += 1
        G.eval(); vl1 = 0.0
        with torch.no_grad():
            for L, ab in vl:
                vl1 += l1(G(L.to(dev)), ab.to(dev)).item() * len(L)
        vl1 /= len(val_ds)
        rec = {"epoch": ep, **{k: v / n for k, v in s.items()}, "val_L1": vl1, "time": time.time() - t0}
        hist.append(rec); print({k: round(v, 4) for k, v in rec.items()})
        save_samples(ep)
        torch.save({"G": G.state_dict(), "depth": depth, "nf": a.nf, "size": a.size,
                    "mode": a.mode, "epoch": ep}, out / "G.pt")
        json.dump(hist, open(out / "history.json", "w"), indent=1)
    print("saved ->", out / "G.pt")


if __name__ == "__main__":      # required on Windows when DataLoader workers are used
    main()
