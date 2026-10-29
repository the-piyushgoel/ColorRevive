"""Evaluate cGAN vs L1-only baseline vs grayscale-copy baseline on the held-out test set.
Produces: results/metrics.csv, metrics_bar.png, training_curves.png, comparison_grid.png"""
import argparse, json, os
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from colorrevive.data import get_datasets
from colorrevive.models import load_generator
from colorrevive.utils import lab_tensors_to_rgb, psnr, ssim, colorfulness


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs"); p.add_argument("--out", default="results")
    p.add_argument("--data_dir", default="data"); p.add_argument("--dataset", default="auto")
    p.add_argument("--max_images", type=int, default=8000); p.add_argument("--n_show", type=int, default=8)
    p.add_argument("--workers", type=int, default=0 if os.name == "nt" else 4)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    models = {}
    for name, key in [("L1-only U-Net (baseline)", "l1"), ("ColorRevive cGAN", "gan")]:
        pth = Path(a.runs) / key / "G.pt"
        if pth.exists(): models[name] = load_generator(pth, dev)
        else: print(f"[warn] {pth} missing - skipping {name}")
    size = next(iter(models.values())).size if models else 128
    if len({G.size for G in models.values()}) > 1:
        print("[warn] the two models were trained at different --size; evaluating at", size)

    _, _, test = get_datasets(a.dataset, a.data_dir, size, n_unlabeled=0)
    test = Subset(test, range(min(a.max_images, len(test))))
    dl = DataLoader(test, 50, num_workers=a.workers)

    methods = ["Grayscale copy (baseline)"] + list(models) + ["Ground truth"]
    acc = {m: {"PSNR": [], "SSIM": [], "Colorfulness": []} for m in methods}
    grid = {}
    with torch.no_grad():
        for bi, (L, ab) in enumerate(tqdm(dl, desc="evaluating")):
            gt = lab_tensors_to_rgb(L, ab)
            outs = {"Grayscale copy (baseline)": lab_tensors_to_rgb(L, torch.zeros_like(ab)), "Ground truth": gt}
            for n, G in models.items(): outs[n] = lab_tensors_to_rgb(L, G(L.to(dev)))
            for m, arr in outs.items():
                for x, y in zip(arr, gt):
                    acc[m]["PSNR"].append(psnr(x, y) if m != "Ground truth" else np.nan)
                    acc[m]["SSIM"].append(ssim(x, y) if m != "Ground truth" else np.nan)
                    acc[m]["Colorfulness"].append(colorfulness(x))
            if bi == 0: grid = {m: v[:a.n_show] for m, v in outs.items()}

    rows = []
    for m in methods:
        d = acc[m]; rows.append({"Method": m, "PSNR (dB)": np.nanmean(d["PSNR"]) if m != "Ground truth" else np.nan,
                                 "SSIM": np.nanmean(d["SSIM"]) if m != "Ground truth" else np.nan,
                                 "Colorfulness": np.mean(d["Colorfulness"]), "N": len(d["Colorfulness"])})
    df = pd.DataFrame(rows); df.to_csv(out / "metrics.csv", index=False)
    print("\n", df.round(3).to_string(index=False))

    mm = df[df.Method != "Ground truth"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for axi, c in zip(ax, ["PSNR (dB)", "SSIM", "Colorfulness"]):
        axi.bar(range(len(mm)), mm[c], color=["#999", "#5b8def", "#e0953a"][:len(mm)])
        axi.set_xticks(range(len(mm))); axi.set_xticklabels([s.split(" (")[0] for s in mm.Method], rotation=15, fontsize=8)
        axi.set_title(c)
        if c == "Colorfulness":
            axi.axhline(df[df.Method == "Ground truth"][c].iloc[0], ls="--", c="k", lw=1, label="ground truth"); axi.legend()
    plt.tight_layout(); plt.savefig(out / "metrics_bar.png", dpi=150); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for key, c in [("gan", "#e0953a"), ("l1", "#5b8def")]:
        f = Path(a.runs) / key / "history.json"
        if not f.exists(): continue
        h = pd.DataFrame(json.load(open(f)))
        ax[0].plot(h.epoch, h.L1, c=c, label=f"{key} train L1"); ax[0].plot(h.epoch, h.val_L1, c=c, ls="--", label=f"{key} val L1")
        if key == "gan":
            ax[1].plot(h.epoch, h.G, label="Generator total"); ax[1].plot(h.epoch, h.adv, label="Generator adversarial"); ax[1].plot(h.epoch, h.D, label="Discriminator")
    ax[0].set_title("L1 loss on ab channels"); ax[1].set_title("cGAN losses")
    for x in ax: x.set_xlabel("epoch"); x.legend(); x.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(out / "training_curves.png", dpi=150); plt.close()

    n_show = len(next(iter(grid.values())))
    order = ["Grayscale copy (baseline)"] + list(models) + ["Ground truth"]
    fig, ax = plt.subplots(len(order), n_show, figsize=(2 * n_show, 2.1 * len(order)), squeeze=False)
    for r, m in enumerate(order):
        for c in range(n_show):
            ax[r, c].imshow(grid[m][c]); ax[r, c].axis("off")
        ax[r, 0].set_title(m, loc="left", fontsize=9)
    plt.tight_layout(); plt.savefig(out / "comparison_grid.png", dpi=150); plt.close()
    print("saved to", out)


if __name__ == "__main__":
    main()
