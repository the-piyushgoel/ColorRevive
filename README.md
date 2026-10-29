# ColorRevive: Automatic Image Colorization with a Conditional GAN

pix2pix-style cGAN that colorizes grayscale photos. Deep Learning & Generative AI, CA-2.

## How it works
- **Colour space:** images are converted to CIE Lab. The generator receives the lightness channel **L** and predicts the colour channels **ab**. Output = original L + predicted ab, so structure is never altered.
- **Generator:** U-Net (6 down / 5 up blocks + skip connections, dropout in the deepest layers), `L (1x128x128) -> ab (2x128x128)`.
- **Discriminator:** 70x70 PatchGAN on the pair `(L, ab)` – it judges whether colours are plausible *given* the grayscale image (the "conditional" part).
- **Loss:** `L_G = BCE_adv + 100 * L1(ab_fake, ab_real)`; `L_D = 0.5 * (BCE(real,1) + BCE(fake,0))`. Adam, lr 2e-4, betas (0.5, 0.999).
- **Dataset:** either any folder of colour photos in `data/images/` (works offline, e.g. COCO val2017) or STL-10 (natural photos, 96px, resized to 128). Train = 5k labelled + 20k unlabelled images; validation = first 500 test images; test = 8,000 test images. Auto-downloaded by torchvision. Labels are never used.
- **Baselines:** (1) *grayscale copy* (ab = 0); (2) *L1-only U-Net*, the same generator trained without the discriminator.
- **Metrics:** PSNR and SSIM against the ground-truth RGB image, plus Hasler-Süsstrunk colourfulness (explained below).

## Data (important)
Put **at least a few thousand colour photos** in `data/images/` (sub-folders are fine), e.g. unzip COCO val2017 (https://images.cocodataset.org/zips/val2017.zip, 5,000 photos) there. 10% are held out as the test set. If `data/images` is empty the scripts try to download STL-10 instead (needs access to cs.stanford.edu).

## Run it
Always use `python -m pip ...` so packages go to the same Python you run the scripts with.

```bash
pip install -r requirements.txt

python train.py --mode gan --epochs 30        # ColorRevive cGAN  -> runs/gan/
# CPU-only laptop: python train.py --mode gan --size 64 --nf 32 --bs 16 --epochs 20 (and same for --mode l1)
python train.py --mode l1  --epochs 30        # baseline          -> runs/l1/
python evaluate.py                            # -> results/*.csv, *.png
streamlit run app.py                          # web demo at localhost:8501
```
Useful flags: `--bs 16` (low VRAM), `--n_unlabeled 0` (smaller/faster, ~5k images), `--workers 0` (Windows issues), `--epochs 5` (quick smoke test).
Rough timing: ~2-4 min/epoch on a Colab T4 at the default settings; CPU-only is very slow (use `--n_unlabeled 0 --epochs 10` and a smaller subset).
STL-10 downloads ~2.6 GB the first time (the unlabelled split is loaded into RAM, ~3 GB; use `--n_unlabeled 0` to skip it).

## Outputs for your report/slides
| File | Use |
|---|---|
| `runs/*/history.json`, `results/training_curves.png` | training graphs (losses, validation L1) |
| `runs/gan/samples/epoch_XXX.png` | colourization improving over epochs (rows: gray / predicted / truth) |
| `results/metrics.csv`, `results/metrics_bar.png` | PSNR, SSIM, colourfulness table + chart |
| `results/comparison_grid.png` | gray vs baselines vs cGAN vs ground truth |

## Reading the results honestly
PSNR/SSIM reward being *close to the original pixels*, not looking good. An L1-only model hedges: when a car could be red or blue it predicts brownish grey, which scores well on PSNR but looks washed out. The cGAN commits to vivid colours, which can cost a little PSNR/SSIM yet looks more realistic. This is why the evaluation also reports **colourfulness** and the visual grid. Report whatever numbers you actually get, and discuss this trade-off in the viva rather than only quoting one metric. Both learned models should beat the grayscale-copy baseline on PSNR and SSIM.

## Limitations / future work
Small 128px model trained on STL-10 (so best on objects, animals, landscapes); colours are plausible, not guaranteed correct (the task is ambiguous); the demo upsamples predicted chroma and reuses the original luminance for full-resolution output. Ideas: larger dataset (Places365 / ImageNet subset), perceptual or LPIPS metrics, user colour hints, spectral norm / WGAN-GP for stability.

## Structure
```
train.py  evaluate.py  app.py
colorrevive/{models,data,utils}.py
```
