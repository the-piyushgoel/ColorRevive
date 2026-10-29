"""ColorRevive web demo:  streamlit run app.py"""
import io
from pathlib import Path
import numpy as np, streamlit as st, torch
from PIL import Image
from colorrevive.models import load_generator
from colorrevive.utils import colorize, psnr, ssim

st.set_page_config(page_title="ColorRevive", page_icon="🎞️", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=IBM+Plex+Sans:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'Fraunces', serif !important; letter-spacing: -0.01em; }
.block-container { max-width: 1250px; padding-top: 2.2rem; }
.tag { color:#9aa3ad; font-size:.95rem; margin-top:-.6rem; margin-bottom:1.4rem; }
.cap { color:#9aa3ad; font-size:.85rem; margin-bottom:.3rem; }
</style>""", unsafe_allow_html=True)

st.title("ColorRevive")
st.markdown('<div class="tag">Upload a black-and-white photo. A conditional GAN predicts its colours.</div>', unsafe_allow_html=True)

dev = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = {"ColorRevive cGAN": "runs/gan/G.pt", "L1-only baseline": "runs/l1/G.pt"}
avail = {k: v for k, v in MODELS.items() if Path(v).exists()}
if not avail:
    st.error("No trained model found. Run `python train.py --mode gan` first (see README).")
    st.stop()

@st.cache_resource
def get_model(path): return load_generator(path, dev)

with st.sidebar:
    st.header("Settings")
    name = st.selectbox("Model", list(avail))
    boost = st.slider("Colour intensity", 0.5, 1.5, 1.0, 0.05,
                      help="Scales the predicted colour channels after inference. 1.0 = raw model output.")
    st.caption(f"Running on: {dev.upper()}")

f = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
if f is None:
    st.info("Drop a JPG or PNG above. Colour photos work too: they are converted to grayscale first, "
            "and the result is scored against your original.")
    st.stop()

img = Image.open(f)
with st.spinner("Colorizing…"):
    G = get_model(avail[name])
    gray, color, orig = colorize(G, img, dev, size=G.size, ab_scale=boost)

def show(col, im):   # works across Streamlit versions
    try: col.image(im, width="stretch")
    except Exception: col.image(im, use_container_width=True)

c1, c2 = st.columns(2)
c1.markdown('<div class="cap">Grayscale input</div>', unsafe_allow_html=True); show(c1, gray)
c2.markdown(f'<div class="cap">Colorized by {name}</div>', unsafe_allow_html=True); show(c2, color)

def png(im):
    b = io.BytesIO(); im.save(b, "PNG"); return b.getvalue()

d1, d2, d3 = st.columns(3)
d1.download_button("Download colorized", png(color), "colorized.png", "image/png", type="primary")
d2.download_button("Download grayscale", png(gray), "grayscale.png", "image/png")
sbs = Image.new("RGB", (gray.width * 2, gray.height)); sbs.paste(gray, (0, 0)); sbs.paste(color, (gray.width, 0))
d3.download_button("Download side by side", png(sbs), "comparison.png", "image/png")

# Only meaningful if the upload was actually a colour photo
a_ = np.asarray(orig).astype(int)
if np.abs(a_[..., 0] - a_[..., 1]).mean() > 2 or np.abs(a_[..., 1] - a_[..., 2]).mean() > 2:
    st.subheader("Compared with your original")
    o = np.asarray(orig) / 255.0; c = np.asarray(color) / 255.0; g = np.asarray(gray) / 255.0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PSNR (this model)", f"{psnr(c, o):.2f} dB"); m2.metric("SSIM (this model)", f"{ssim(c, o):.3f}")
    m3.metric("PSNR (grayscale copy)", f"{psnr(g, o):.2f} dB"); m4.metric("SSIM (grayscale copy)", f"{ssim(g, o):.3f}")
    st.image(orig, caption="Original", width=380)
