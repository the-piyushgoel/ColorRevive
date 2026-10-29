# import os
# from torchvision.datasets import CIFAR10

# os.makedirs("data/images", exist_ok=True)
# ds = CIFAR10(root="tmp_cifar", train=True, download=True)
# for i in range(5000):
#     img, _ = ds[i]
#     img.resize((128, 128)).save(f"data/images/{i:05d}.png")
import os, io, requests
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np

N = 3000
OUT = "data/images"
os.makedirs(OUT, exist_ok=True)

def grab(i):
    path = f"{OUT}/{i:05d}.jpg"
    if os.path.exists(path):          # resume: pehle se hai to skip
        return
    try:
        r = requests.get(f"https://picsum.photos/seed/{i}/128/128", timeout=30)
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        a = np.asarray(img).astype(int)
        # black & white photo ho to skip (colourization ke kaam ki nahi)
        if np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean() < 4:
            return
        img.save(path, quality=95)
    except Exception as e:
        print("skip", i, e)

with ThreadPoolExecutor(max_workers=8) as ex:
    list(ex.map(grab, range(N)))

print("done:", len(os.listdir(OUT)), "images")