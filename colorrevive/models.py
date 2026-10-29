"""pix2pix-style conditional GAN: U-Net generator + PatchGAN discriminator."""
import math
import torch
import torch.nn as nn


def depth_for(size):
    """U-Net depth so the bottleneck is 2x2.  size must be a power of two >= 32."""
    d = int(math.log2(size)) - 1
    if 2 ** (d + 1) != size or size < 32:
        raise ValueError("--size must be a power of two >= 32 (e.g. 64, 128, 256)")
    return d


def _init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class Down(nn.Sequential):
    def __init__(self, i, o, norm=True):
        layers = [nn.Conv2d(i, o, 4, 2, 1, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(o))
        layers.append(nn.LeakyReLU(0.2, True))
        super().__init__(*layers)


class Up(nn.Sequential):
    def __init__(self, i, o, dropout=False):
        layers = [nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o)]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(True))
        super().__init__(*layers)


class UNetGenerator(nn.Module):
    """L (1ch) -> ab (2ch). Input side must equal 2**(depth+1)."""

    def __init__(self, in_ch=1, out_ch=2, nf=64, depth=6):
        super().__init__()
        chs = [min(nf * 2 ** i, nf * 8) for i in range(depth)]
        self.downs = nn.ModuleList()
        prev = in_ch
        for i, c in enumerate(chs):
            self.downs.append(Down(prev, c, norm=i > 0))
            prev = c
        self.ups = nn.ModuleList()
        for i in range(depth - 1, 0, -1):
            in_c = chs[i] if i == depth - 1 else chs[i] * 2
            self.ups.append(Up(in_c, chs[i - 1], dropout=i >= depth - 3))
        self.final = nn.Sequential(nn.ConvTranspose2d(chs[0] * 2, out_ch, 4, 2, 1), nn.Tanh())
        self.apply(_init)

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x)
            skips.append(x)
        x = skips[-1]
        for i, u in enumerate(self.ups):
            x = torch.cat([u(x), skips[-2 - i]], 1)
        return self.final(x)


class PatchDiscriminator(nn.Module):
    """Judges overlapping patches of the (L, ab) pair as real/fake."""

    def __init__(self, in_ch=3, nf=64):
        super().__init__()

        def block(i, o, stride, norm=True):
            l = [nn.Conv2d(i, o, 4, stride, 1, bias=not norm)]
            if norm:
                l.append(nn.BatchNorm2d(o))
            l.append(nn.LeakyReLU(0.2, True))
            return l

        self.net = nn.Sequential(
            *block(in_ch, nf, 2, norm=False), *block(nf, nf * 2, 2),
            *block(nf * 2, nf * 4, 2), *block(nf * 4, nf * 8, 1),
            nn.Conv2d(nf * 8, 1, 4, 1, 1))
        self.apply(_init)

    def forward(self, L, ab):
        return self.net(torch.cat([L, ab], 1))


def load_generator(path, device="cpu"):
    """Rebuilds the generator from the settings stored in the checkpoint (size, nf, depth)."""
    ck = torch.load(path, map_location=device)
    G = UNetGenerator(nf=ck.get("nf", 64), depth=ck.get("depth", 6)).to(device)
    G.load_state_dict(ck["G"])
    G.size = ck.get("size", 128)
    return G.eval()
