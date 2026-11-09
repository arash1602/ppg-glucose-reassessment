"""Re-implementation of two published PPG foundation-model encoders, written so that the
released weights load by parameter name (weights are read with torch.load(weights_only=True)).

PaPaGei-S  (Pillai et al., ICLR 2025; weights: Zenodo 10.5281/zenodo.13983110, BSD-3-Clause)
  1-D ResNet, 18 blocks, 32 base filters, kernel 3; input: 10-s segment at 125 Hz, z-scored;
  embedding: 512-d linear projection of the average-pooled feature map (the model's first output).
Pulse-PPG  (Saha, Xu et al., IMWUT 2025; weights: Zenodo 10.5281/zenodo.17345536, MIT)
  1-D ResNet, 12 blocks, 128 base filters, kernel 11, instance-normalised input at 50 Hz;
  embedding: 512-d max-pooled feature map.
Only the encoders are re-implemented; the pre-training heads are not needed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvSame(nn.Module):
    def __init__(self, cin, cout, k, stride):
        super().__init__()
        self.k, self.stride = k, stride
        self.conv = nn.Conv1d(cin, cout, k, stride)

    def forward(self, x):
        n = x.shape[-1]
        p = max(0, ((n + self.stride - 1) // self.stride - 1) * self.stride + self.k - n)
        return self.conv(F.pad(x, (p // 2, p - p // 2)))


class PoolSame(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.max_pool = nn.MaxPool1d(k)

    def forward(self, x):
        p = max(0, self.k - 1)                     # SAME padding for stride-1 arithmetic, as published
        return self.max_pool(F.pad(x, (p // 2, p - p // 2)))


class Block(nn.Module):
    def __init__(self, cin, cout, k, stride, downsample, first):
        super().__init__()
        self.cin, self.cout, self.down, self.first = cin, cout, downsample, first
        s = stride if downsample else 1
        self.bn1, self.do1 = nn.BatchNorm1d(cin), nn.Dropout(0.5)
        self.conv1 = ConvSame(cin, cout, k, s)
        self.bn2, self.do2 = nn.BatchNorm1d(cout), nn.Dropout(0.5)
        self.conv2 = ConvSame(cout, cout, k, 1)
        self.max_pool = PoolSame(s)

    def forward(self, x):
        out = x if self.first else self.do1(F.relu(self.bn1(x)))
        out = self.conv2(self.do2(F.relu(self.bn2(self.conv1(out)))))
        idn = self.max_pool(x) if self.down else x
        if self.cout != self.cin:
            a = (self.cout - self.cin) // 2
            idn = F.pad(idn.transpose(-1, -2), (a, self.cout - self.cin - a)).transpose(-1, -2)
        return out + idn


class Encoder(nn.Module):
    def __init__(self, base, k, n_block, stride=2, kind="papagei", n_out=512):
        super().__init__()
        self.kind = kind
        self.first_block_conv = ConvSame(1, base, k, 1)
        self.first_block_bn = nn.BatchNorm1d(base)
        blocks, cout = [], base
        for i in range(n_block):
            cin = base if i == 0 else int(base * 2 ** ((i - 1) // 4))
            cout = cin * 2 if (i % 4 == 0 and i != 0) else cin
            blocks.append(Block(cin, cout, k, stride, downsample=(i % 2 == 1), first=(i == 0)))
        self.basicblock_list = nn.ModuleList(blocks)
        self.out_channels = cout
        if kind == "papagei":
            self.final_bn = nn.BatchNorm1d(cout)
            self.dense = nn.Linear(cout, n_out)
        self.dim = n_out if kind == "papagei" else cout

    def features(self, x):
        if self.kind == "pulseppg":                 # InstanceNorm1d without affine = per-sample z-score
            x = (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + 1e-5)
        out = F.relu(self.first_block_bn(self.first_block_conv(x)))
        for b in self.basicblock_list:
            out = b(out)
        if self.kind == "papagei":
            return self.dense(F.relu(self.final_bn(out)).mean(-1))
        return out.max(-1)[0]

    def forward(self, x):
        return self.features(x)


def papagei_s():
    return Encoder(base=32, k=3, n_block=18, kind="papagei")


def pulseppg():
    return Encoder(base=128, k=11, n_block=12, kind="pulseppg")


def load(model, path, key=None):
    """Load released weights by name; returns the list of encoder parameters that were NOT found."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if key is not None:
        sd = sd[key]
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    own = model.state_dict()
    hit = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    missing = [k for k in own if k not in hit]
    own.update(hit)
    model.load_state_dict(own)
    return missing


FS = {"papagei": 125, "pulseppg": 50}
