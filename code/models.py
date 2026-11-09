"""Models. ResNet34-1D is copied from the original authors' repository (resnet.py)
so results are directly comparable; TinyCNN is sized for an ESP32-S3."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv1d(in_planes, planes, 1, stride, bias=False),
                                          nn.BatchNorm1d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class Head(nn.Module):
    """MLP head; optionally late-fuses demographic covariates (age, sex, BMI)."""

    def __init__(self, d_in, n_demo=0, hidden=(256, 128), p=0.5):
        super().__init__()
        layers, d = [], d_in + n_demo
        for h in hidden:
            layers += [nn.Dropout(p), nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, z, demo=None):
        if demo is not None:
            z = torch.cat([z, demo], 1)
        return self.mlp(z)


class ResNet1D(nn.Module):
    def __init__(self, num_blocks=(3, 4, 6, 3), n_demo=0):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv1d(1, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layers = nn.Sequential(*[self._make(p, n, s) for p, n, s in
                                      zip((64, 128, 256, 512), num_blocks, (1, 2, 2, 2))])
        self.head = Head(512, n_demo)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make(self, planes, n, stride):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, demo=None):
        z = F.relu(self.bn1(self.conv1(x)))
        z = self.layers(z)
        z = F.adaptive_avg_pool1d(z, 1).flatten(1)
        return self.head(z, demo)


class TinyCNN(nn.Module):
    """~20k parameters; int8 size ~25 kB -> fits ESP32 / ESP32-S3 SRAM easily."""

    def __init__(self, n_demo=0, ch=(16, 24, 32, 48)):
        super().__init__()
        layers, c_in = [], 1
        for c, k in zip(ch, (7, 5, 5, 3)):
            layers += [nn.Conv1d(c_in, c, k, 2, k // 2, bias=False), nn.BatchNorm1d(c), nn.ReLU()]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = Head(c_in, n_demo, hidden=(32,), p=0.2)

    def forward(self, x, demo=None):
        z = self.features(x)
        z = F.adaptive_avg_pool1d(z, 1).flatten(1)
        return self.head(z, demo)


def build(name, n_demo=0):
    if name == "resnet34":
        return ResNet1D((3, 4, 6, 3), n_demo)
    if name == "tiny":
        return TinyCNN(n_demo)
    raise ValueError(name)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
