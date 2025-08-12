import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- Bloques utilitarios ---------
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=(1,3), stride=(1,1), padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = (ksize[0]//2, ksize[1]//2)
        self.conv = nn.Conv2d(in_ch, out_ch, ksize, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.GELU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SEBlock(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        inner = max(8, channels // r)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Conv2d(channels, inner, 1, bias=True), nn.GELU(),
            nn.Conv2d(inner, channels, 1, bias=True), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w

class Bottleneck2D(nn.Module):
    """
    Bloque residual: 1x1 -> (3 x k_t) dilatado -> 1x1 + SE.
    Downsample solo en el eje temporal con stride_t.
    """
    def __init__(self, in_ch, out_ch, k_t=7, dilation_t=1, stride_t=1, drop=0.0):
        super().__init__()
        mid = out_ch // 4
        self.conv1 = ConvBNAct(in_ch, mid, ksize=(1,1))
        pad_t = dilation_t * (k_t // 2)
        self.conv2 = ConvBNAct(mid, mid, ksize=(3, k_t),
                               stride=(1, stride_t),
                               padding=(1, pad_t))
        # aplicar dilatación temporal en conv2
        self.conv2.conv.dilation = (1, dilation_t)

        self.conv3 = nn.Conv2d(mid, out_ch, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_ch)
        self.se    = SEBlock(out_ch)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(drop) if drop > 0 else nn.Identity()

        self.shortcut = None
        if in_ch != out_ch or stride_t != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(1, stride_t), bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        out = self.act(out + identity)
        out = self.drop(out)
        return out

# --------- Modelo principal ---------
class ECGConv2D(nn.Module):
    """
    Entrada esperada: (B, 12, T) => se convierte a (B,1,12,T)
    Baja resolución solo en tiempo: T/2 por etapa con stride_t=2.
    """
    def __init__(self, n_classes=2, drop=0.1):
        super().__init__()
        # Stem
        self.stem = ConvBNAct(1, 32, ksize=(3, 15), stride=(1, 2), padding=(1, 7))  # T: /2

        # Stages
        self.stage1 = nn.Sequential(
            Bottleneck2D(32,  64,  k_t=15, dilation_t=1, stride_t=1, drop=drop),
            Bottleneck2D(64,  64,  k_t=7,  dilation_t=2, stride_t=1, drop=drop),
        )
        self.stage2 = nn.Sequential(
            Bottleneck2D(64,  96,  k_t=11, dilation_t=1, stride_t=2, drop=drop),   # T: /2
            Bottleneck2D(96,  96,  k_t=7,  dilation_t=2, stride_t=1, drop=drop),
        )
        self.stage3 = nn.Sequential(
            Bottleneck2D(96,  144, k_t=9,  dilation_t=1, stride_t=2, drop=drop),   # T: /2
            Bottleneck2D(144, 144, k_t=7,  dilation_t=3, stride_t=1, drop=drop),
        )
        self.stage4 = nn.Sequential(
            Bottleneck2D(144, 192, k_t=7,  dilation_t=1, stride_t=2, drop=drop),   # T: /2
            Bottleneck2D(192, 192, k_t=5,  dilation_t=3, stride_t=1, drop=drop),
        )

        self.global_avg = nn.AdaptiveAvgPool2d(1)
        self.global_max = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 2, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 12, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B,1,12,T)
        x = self.stem(x)       # (B,32,12,T/2)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        avg = self.global_avg(x)
        mx  = self.global_max(x)
        z = torch.cat([avg, mx], dim=1)  # (B, 384, 1, 1)
        return self.head(z)              # (B, n_classes)