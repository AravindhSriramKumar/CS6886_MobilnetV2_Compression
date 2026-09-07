"""
CIFAR-adapted MobileNet-v2 for CS6886 Assignment 2 (Q1b).

Reimplements the MobileNet-v2 inverted-residual architecture (see
https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py)
directly rather than importing torchvision's model class, so individual conv/
BN layers stay directly accessible for the Q2 compression pass.

CIFAR adaptation: the stock ImageNet stride schedule downsamples a 224x224
input by 32x, which would collapse a 32x32 CIFAR-10 image to below 1x1
before the classifier. Following the standard CIFAR MobileNet-v2 convention,
the stem stride and the first expansion stage's stride are both reduced from
2 to 1, so the network downsamples by 8x instead of 32x (32x32 -> 4x4 feature
map before global average pooling).
"""

import torch
import torch.nn as nn

# (expand_ratio t, out_channels c, num_blocks n, stride s)
# Stride of the (t=6, c=24) stage changed 2 -> 1 for CIFAR-10's 32x32 input.
CIFAR_INVERTED_RESIDUAL_SETTING = [
    (1, 16, 1, 1),
    (6, 24, 2, 1),   # CIFAR: stride 2 -> 1
    (6, 32, 3, 2),
    (6, 64, 4, 2),
    (6, 96, 3, 1),
    (6, 160, 3, 2),
    (6, 320, 1, 1),
]

def _make_divisible(v, divisor=8, min_value=None):
    """Round channel counts to the nearest multiple of `divisor` (>=90% of v)."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        assert stride in (1, 2)
        hidden_dim = int(round(in_ch * expand_ratio))
        self.use_res_connect = stride == 1 and in_ch == out_ch

        layers = []
        if expand_ratio != 1:
            # pointwise expansion
            layers.append(ConvBNReLU(in_ch, hidden_dim, kernel_size=1))
        layers += [
            # depthwise
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pointwise-linear projection (no activation)
            nn.Conv2d(hidden_dim, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2CIFAR(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0, dropout=0.2,
                 inverted_residual_setting=None):
        super().__init__()
        if inverted_residual_setting is None:
            inverted_residual_setting = CIFAR_INVERTED_RESIDUAL_SETTING

        input_channel = _make_divisible(32 * width_mult)
        last_channel = _make_divisible(1280 * max(1.0, width_mult))

        # CIFAR stem: stride 1 (ImageNet default is stride 2).
        features = [ConvBNReLU(3, input_channel, stride=1)]

        for t, c, n, s in inverted_residual_setting:
            out_channel = _make_divisible(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, out_channel,
                                                  stride, expand_ratio=t))
                input_channel = out_channel

        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_channel, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_model(num_classes=10, width_mult=1.0, dropout=0.2):
    return MobileNetV2CIFAR(num_classes=num_classes, width_mult=width_mult,
                            dropout=dropout)


if __name__ == "__main__":
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(model)
    print(f"params: {n_params:,}")
    print(f"output shape: {tuple(y.shape)}")
