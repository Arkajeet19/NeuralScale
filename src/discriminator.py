"""
SRGAN-style discriminator: a standard VGG-style CNN classifier that takes a
192x192 HR patch (real or generator-produced) and outputs a single score for
"does this look like a real high-res image or a generated one."

Architecture: alternating conv blocks that double channels while halving
spatial size (strided convs), ending in dense layers -> single logit.
This matches the discriminator design from the original SRGAN paper,
sized for 192x192 input patches (same HR patch size as training.py uses).
"""
import torch
import torch.nn as nn


def conv_block(in_ch, out_ch, stride, use_bn=True):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)]
    if use_bn:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers


class Discriminator(nn.Module):
    def __init__(self, patch_size=192):
        super().__init__()
        layers = []
        # first block: no BatchNorm (matches SRGAN paper -- avoids normalizing
        # away the raw intensity info the discriminator needs on its first look)
        layers += conv_block(3, 64, stride=1, use_bn=False)
        layers += conv_block(64, 64, stride=2, use_bn=True)

        layers += conv_block(64, 128, stride=1, use_bn=True)
        layers += conv_block(128, 128, stride=2, use_bn=True)

        layers += conv_block(128, 256, stride=1, use_bn=True)
        layers += conv_block(256, 256, stride=2, use_bn=True)

        layers += conv_block(256, 512, stride=1, use_bn=True)
        layers += conv_block(512, 512, stride=2, use_bn=True)

        self.features = nn.Sequential(*layers)

        # input spatially shrinks by 2^4=16 across the four stride-2 blocks
        reduced = patch_size // 16
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(reduced),  # robust to slightly different input sizes
            nn.Flatten(),
            nn.Linear(512 * reduced * reduced, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            # no sigmoid here -- using BCEWithLogitsLoss in training for numerical stability
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    model = Discriminator(patch_size=192)
    x = torch.randn(2, 3, 192, 192)
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}  (expect [2, 1])")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
