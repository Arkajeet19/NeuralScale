"""
EDSR-baseline: a compact residual CNN for super-resolution.

Why this architecture:
- Residual blocks let the network learn the *difference* between LR and HR
  detail, rather than reconstructing the whole image from scratch — much
  easier to train and converges faster.
- No BatchNorm: BatchNorm normalizes activation statistics per-batch, which
  actually removes the pixel-intensity information super-resolution depends
  on. The original EDSR paper found removing it improved quality AND cut
  memory usage by ~40%, which also happens to help a lot on 8GB VRAM.
- Sub-pixel convolution (PixelShuffle) for the final upsampling: instead of
  a transposed conv (which causes checkerboard artifacts), we predict
  scale^2 times the channels at low resolution then rearrange pixels into
  the higher-resolution grid. This is exactly the trick real-time SR/DLSS-
  style systems use because it's cheap and artifact-free.

Sizing: 16 residual blocks x 64 channels comfortably fits an RTX 3050 8GB
at a 48px LR patch size (192px HR) with batch size 16.
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels, res_scale=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # res_scale: scales the residual before adding back. Without this,
        # deep residual SR networks are prone to training instability.
        self.res_scale = res_scale

    def forward(self, x):
        out = self.conv2(self.relu(self.conv1(x)))
        return x + out * self.res_scale


class UpsampleBlock(nn.Module):
    """Sub-pixel convolution upsampling. Handles scale=2 or scale=4
    (4x done as two successive 2x sub-pixel steps, standard practice)."""
    def __init__(self, channels, scale):
        super().__init__()
        layers = []
        if scale == 4:
            for _ in range(2):
                layers += [
                    nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),
                ]
        elif scale == 2:
            layers += [
                nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
            ]
        else:
            raise ValueError(f"Unsupported scale: {scale}")
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class EDSRBaseline(nn.Module):
    def __init__(self, scale=4, num_blocks=16, channels=64, res_scale=1.0):
        # res_scale=1.0 (i.e. no dampening) matches the actual EDSR-baseline
        # config from the paper for a 16-block network. res_scale=0.1 is only
        # used in their much deeper 32-block/256-channel model to prevent
        # instability -- using it here just throttled convergence for no benefit.
        super().__init__()
        self.scale = scale

        # shallow feature extraction
        self.head = nn.Conv2d(3, channels, kernel_size=3, padding=1)

        # deep feature extraction: stack of residual blocks
        self.body = nn.Sequential(
            *[ResidualBlock(channels, res_scale) for _ in range(num_blocks)]
        )
        self.body_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        # upsampling + reconstruction
        self.upsample = UpsampleBlock(channels, scale)
        self.tail = nn.Conv2d(channels, 3, kernel_size=3, padding=1)

    def forward(self, x):
        feat = self.head(x)
        res = self.body(feat)
        res = self.body_conv(res)
        feat = feat + res  # global residual connection (skip the whole body)
        out = self.upsample(feat)
        out = self.tail(out)
        return out


if __name__ == "__main__":
    # sanity check: verify shapes and count parameters
    model = EDSRBaseline(scale=4, num_blocks=16, channels=64)
    x = torch.randn(2, 3, 48, 48)  # batch of 2 LR patches
    y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}  (expect [2, 3, 192, 192])")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")