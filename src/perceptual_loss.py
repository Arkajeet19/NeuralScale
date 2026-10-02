"""
VGG perceptual loss: instead of comparing generated and target images
pixel-by-pixel (which is what L1/MSE loss does, and what tends to produce
safe, slightly blurry outputs), this compares them in VGG19's learned
feature space. Two images that activate similar high-level features
("looks like fur", "looks like an edge") are considered similar even if
their exact pixel values differ -- which is much closer to how humans
judge image similarity, and is the key ingredient that makes GAN-based SR
output look sharp and realistic rather than just accurate-on-average.

Uses relu5_4 features (deep layer -> high-level perceptual similarity),
matching the original SRGAN paper's choice.
"""
import torch
import torch.nn as nn
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        # relu5_4 is layer index 35 in torchvision's vgg19 .features sequential
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:36]).to(device).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # frozen -- we're only using VGG to compute a loss, not training it

        # VGG expects ImageNet-normalized input
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.criterion = nn.L1Loss()

    def normalize(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def forward(self, pred, target):
        pred_norm = self.normalize(pred)
        target_norm = self.normalize(target)
        pred_features = self.feature_extractor(pred_norm)
        with torch.no_grad():
            target_features = self.feature_extractor(target_norm)
        return self.criterion(pred_features, target_features)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = VGGPerceptualLoss(device)
    a = torch.rand(2, 3, 192, 192).to(device)
    b = torch.rand(2, 3, 192, 192).to(device)
    loss = loss_fn(a, b)
    print(f"Perceptual loss on random tensors: {loss.item():.4f}  (sanity check -- should be a positive number)")
