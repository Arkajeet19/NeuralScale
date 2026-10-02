"""
SRGAN-style fine-tuning on top of the existing EDSR-baseline model.

Warm-starts the generator from models/best.pth (the plain L1-trained model)
rather than training from scratch -- this skips the unstable early phase
GAN training is notorious for, since the generator already produces
reasonable output before the discriminator/perceptual loss get involved.

Three loss terms combine to train the generator each step:
  - Pixel (L1) loss   : keeps output grounded in the actual target content
  - Perceptual loss    : VGG feature-space similarity -> sharper, more
                          realistic texture than pixel loss alone produces
  - Adversarial loss   : pushes output to fool the discriminator into
                          thinking it's a real high-res image

Expect val_psnr to look WORSE than the plain model despite images looking
sharper -- this is normal and well-documented for GAN-based SR (the
original SRGAN paper reports the same tradeoff). Judge this phase mostly
by eye by looking at outputs/eval_samples after running evaluate.py.

Run from the project root:
    py src/train_gan.py
"""
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(__file__))
from dataset import DIV2KPatchDataset
from model import EDSRBaseline
from discriminator import Discriminator
from perceptual_loss import VGGPerceptualLoss

# ---- config ----
# Fine-tuning on your collected game screenshots, not the original DIV2K set --
# that's the dataset prepare_screenshot_data.py builds. Point this back at
# r"G:\mini dlss\data\raw" with DIV2K_train_HR/DIV2K_train_LR_bicubic folder
# names below if you ever want to GAN-finetune on DIV2K instead.
DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "game_screenshots")
HR_FOLDER_NAME = "train_HR"
LR_FOLDER_NAME = os.path.join("train_LR_bicubic", "X4")
GENERATOR_CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "models", "best.pth")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "logs_gan")

SCALE = 4
LR_PATCH_SIZE = 48
NUM_BLOCKS = 16
CHANNELS = 64

BATCH_SIZE = 8            # lower than plain training (16) -- 3 networks in VRAM now
REPEAT = 24
NUM_EPOCHS = 30            # fine-tuning, not training from scratch -- needs far fewer epochs
GEN_LR = 1e-4
DISC_LR = 1e-4
NUM_WORKERS = 0
CHECKPOINT_EVERY = 5

# loss term weights -- perceptual/adversarial scaled down since their raw
# magnitudes are much larger than pixel L1 loss (standard SRGAN-paper scaling)
PIXEL_WEIGHT = 1.0
PERCEPTUAL_WEIGHT = 0.006
ADVERSARIAL_WEIGHT = 0.001


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # ---- data ----
    train_ds = DIV2KPatchDataset(
        hr_dir=os.path.join(DATA_ROOT, HR_FOLDER_NAME),
        lr_dir=os.path.join(DATA_ROOT, LR_FOLDER_NAME),
        scale=SCALE, lr_patch_size=LR_PATCH_SIZE, train=True, repeat=REPEAT,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    print(f"Train patches/epoch: {len(train_ds)}")

    # ---- models ----
    generator = EDSRBaseline(scale=SCALE, num_blocks=NUM_BLOCKS, channels=CHANNELS).to(device)
    if not os.path.exists(GENERATOR_CHECKPOINT):
        raise FileNotFoundError(
            f"No generator checkpoint at {GENERATOR_CHECKPOINT}. "
            f"Run train.py first to produce models/best.pth before GAN fine-tuning."
        )
    gen_ckpt = torch.load(GENERATOR_CHECKPOINT, map_location=device)
    generator.load_state_dict(gen_ckpt["model_state"])
    print(f"Warm-started generator from epoch {gen_ckpt['epoch']+1} "
          f"(val_psnr={gen_ckpt.get('val_psnr', 'n/a')})")

    discriminator = Discriminator(patch_size=LR_PATCH_SIZE * SCALE).to(device)

    perceptual_loss_fn = VGGPerceptualLoss(device).to(device)
    pixel_loss_fn = nn.L1Loss()
    adversarial_loss_fn = nn.BCEWithLogitsLoss()

    optim_g = torch.optim.Adam(generator.parameters(), lr=GEN_LR, betas=(0.9, 0.999))
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=DISC_LR, betas=(0.9, 0.999))

    writer = SummaryWriter(LOG_DIR)
    start_epoch = 0

    # resume GAN fine-tuning if a checkpoint for THIS script already exists
    gan_ckpt_path = os.path.join(CHECKPOINT_DIR, "gan_latest.pth")
    if os.path.exists(gan_ckpt_path):
        ckpt = torch.load(gan_ckpt_path, map_location=device)
        generator.load_state_dict(ckpt["generator_state"])
        discriminator.load_state_dict(ckpt["discriminator_state"])
        optim_g.load_state_dict(ckpt["optim_g_state"])
        optim_d.load_state_dict(ckpt["optim_d_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed GAN fine-tuning from epoch {start_epoch}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        generator.train()
        discriminator.train()
        t0 = time.time()

        running_g_loss = running_d_loss = running_pixel = running_percep = running_adv = 0.0

        for i, (lr_img, hr_img) in enumerate(train_loader):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            batch_size = lr_img.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # ---- train discriminator ----
            optim_d.zero_grad()
            with torch.no_grad():
                sr_img = generator(lr_img)
            d_real = discriminator(hr_img)
            d_fake = discriminator(sr_img.detach())
            d_loss = (adversarial_loss_fn(d_real, real_labels) +
                      adversarial_loss_fn(d_fake, fake_labels)) / 2
            d_loss.backward()
            optim_d.step()

            # ---- train generator ----
            optim_g.zero_grad()
            sr_img = generator(lr_img)
            pixel_loss = pixel_loss_fn(sr_img, hr_img)
            percep_loss = perceptual_loss_fn(sr_img, hr_img)
            g_fake_pred = discriminator(sr_img)
            adv_loss = adversarial_loss_fn(g_fake_pred, real_labels)  # generator WANTS discriminator to say "real"

            g_loss = (PIXEL_WEIGHT * pixel_loss +
                      PERCEPTUAL_WEIGHT * percep_loss +
                      ADVERSARIAL_WEIGHT * adv_loss)
            g_loss.backward()
            optim_g.step()

            running_g_loss += g_loss.item()
            running_d_loss += d_loss.item()
            running_pixel += pixel_loss.item()
            running_percep += percep_loss.item()
            running_adv += adv_loss.item()

            if i % 20 == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}  Batch {i}/{len(train_loader)}  "
                      f"G_loss={g_loss.item():.4f}  D_loss={d_loss.item():.4f}")

        n = len(train_loader)
        elapsed = time.time() - t0
        print(f"[Epoch {epoch+1}] G_loss={running_g_loss/n:.4f}  D_loss={running_d_loss/n:.4f}  "
              f"pixel={running_pixel/n:.4f}  perceptual={running_percep/n:.4f}  "
              f"adversarial={running_adv/n:.4f}  time={elapsed:.1f}s")

        writer.add_scalar("Loss/generator", running_g_loss / n, epoch)
        writer.add_scalar("Loss/discriminator", running_d_loss / n, epoch)
        writer.add_scalar("Loss/pixel", running_pixel / n, epoch)
        writer.add_scalar("Loss/perceptual", running_percep / n, epoch)
        writer.add_scalar("Loss/adversarial", running_adv / n, epoch)

        state = {
            "epoch": epoch,
            "generator_state": generator.state_dict(),
            "discriminator_state": discriminator.state_dict(),
            "optim_g_state": optim_g.state_dict(),
            "optim_d_state": optim_d.state_dict(),
        }
        torch.save(state, gan_ckpt_path)

        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            torch.save(state, os.path.join(CHECKPOINT_DIR, f"gan_epoch_{epoch+1}.pth"))
            # also save just the generator weights in the same format evaluate.py/app.py expect,
            # so you can point either at this checkpoint to compare against the plain model
            torch.save({"epoch": epoch, "model_state": generator.state_dict()},
                       os.path.join(CHECKPOINT_DIR, "gan_generator.pth"))

    # final save of generator-only weights
    torch.save({"epoch": NUM_EPOCHS - 1, "model_state": generator.state_dict()},
               os.path.join(CHECKPOINT_DIR, "gan_generator.pth"))
    writer.close()
    print("GAN fine-tuning complete. Generator-only weights saved to models/gan_generator.pth")


if __name__ == "__main__":
    main()
