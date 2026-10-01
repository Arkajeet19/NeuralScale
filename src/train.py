"""
Trains the EDSR-baseline model on DIV2K.

Run from the project root:
    py src/train.py

Loss: L1 (mean absolute error), not MSE. This is a well-known SR trick —
MSE penalizes large errors heavily and pushes the model toward blurry,
"safe" averaged predictions. L1 produces sharper results and is what
EDSR/most modern SR papers use.

Metric: PSNR (Peak Signal-to-Noise Ratio) on the validation set — the
standard quantitative metric for SR quality. We'll add SSIM in the
evaluation script (Phase 3) for a fuller picture.
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

# ---- config ----
DATA_ROOT = r"G:\mini dlss\data\raw"
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "logs")

SCALE = 4
LR_PATCH_SIZE = 48
NUM_BLOCKS = 16
CHANNELS = 64

BATCH_SIZE = 16
REPEAT = 16              # patches sampled per image per epoch (was implicitly 1 -- too few iterations/epoch)
NUM_EPOCHS = 60           # fewer epochs needed now since each one has 16x more iterations
LEARNING_RATE = 1e-4
NUM_WORKERS = 0          # 0 keeps the in-memory image cache in a single process (see dataset.py).
                          # Multiple worker processes would each hold a separate, duplicate cache.
CHECKPOINT_EVERY = 5     # epochs


def psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # ---- data ----
    train_ds = DIV2KPatchDataset(
        hr_dir=os.path.join(DATA_ROOT, "DIV2K_train_HR"),
        lr_dir=os.path.join(DATA_ROOT, "DIV2K_train_LR_bicubic", "X4"),
        scale=SCALE, lr_patch_size=LR_PATCH_SIZE, train=True, repeat=REPEAT,
    )
    val_ds = DIV2KPatchDataset(
        hr_dir=os.path.join(DATA_ROOT, "DIV2K_valid_HR"),
        lr_dir=os.path.join(DATA_ROOT, "DIV2K_valid_LR_bicubic", "X4"),
        scale=SCALE, lr_patch_size=LR_PATCH_SIZE, train=False,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Train patches/epoch: {len(train_ds)}  |  Val images: {len(val_ds)}")

    # ---- model, loss, optimizer ----
    model = EDSRBaseline(scale=SCALE, num_blocks=NUM_BLOCKS, channels=CHANNELS).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    writer = SummaryWriter(LOG_DIR)
    best_psnr = 0.0
    start_epoch = 0

    # resume from checkpoint if one exists
    ckpt_path = os.path.join(CHECKPOINT_DIR, "latest.pth")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_psnr", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    # ---- training loop ----
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for i, (lr_img, hr_img) in enumerate(train_loader):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)

            optimizer.zero_grad()
            pred = model(lr_img)
            loss = criterion(pred, hr_img)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if i % 20 == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}  Batch {i}/{len(train_loader)}  "
                      f"Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()

        # ---- validation ----
        model.eval()
        val_psnr = 0.0
        with torch.no_grad():
            for lr_img, hr_img in val_loader:
                lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                pred = model(lr_img).clamp(0, 1)
                val_psnr += psnr(pred, hr_img).item()
        val_psnr /= len(val_loader)

        elapsed = time.time() - t0
        print(f"[Epoch {epoch+1}] avg_loss={avg_loss:.4f}  val_psnr={val_psnr:.2f}dB  "
              f"time={elapsed:.1f}s")

        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("PSNR/val", val_psnr, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # ---- checkpointing ----
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_psnr": val_psnr,
            "best_psnr": best_psnr,
        }
        torch.save(state, ckpt_path)

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            state["best_psnr"] = best_psnr
            torch.save(state, os.path.join(CHECKPOINT_DIR, "best.pth"))
            print(f"  -> new best model saved (PSNR {best_psnr:.2f}dB)")

        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            torch.save(state, os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pth"))

    writer.close()
    print("Training complete.")

if __name__ == "__main__":
    main()