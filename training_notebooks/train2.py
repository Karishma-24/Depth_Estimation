# Config (edit before each run) 
# EDIT THESE ONLY 
RESUME_PATH  = "./models/best_depth_model_v8.pth"   # Leave empty on FIRST run.
                     # After a crash: set to your mid-V8 checkpoint path.
START_EPOCH  = 30    # Set to 0 on first run.
                     # After a crash: set to last completed epoch number.

# Copy your printed loss lists here ONLY when resuming after a crash:
HARDCODED_TRAIN_LOSSES = [0.1388, 0.0889, 0.0748, 0.0667, 0.0607, 0.0567, 0.0537, 0.0516, 0.0496, 0.0479, 0.0467, 0.0454, 0.0445, 0.0435, 0.0428, 0.0419, 0.0413, 0.0406, 0.0401, 0.0395, 0.0390, 0.0385, 0.0381, 0.0376, 0.0372, 0.0369, 0.0365, 0.0362, 0.0358, 0.0356]
HARDCODED_VAL_LOSSES   = [0.0963, 0.0767, 0.0679, 0.0625, 0.0593, 0.0566, 0.0548, 0.0548, 0.0524, 0.0514, 0.0498, 0.0492, 0.0492, 0.0488, 0.0479, 0.0472, 0.0472, 0.0467, 0.0464, 0.0459, 0.0460, 0.0456, 0.0456, 0.0449, 0.0446, 0.0447, 0.0445, 0.0446, 0.0438, 0.0444]

# ───────────────────────────────────────────────────────────


#Imports & Setup
import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim.lr_scheduler import ReduceLROnPlateau

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ───────────────────────────────────────────────────────────

#Hyperparameters
BASE_PATH    = "./dataset/train"
SAVE_PATH    = "./models/best_depth_model_v8.pth"
RESULTS_PATH = "./results/all_results_v8.pkl"

IMG_SIZE       = 256
BATCH_SIZE     = 16
NUM_EPOCHS     = 50
ENCODER_LR     = 1e-5
DECODER_LR     = 1e-4
WEIGHT_DECAY   = 1e-5
PATIENCE_SCHED = 3
PATIENCE_STOP  = 7
NUM_SAMPLES    = 50000
SEED           = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Training on {NUM_SAMPLES} samples | Batch: {BATCH_SIZE} | Epochs: {NUM_EPOCHS}")

# ───────────────────────────────────────────────────────────

#Dataset Class
class NYUDepthDataset(Dataset):
    def __init__(self, csv_path, base_path, transform=None):
        self.df        = pd.read_csv(csv_path, header=None, names=["image", "depth"])
        self.base_path = base_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path   = os.path.join(self.base_path, self.df.iloc[idx]["image"])
        depth_path = os.path.join(self.base_path, self.df.iloc[idx]["depth"])

        image = Image.open(img_path).convert("RGB")
        depth = Image.open(depth_path)

        if self.transform:
            image = self.transform(image)

        depth = np.array(depth, dtype=np.float32)
        depth = depth / depth.max()              # per-image normalisation
        depth = torch.tensor(depth).unsqueeze(0) # [1, H, W]
        depth = F.interpolate(
            depth.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear", align_corners=False
        ).squeeze(0)

        return image, depth

print("Dataset class defined.")

# ───────────────────────────────────────────────────────────
#DataLoaders
img_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_csv = os.path.join(BASE_PATH, "data/nyu2_train.csv")
test_csv  = os.path.join(BASE_PATH, "data/nyu2_test.csv")

full_train = NYUDepthDataset(train_csv, BASE_PATH, transform=img_transform)

# Sample NUM_SAMPLES from full training set
rng     = np.random.RandomState(SEED)
indices = rng.choice(len(full_train), size=NUM_SAMPLES, replace=False)
train_subset = torch.utils.data.Subset(full_train, indices)

# 90/10 train/val split
val_size   = int(0.1 * NUM_SAMPLES)
train_size = NUM_SAMPLES - val_size
train_data, val_data = torch.utils.data.random_split(
    train_subset, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

test_data = NYUDepthDataset(test_csv, BASE_PATH, transform=img_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

# ───────────────────────────────────────────────────────────
#SSIM Loss
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, pred, target):
        pad  = self.window_size // 2
        mu_p = F.avg_pool2d(pred,   self.window_size, stride=1, padding=pad)
        mu_t = F.avg_pool2d(target, self.window_size, stride=1, padding=pad)
        mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t

        sigma_p  = F.avg_pool2d(pred ** 2,     self.window_size, stride=1, padding=pad) - mu_p2
        sigma_t  = F.avg_pool2d(target ** 2,   self.window_size, stride=1, padding=pad) - mu_t2
        sigma_pt = F.avg_pool2d(pred * target, self.window_size, stride=1, padding=pad) - mu_pt

        ssim_map = ((2 * mu_pt + self.C1) * (2 * sigma_pt + self.C2)) / \
                   ((mu_p2 + mu_t2 + self.C1) * (sigma_p + sigma_t + self.C2))
        return 1 - ssim_map.mean()

print("SSIMLoss defined.")

# ───────────────────────────────────────────────────────────

#Combined Loss
class DepthLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssim = SSIMLoss()

    def gradient_loss(self, pred, target):
        pred_dx   = pred[:, :, :, 1:]   - pred[:, :, :, :-1]
        pred_dy   = pred[:, :, 1:, :]   - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        return (torch.abs(pred_dx - target_dx).mean() +
                torch.abs(pred_dy - target_dy).mean())

    def forward(self, pred, target):
        l1_loss   = torch.abs(pred - target).mean()
        ssim_loss = self.ssim(pred, target)
        grad_loss = self.gradient_loss(pred, target)
        return 0.85 * ssim_loss + 0.15 * l1_loss + 0.1 * grad_loss

print("DepthLoss defined.")

# ───────────────────────────────────────────────────────────

#Model Architecture
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNet50UNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        self.enc0 = nn.Sequential(base.conv1, base.bn1, base.relu)  # 64ch,  H/2
        self.pool = base.maxpool                                      # 64ch,  H/4
        self.enc1 = base.layer1   # 256ch,  H/4
        self.enc2 = base.layer2   # 512ch,  H/8
        self.enc3 = base.layer3   # 1024ch, H/16
        self.enc4 = base.layer4   # 2048ch, H/32

        self.dec4 = DecoderBlock(2048, 1024, 512)
        self.dec3 = DecoderBlock(512,  512,  256)
        self.dec2 = DecoderBlock(256,  256,  128)
        self.dec1 = DecoderBlock(128,  64,   64)

        self.final_upsample = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        s0 = self.enc0(x)               # [B, 64,   H/2,  W/2]
        s1 = self.enc1(self.pool(s0))   # [B, 256,  H/4,  W/4]
        s2 = self.enc2(s1)              # [B, 512,  H/8,  W/8]
        s3 = self.enc3(s2)              # [B, 1024, H/16, W/16]
        b  = self.enc4(s3)              # [B, 2048, H/32, W/32]

        x = self.dec4(b,  s3)           # [B, 512,  H/16, W/16]
        x = self.dec3(x,  s2)           # [B, 256,  H/8,  W/8]
        x = self.dec2(x,  s1)           # [B, 128,  H/4,  W/4]
        x = self.dec1(x,  s0)           # [B, 64,   H/2,  W/2]

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.final_upsample(x)      # [B, 32,   H,    W]
        return self.out_conv(x)         # [B, 1,    H,    W]

print("ResNet50UNet defined.")

# ───────────────────────────────────────────────────────────

#Initialize Model & Optimizer
model = ResNet50UNet().to(device)
print("Model initialized with ImageNet pretrained encoder, decoder from scratch.")

encoder_params = (list(model.enc0.parameters()) + list(model.enc1.parameters()) +
                  list(model.enc2.parameters()) + list(model.enc3.parameters()) +
                  list(model.enc4.parameters()))
decoder_params = (list(model.dec4.parameters()) + list(model.dec3.parameters()) +
                  list(model.dec2.parameters()) + list(model.dec1.parameters()) +
                  list(model.final_upsample.parameters()) +
                  list(model.out_conv.parameters()))

optimizer = torch.optim.Adam([
    {"params": encoder_params, "lr": ENCODER_LR},
    {"params": decoder_params, "lr": DECODER_LR},
], weight_decay=WEIGHT_DECAY)

scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=PATIENCE_SCHED,
                              factor=0.5)
criterion = DepthLoss()

# Only runs if resuming after a mid-V8 crash
if RESUME_PATH:
    print(f"Resuming from checkpoint: {RESUME_PATH}")
    model.load_state_dict(torch.load(RESUME_PATH, map_location=device))

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print(f"Encoder LR: {ENCODER_LR} | Decoder LR: {DECODER_LR}")


#Train & Validate Functions
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, depths in loader:
        imgs, depths = imgs.to(device), depths.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, depths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        del imgs, depths, preds, loss
        torch.cuda.empty_cache()
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, depths in loader:
            imgs, depths = imgs.to(device), depths.to(device)
            preds = model(imgs)
            loss  = criterion(preds, depths)
            total_loss += loss.item()
            del imgs, depths, preds, loss
            torch.cuda.empty_cache()
    return total_loss / len(loader)

print("Training functions defined.")

# ───────────────────────────────────────────────────────────

#Training Loop
train_losses = list(HARDCODED_TRAIN_LOSSES)
val_losses   = list(HARDCODED_VAL_LOSSES)

best_val_loss     = min(val_losses) if val_losses else float("inf")
epochs_no_improve = 0

print(f"Starting V8: ResNet50 U-Net | 256×256 | 50k samples | from scratch")
print(f"Resuming from epoch {START_EPOCH + 1}" if START_EPOCH > 0 else "Starting from epoch 1")
print("=" * 70)

for epoch in range(START_EPOCH + 1, NUM_EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss   = validate(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    scheduler.step(val_loss)
    gc.collect()

    saved = ""
    if val_loss < best_val_loss:
        best_val_loss     = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), SAVE_PATH)
        saved = " ✓ saved"
    else:
        epochs_no_improve += 1

    print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
          f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
          f"Best: {best_val_loss:.4f}{saved}")

    if epochs_no_improve >= PATIENCE_STOP:
        print(f"\nEarly stopping triggered at epoch {epoch}.")
        break

print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

# ───────────────────────────────────────────────────────────

#Save Results & Plot
results = {
    "train_losses":  train_losses,
    "val_losses":    val_losses,
    "best_val_loss": best_val_loss,
}
with open(RESULTS_PATH, "wb") as f:
    pickle.dump(results, f)
print("Results saved.")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss", linewidth=2)
plt.plot(val_losses,   label="Val Loss",   linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("V8: ResNet50 U-Net | 256×256 | 50k samples | From Scratch")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/kaggle/working/v8_training_curve.png", dpi=150)
plt.show()
print("Training curve saved.")


# ───────────────────────────────────────────────────────────
# Evaluation Metrics
# Load best checkpoint before evaluating
load_path = RESUME_PATH if RESUME_PATH else SAVE_PATH
model.load_state_dict(torch.load(load_path, map_location=device))
print(f"Loaded best checkpoint from: {load_path}")

model.eval()
all_preds, all_targets = [], []

with torch.no_grad():
    for imgs, depths in test_loader:
        imgs = imgs.to(device)
        preds = model(imgs)
        all_preds.append(preds.cpu())
        all_targets.append(depths.cpu())
        del imgs, preds
        torch.cuda.empty_cache()

preds   = torch.cat(all_preds,   dim=0).numpy()
targets = torch.cat(all_targets, dim=0).numpy()

thresh   = np.maximum(preds / (targets + 1e-6), targets / (preds + 1e-6))
delta1   = (thresh < 1.25     ).mean()
delta2   = (thresh < 1.25 ** 2).mean()
delta3   = (thresh < 1.25 ** 3).mean()
abs_rel  = np.mean(np.abs(preds - targets) / (targets + 1e-6))
sq_rel   = np.mean(((preds - targets) ** 2) / (targets + 1e-6))
rmse     = np.sqrt(np.mean((preds - targets) ** 2))
rmse_log = np.sqrt(np.mean((np.log(preds + 1e-6) - np.log(targets + 1e-6)) ** 2))

print("\n── V8 Test Metrics ──────────────────────────")
print(f"δ1      (↑): {delta1:.4f}")
print(f"δ2      (↑): {delta2:.4f}")
print(f"δ3      (↑): {delta3:.4f}")
print(f"AbsRel  (↓): {abs_rel:.4f}")
print(f"SqRel   (↓): {sq_rel:.4f}")
print(f"RMSE    (↓): {rmse:.4f}")
print(f"RMSElog (↓): {rmse_log:.4f}")

# ───────────────────────────────────────────────────────────

# Qualitative Predictions
model.eval()
mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])

fig, axes = plt.subplots(5, 3, figsize=(12, 18))
axes[0][0].set_title("RGB Input",     fontsize=13)
axes[0][1].set_title("Ground Truth",  fontsize=13)
axes[0][2].set_title("V8 Prediction", fontsize=13)

count = 0
with torch.no_grad():
    for imgs, depths in test_loader:
        preds = model(imgs.to(device)).cpu()
        for i in range(imgs.size(0)):
            if count >= 5:
                break
            img_np = imgs[i].permute(1, 2, 0).numpy()
            img_np = (img_np * std + mean).clip(0, 1)
            axes[count][0].imshow(img_np)
            axes[count][1].imshow(depths[i][0].numpy(), cmap="plasma")
            axes[count][2].imshow(preds[i][0].numpy(),  cmap="plasma")
            for j in range(3):
                axes[count][j].axis("off")
            count += 1
        if count >= 5:
            break

plt.suptitle("V8: ResNet50 U-Net | 50k samples", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig("/kaggle/working/v8_predictions.png", dpi=150)
plt.show()
print("Qualitative predictions saved.")

# ───────────────────────────────────────────────────────────