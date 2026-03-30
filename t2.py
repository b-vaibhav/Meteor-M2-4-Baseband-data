# ==========================================
# FILE: t1_final.py (Master Training Script)
# ==========================================
import os
import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Set your directory
os.chdir(r"C:\Users\vaibh\OneDrive\Desktop\SatImg")

# =========================================
# 1. BULLETPROOF PAIRED DATASET LOADER
# =========================================
class PairedSatelliteDataset(Dataset):
    def __init__(self, clean_dir, corrupt_dir):
        self.clean_dir = clean_dir
        self.corrupt_dir = corrupt_dir
        self.files = [f for f in os.listdir(clean_dir) if f.endswith(('.png','.jpg','.jpeg'))]
        self.transform = transforms.ToTensor()

    def __len__(self): 
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        name, ext = os.path.splitext(file_name)
        corrupt_name = f"{name}_corrupted{ext}" 

        clean_path = os.path.join(self.clean_dir, file_name)
        corrupt_path = os.path.join(self.corrupt_dir, corrupt_name)

        clean_img = cv2.cvtColor(cv2.imread(clean_path), cv2.COLOR_BGR2RGB)
        corrupt_img = cv2.cvtColor(cv2.imread(corrupt_path), cv2.COLOR_BGR2RGB)

        # Subtract images to find the exact glitches
        diff = np.abs(clean_img.astype(np.float32) - corrupt_img.astype(np.float32))
        raw_mask = (np.max(diff, axis=-1) > 15).astype(np.uint8)
        
        # Morphological Closing to fill "Swiss Cheese" holes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        solid_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
        mask_2d = solid_mask.astype(np.float32)

        # Joint Data Augmentation
        if np.random.rand() > 0.5:
            clean_img = cv2.flip(clean_img, 1)
            corrupt_img = cv2.flip(corrupt_img, 1)
            mask_2d = cv2.flip(mask_2d, 1)
            
        if np.random.rand() > 0.5:
            clean_img = cv2.flip(clean_img, 0)
            corrupt_img = cv2.flip(corrupt_img, 0)
            mask_2d = cv2.flip(mask_2d, 0)

        clean_tensor = self.transform(clean_img)
        corrupt_tensor = self.transform(corrupt_img)
        mask_tensor = torch.tensor(mask_2d.copy()).unsqueeze(0)

        # 4-channel input: Image (3) + Mask (1)
        input_tensor = torch.cat([corrupt_tensor, mask_tensor], dim=0)
        
        return input_tensor, clean_tensor

# =========================
# 2. MODEL ARCHITECTURE
# =========================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x): return self.conv(x)

class AdvancedUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(4, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.final = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d1 = self.up1(e3)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        return torch.sigmoid(self.final(d2))

# =========================
# 3. STATS & LOSS FUNCTIONS
# =========================
def calculate_r2(target, output):
    """Calculates R-squared variance captured by the model."""
    target_flat = target.view(-1)
    output_flat = output.view(-1)
    
    target_mean = torch.mean(target_flat)
    ss_tot = torch.sum((target_flat - target_mean) ** 2)
    ss_res = torch.sum((target_flat - output_flat) ** 2)
    
    r2 = 1 - (ss_res / (ss_tot + 1e-7))
    return r2.item()

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.enc_1 = nn.Sequential(*vgg16[:4])
        self.enc_2 = nn.Sequential(*vgg16[4:9])
        self.enc_3 = nn.Sequential(*vgg16[9:16])
        for param in self.parameters(): 
            param.requires_grad = False
            
    def forward(self, x):
        enc1 = self.enc_1(x)
        enc2 = self.enc_2(enc1)
        enc3 = self.enc_3(enc2)
        return [enc1, enc2, enc3]

def gram_matrix(y):
    # Force float32 to prevent NaN memory overflows
    y = y.to(torch.float32)
    b, c, h, w = y.size()
    features = y.view(b, c, w * h)
    features_t = features.transpose(1, 2)
    return features.bmm(features_t) / (c * h * w)

class InpaintingLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vgg = VGG16FeatureExtractor().to(device)
        self.l1 = nn.L1Loss()
        
    def forward(self, output, target, mask):
        mask_expanded = mask[:, 3:4, :, :] 
        
        # 5x priority to missing pixels
        pixel_loss = self.l1(output, target) + 5.0 * self.l1(output * mask_expanded, target * mask_expanded)
        
        out_features = self.vgg(output)
        target_features = self.vgg(target)
        perceptual_loss = 0
        style_loss = 0
        
        for out_feat, target_feat in zip(out_features, target_features):
            perceptual_loss += self.l1(out_feat, target_feat)
            style_loss += self.l1(gram_matrix(out_feat), gram_matrix(target_feat))
            
        tv_loss = torch.mean(torch.abs(output[:, :, :, :-1] - output[:, :, :, 1:])) + \
                  torch.mean(torch.abs(output[:, :, :-1, :] - output[:, :, 1:, :]))
                  
        return pixel_loss + (0.05 * perceptual_loss) + (120 * style_loss) + (0.1 * tv_loss)

# =========================
# 4. EXECUTION / TRAINING LOOP
# =========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Master Training on: {device}")

    # 🚀 NEW: Hardware Acceleration for RTX GPUs
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Set Paths
    TRAIN_CLEAN = r"dataset\train"
    TRAIN_CORRUPT = r"dataset\train_corrupted"
    VAL_CLEAN = r"dataset\val"
    VAL_CORRUPT = r"dataset\val_corrupted"
    CSV_FILE = "training_metrics.csv"

    # Initialize Engine
    model = AdvancedUNet().to(device)
    criterion = InpaintingLoss(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Modern PyTorch 2.x syntax for Scaler
    scaler = torch.amp.GradScaler('cuda')

    # 🚀 NEW: Tuned Data Loaders for 75% VRAM (approx. 6GB target)
    BATCH_SIZE = 16 
    
    train_dataset = PairedSatelliteDataset(TRAIN_CLEAN, TRAIN_CORRUPT)
    val_dataset = PairedSatelliteDataset(VAL_CLEAN, VAL_CORRUPT)
    
    # Increased num_workers to 4 to feed the GPU faster
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Dataset Loaded | Train pairs: {len(train_dataset)} | Val pairs: {len(val_dataset)}")

    epochs = 100
    best_val = float("inf")
    train_losses, val_losses, r2_scores = [], [], []

    # Initialize CSV File with Headers
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Val Loss", "Val R-Squared", "Learning Rate"])

    for epoch in range(epochs):
        # ---------- TRAINING ----------
        model.train()
        train_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            # 1. Fast 16-bit prediction
            with torch.amp.autocast('cuda'):
                out = model(x)
                
            # 2. STEP OUT of autocast and force 32-bit for the heavy math!
            loss = criterion(out.float(), y.float(), x.float())
                
            scaler.scale(loss).backward()
            
            # SAFETY NET: Gradient Clipping to stop NaNs
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()

        # ---------- VALIDATION ----------
        model.eval()
        val_loss = 0
        val_r2_total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                
                # 1. Fast 16-bit prediction
                with torch.amp.autocast('cuda'):
                    out = model(x)
                
                # 2. Safe 32-bit evaluation
                val_loss += criterion(out.float(), y.float(), x.float()).item()
                val_r2_total += calculate_r2(y.float(), out.float())

        # Calculate Epoch Averages
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        avg_r2 = val_r2_total / len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        r2_scores.append(avg_r2)
        
        # Step Scheduler & Get LR
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val R²: {avg_r2:.4f} | LR: {current_lr}")

        # Save Best Model
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_model_FINAL.pth")
            print(f"   🌟 New best model saved! (R²: {avg_r2:.4f})")
            
        # Log to CSV immediately so data is safe if it crashes
        with open(CSV_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, train_loss, val_loss, avg_r2, current_lr])
            
    # ==================================
    # 5. FINAL VISUALIZATION & SAVING
    # ==================================
    plt.figure(figsize=(12,5))
    
    # Plot 1: Losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", color='blue')
    plt.plot(val_losses, label="Validation Loss", color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Loss Score")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    
    # Plot 2: R-Squared
    plt.subplot(1, 2, 2)
    plt.plot(r2_scores, label="Val R² Score", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("R-Squared (1.0 is Perfect)")
    plt.title("Model Variance Capture (R²)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("final_metrics_graphs.png")
    print("✅ Training complete! Data saved to training_metrics.csv and graphs saved to final_metrics_graphs.png.")