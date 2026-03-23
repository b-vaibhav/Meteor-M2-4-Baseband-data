# ==========================================
# FILE: t1.py (Advanced Training Script)
# ==========================================
import os
import cv2
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

# =========================
# CORRECTED CORRUPTION FUNCTION
# =========================
def corrupt_image(img):
    corrupted = img.copy()
    h, w, _ = img.shape

    # 1. Add noise FIRST, so we don't pollute the pure black dropouts
    noise = np.random.normal(0, 15, img.shape) 
    corrupted = np.clip(corrupted + noise, 0, 255).astype(np.uint8)

    # 2. Create an explicit, perfect mask array
    mask = np.zeros((h, w), dtype=np.float32)

    # 3. Add Horizontal dropouts (and log them in the mask)
    for i in range(0, h, 10):
        if np.random.rand() > 0.4:
            thickness = np.random.randint(5, 15)
            corrupted[i:i+thickness, :] = 0
            mask[i:i+thickness, :] = 1.0  # Explicitly tell the AI this is missing!

    # 4. Add random missing pixels (and log them)
    random_mask = np.random.rand(h, w) < 0.02
    corrupted[random_mask] = 0
    mask[random_mask] = 1.0

    # Returns BOTH the image and the mask
    return corrupted, mask

# =========================
# CORRECTED DATASET
# =========================
class SatelliteDataset(Dataset):
    def __init__(self, folder):
        self.files = [f for f in os.listdir(folder) if f.endswith(('.png','.jpg','.jpeg'))]
        self.folder = folder
        self.transform = transforms.ToTensor()

    def __len__(self): 
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.folder, self.files[idx])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Ensure image is 256x256
        img = cv2.resize(img, (256, 256))

        # Get the corrupted image AND the perfect mask directly
        corrupted, mask_array = corrupt_image(img)

        clean = self.transform(img)
        corrupted = self.transform(corrupted)
        
        # Convert the numpy mask to a PyTorch tensor shape (1, H, W)
        mask = torch.tensor(mask_array).unsqueeze(0)

        # Concatenate into a 4-channel input (Image + Mask)
        input_tensor = torch.cat([corrupted, mask], dim=0)
        
        return input_tensor, clean

# =========================
# MODEL ARCHITECTURE
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
    def forward(self, x): 
        return self.conv(x)

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
# LOSS FUNCTIONS (VGG Perceptual)
# =========================
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
        # The mask is the 4th channel of the input (index 3)
        mask_expanded = mask[:, 3:4, :, :]
        
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
# EXECUTION / TRAINING LOOP
# =========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = AdvancedUNet().to(device)
    criterion = InpaintingLoss(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    # ⚠️ IMPORTANT: If you get a "CUDA Out of Memory" error, change batch_size from 4 to 2
    train_dataset = SatelliteDataset("dataset/train")
    val_dataset = SatelliteDataset("dataset/val")
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    print(f"Train samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

    epochs = 50
    best_val = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y, x)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss += criterion(out, y, x).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}: Train={train_loss:.4f}, Val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_model_v2.pth")
            print("✅ Best model saved as best_model_v2.pth!")
            
    # Final plot to show learning progress
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve_v2.png")
    print("Training complete! Loss curve saved as loss_curve_v2.png.")