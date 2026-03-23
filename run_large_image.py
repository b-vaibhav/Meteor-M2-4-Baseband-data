# ==========================================
# FILE: run_large_image.py
# ==========================================
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

os.chdir(r"C:\Users\vaibh\OneDrive\Desktop\SatImg")

# --- MODEL DEFINITION (Must match training script) ---
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

# --- INFERENCE ENGINE ---
def restore_large_image(image_path, model_weights="best_model_v2.pth", patch_size=256, overlap=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model to {device}...")
    
    model = AdvancedUNet().to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.eval()
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not read image at {image_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_orig, w_orig, _ = img.shape
    
    # 1. Sliding Window Padding
    stride = int(patch_size * (1 - overlap))
    pad_h = (patch_size - (h_orig % stride)) % patch_size
    pad_w = (patch_size - (w_orig % stride)) % patch_size
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    h_pad, w_pad, _ = img_padded.shape
    
    # 2. EXACT MASKING WITH DEEP DILATION
    global_mask_2d = np.all(img_padded == 0, axis=-1).astype(np.float32)
    
    # INCREASED: 5x5 kernel and 3 iterations to completely swallow edge static
    kernel = np.ones((5, 5), np.float32)
    global_mask_2d_dilated = cv2.dilate(global_mask_2d, kernel, iterations=3)
    global_mask_3d = np.expand_dims(global_mask_2d_dilated, axis=2)
    
    # Cast to uint8 so PyTorch scales the image correctly!
    img_padded_for_ai = (img_padded * (1 - global_mask_3d)).astype(np.uint8)
    
    final_image = np.zeros((h_pad, w_pad, 3), dtype=np.float32)
    weight_map = np.zeros((h_pad, w_pad, 1), dtype=np.float32)
    
    window_1d = np.hanning(patch_size)
    window_2d = np.outer(window_1d, window_1d).reshape(patch_size, patch_size, 1).astype(np.float32)
    
    transform = transforms.ToTensor()
    print("Starting sliding window restoration (this may take a minute)...")
    
    # 3. Process Patches
    with torch.no_grad():
        for y in range(0, h_pad - patch_size + 1, stride):
            for x in range(0, w_pad - patch_size + 1, stride):
                
                patch = img_padded_for_ai[y:y+patch_size, x:x+patch_size] 
                mask_patch = global_mask_3d[y:y+patch_size, x:x+patch_size]
                
                img_tensor = transform(patch)
                mask_tensor = torch.tensor(mask_patch).permute(2, 0, 1)
                input_tensor = torch.cat([img_tensor, mask_tensor], dim=0).unsqueeze(0).to(device)
                
                pred = model(input_tensor)[0].cpu().permute(1, 2, 0).numpy()
                
                final_image[y:y+patch_size, x:x+patch_size] += pred * window_2d
                weight_map[y:y+patch_size, x:x+patch_size] += window_2d

    # 4. Normalize AI prediction
    final_image = final_image / (weight_map + 1e-7) 
    final_image = (final_image * 255).clip(0, 255)
    
    # 5. SOFT COMPOSITING (Feathering)
    # Blur the mask to create a soft, seamless transition edge
    feathered_mask_2d = cv2.GaussianBlur(global_mask_2d_dilated, (7, 7), 0)
    feathered_mask_3d = np.expand_dims(feathered_mask_2d, axis=2)
    
    # Use the feathered mask to gently blend original pixels with AI pixels
    composited_image = img_padded * (1 - feathered_mask_3d) + final_image * feathered_mask_3d
    
    # Crop back to original size
    composited_image = composited_image[:h_orig, :w_orig].astype("uint8")
    
    output_filename = "FINAL_RESTORED_" + os.path.basename(image_path)
    cv2.imwrite(output_filename, cv2.cvtColor(composited_image, cv2.COLOR_RGB2BGR))
    print(f"✅ Restoration complete! Saved as {output_filename}")

# --- EXECUTION ---
if __name__ == "__main__":
    TARGET_LARGE_IMAGE = r"C:\Users\vaibh\OneDrive\Desktop\SatImg\msu_mr_rgb_MCIR_corrected.png" 
    if os.path.exists(TARGET_LARGE_IMAGE):
        restore_large_image(TARGET_LARGE_IMAGE)
    else:
        print(f"⚠️ File not found: {TARGET_LARGE_IMAGE}")