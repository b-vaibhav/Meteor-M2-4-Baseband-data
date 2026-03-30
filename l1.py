# ==========================================
# FILE: sequential_pipeline.py
# Program 1 (Horizontal) -> Program 2 (Dual-Res/Color Math)
# ==========================================
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt

os.chdir(r"C:\Users\vaibh\OneDrive\Desktop\SatImg")

# =========================
# 1. SHARED MODEL ARCHITECTURE
# =========================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.InstanceNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.InstanceNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x): return self.conv(x)

class AdvancedUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1, self.enc2, self.enc3 = ConvBlock(4, 64), ConvBlock(64, 128), ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.up1, self.dec1 = nn.ConvTranspose2d(256, 128, 2, stride=2), ConvBlock(256, 128)
        self.up2, self.dec2 = nn.ConvTranspose2d(128, 64, 2, stride=2), ConvBlock(128, 64)
        self.final = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d1 = self.dec1(torch.cat([self.up1(e3), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e1], dim=1))
        return torch.sigmoid(self.final(d2))

# =========================
# 2. PROGRAM 1 LOGIC (Horizontal Scan)
# =========================
def p1_generate_mask(img, pass_index):
    h, w, _ = img.shape
    black_mask = (np.all(img < 5, axis=-1).astype(np.uint8) * 255) if pass_index == 0 else np.zeros((h, w), dtype=np.uint8)
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    directional_blur = cv2.blur(gray, (5 + (pass_index * 10), 1))
    abs_grad_x = np.abs(cv2.Sobel(directional_blur, cv2.CV_32F, 1, 0, ksize=3))
    
    gray_thresh = max(30, 50 - (pass_index * 5))
    potential_gray_lines = cv2.bitwise_and((abs_grad_x < 2).astype(np.uint8), (gray > gray_thresh).astype(np.uint8)) * 255
    true_gray_lines = cv2.morphologyEx(potential_gray_lines, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1)))
    
    final_mask = cv2.dilate(cv2.bitwise_or(black_mask, true_gray_lines), cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2 + (pass_index // 2))
    return np.expand_dims(final_mask, axis=2).astype(np.float32) / 255.0

# =========================
# 3. PROGRAM 2 LOGIC (Dual-Res & Color Math)
# =========================
def match_colors(pred_rgb, real_rgb, mask_2d):
    missing, healthy = mask_2d > 0.1, mask_2d <= 0.1
    if not np.any(healthy) or not np.any(missing): return pred_rgb
    result = pred_rgb.copy()
    for i in range(3):
        real_ch, pred_ch = real_rgb[:, :, i], pred_rgb[:, :, i]
        shifted = (pred_ch[missing] - np.mean(pred_ch[missing])) * ((np.std(real_ch[healthy]) + 1e-6) / (np.std(pred_ch[missing]) + 1e-6)) + np.mean(real_ch[healthy])
        result[:, :, i][missing] = shifted
    return np.clip(result, 0.0, 1.0)

def p2_generate_mask(img_bgr, p_size=128, stride=64):
    h, w, _ = img_bgr.shape
    img_padded = np.pad(img_bgr, ((0, (p_size - (h % stride)) % p_size), (0, (p_size - (w % stride)) % p_size), (0, 0)), mode='reflect')
    mask_padded = np.zeros((img_padded.shape[0], img_padded.shape[1]), dtype=np.uint8)
    
    v_win = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    for y in range(0, img_padded.shape[0] - p_size + 1, stride):
        for x in range(0, img_padded.shape[1] - p_size + 1, stride):
            patch = img_padded[y:y+p_size, x:x+p_size]
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            gray_anomalies = cv2.add(cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, v_win), cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, v_win))
            _, land_mask = cv2.threshold(cv2.morphologyEx(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 1], cv2.MORPH_BLACKHAT, v_win), 20, 255, cv2.THRESH_BINARY)
            _, anomaly_mask = cv2.threshold(cv2.bitwise_or(gray_anomalies, land_mask), 15, 255, cv2.THRESH_BINARY)
            clean_lines = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1)))
            mask_padded[y:y+p_size, x:x+p_size] = cv2.bitwise_or(mask_padded[y:y+p_size, x:x+p_size], cv2.dilate(clean_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), iterations=1))
            
    return np.expand_dims(mask_padded[:h, :w], axis=2).astype(np.float32) / 255.0

# =========================
# 4. UNIFIED EXECUTION
# =========================
def run_sequential_restoration(image_path, model_weights="best_model_FINAL.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Loading AI to {device}...")
    
    model = AdvancedUNet().to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device))
    model.eval()
    
    raw_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if raw_img is None: print(f"❌ Error loading {image_path}"); return
    
    has_alpha = len(raw_img.shape) == 3 and raw_img.shape[2] == 4
    alpha_channel = raw_img[:, :, 3] if has_alpha else None
    original_rgb = cv2.cvtColor(raw_img[:, :, :3] if has_alpha else raw_img, cv2.COLOR_BGR2RGB)
    
    img = original_rgb.copy()
    transform = transforms.ToTensor()
    patch_size, overlap = 256, 0.5
    stride = int(patch_size * (1 - overlap))
    window_2d = np.outer(np.hanning(patch_size), np.hanning(patch_size)).reshape(patch_size, patch_size, 1).astype(np.float32)

    # -----------------------------------------------------
    # EXECUTE PROGRAM 1 (1 Pass default from __main__)
    # -----------------------------------------------------
    print("\n[STAGE 1/2] Running Program 1: Adaptive Multi-Pass Engine...")
    for p in range(1): 
        if p % 2 != 0: img = cv2.rotate(img, cv2.ROTATE_180)
        h_orig, w_orig, _ = img.shape
        mask_3d = p1_generate_mask(img, pass_index=p)
        
        pad_h, pad_w = (patch_size - (h_orig % stride)) % patch_size, (patch_size - (w_orig % stride)) % patch_size
        img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        mask_padded = np.pad(mask_3d, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        
        final_image = np.zeros_like(img_padded, dtype=np.float32)
        weight_map = np.zeros((img_padded.shape[0], img_padded.shape[1], 1), dtype=np.float32)
        
        with torch.no_grad():
            for y in range(0, img_padded.shape[0] - patch_size + 1, stride):
                for x in range(0, img_padded.shape[1] - patch_size + 1, stride):
                    patch, mask_patch = img_padded[y:y+patch_size, x:x+patch_size], mask_padded[y:y+patch_size, x:x+patch_size]
                    in_tensor = torch.cat([transform(patch), torch.tensor(mask_patch).permute(2, 0, 1)], dim=0).unsqueeze(0).to(device)
                    with torch.amp.autocast('cuda'):
                        pred = model(in_tensor)[0].cpu().float().permute(1, 2, 0).numpy()
                    final_image[y:y+patch_size, x:x+patch_size] += pred * window_2d
                    weight_map[y:y+patch_size, x:x+patch_size] += window_2d

        feathered_mask = cv2.GaussianBlur(mask_3d, (7, 7), 0).reshape(h_orig, w_orig, 1)
        stage_out = (final_image / (weight_map + 1e-7) * 255).clip(0, 255).astype(np.float32)[:h_orig, :w_orig]
        img = (img.astype(np.float32) * (1 - feathered_mask) + stage_out * feathered_mask).clip(0, 255).astype(np.uint8)
        if p % 2 != 0: img = cv2.rotate(img, cv2.ROTATE_180)

    # -----------------------------------------------------
    # EXECUTE PROGRAM 2 (2 Passes default from __main__)
    # -----------------------------------------------------
    print("\n[STAGE 2/2] Running Program 2: Dual-Resolution Engine & Color Math...")
    for p in range(2):
        if p % 2 != 0: img = cv2.rotate(img, cv2.ROTATE_180)
        h_orig, w_orig, _ = img.shape
        mask_3d = p2_generate_mask(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        pad_h, pad_w = (patch_size - (h_orig % stride)) % patch_size, (patch_size - (w_orig % stride)) % patch_size
        img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        mask_padded = np.pad(mask_3d, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        
        final_image = np.zeros_like(img_padded, dtype=np.float32)
        weight_map = np.zeros((img_padded.shape[0], img_padded.shape[1], 1), dtype=np.float32)
        
        with torch.no_grad():
            for y in range(0, img_padded.shape[0] - patch_size + 1, stride):
                for x in range(0, img_padded.shape[1] - patch_size + 1, stride):
                    patch, mask_patch = img_padded[y:y+patch_size, x:x+patch_size].copy(), mask_padded[y:y+patch_size, x:x+patch_size].copy()
                    in_tensor = torch.cat([transform(patch), torch.tensor(mask_patch).permute(2, 0, 1)], dim=0).unsqueeze(0).to(device)
                    with torch.amp.autocast('cuda'):
                        pred = model(in_tensor)[0].cpu().float().permute(1, 2, 0).numpy()
                        
                    # Color Math and Mask Eraser applied here
                    pred = match_colors(pred, patch.astype(np.float32) / 255.0, mask_patch.squeeze(-1))
                    final_image[y:y+patch_size, x:x+patch_size] += pred * window_2d
                    weight_map[y:y+patch_size, x:x+patch_size] += window_2d
                    
                    update_area = (window_2d.squeeze(-1) > 0.6) & (mask_patch.squeeze(-1) > 0.1)
                    img_padded[y:y+patch_size, x:x+patch_size][update_area] = (pred * 255).astype(np.uint8)[update_area]
                    mask_padded[y:y+patch_size, x:x+patch_size][update_area] = 0.0

        feathered_mask = cv2.GaussianBlur(mask_3d, (7, 7), 0).reshape(h_orig, w_orig, 1)
        stage_out = (final_image / (weight_map + 1e-7) * 255).clip(0, 255).astype(np.float32)[:h_orig, :w_orig]
        img = (img.astype(np.float32) * (1 - feathered_mask) + stage_out * feathered_mask).clip(0, 255).astype(np.uint8)
        if p % 2 != 0: img = cv2.rotate(img, cv2.ROTATE_180)

    # -----------------------------------------------------
    # FINAL OUTPUT & PLOTTING
    # -----------------------------------------------------
    print("\n📦 Saving Lossless PNG...")
    out_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    final_save_image = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2BGRA) if has_alpha else out_bgr
    if has_alpha: final_save_image[:, :, 3] = alpha_channel

    out_name = f"FINAL_SEQUENTIAL_{os.path.basename(image_path).replace('.jpg', '.png')}"
    cv2.imwrite(out_name, final_save_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"✅ Pipeline complete! Saved as {out_name}")

    # Plot
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1), plt.title("Original Corrupted"), plt.imshow(original_rgb), plt.axis("off")
    plt.subplot(1, 2, 2), plt.title("Restored (Prog 1 -> Prog 2)"), plt.imshow(img), plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    TARGET_IMAGE = r"FINAL_AI_RESTORED_msu_mr_rgb_MCIR_corrected.png"
    if os.path.exists(TARGET_IMAGE):
        run_sequential_restoration(TARGET_IMAGE)
    else:
        print(f"⚠️ File not found: {TARGET_IMAGE}")