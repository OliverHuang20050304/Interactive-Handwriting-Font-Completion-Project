import os
import argparse
import torch
import sys
import numpy as np
import cv2  # 需要用到 opencv 來做膨脹
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from torchvision import transforms

# 確保可以 import models
sys.path.append(os.getcwd())
try:
    from models.generator import Generator
except ImportError:
    print("❌ Error: Could not import 'models.generator'.")
    sys.exit(1)

def tensor2im(var):
    # (C, H, W) -> (H, W)
    var = var.cpu().detach().numpy()
    if var.ndim == 3:
        var = var.squeeze() 
    
    # 反正規化
    var = (var + 1) / 2
    var = np.clip(var, 0, 1)
    var = var * 255
    
    # 轉成 uint8 numpy array 供 OpenCV 使用
    img_np = var.astype('uint8')
    
    # ----------------------------------------------------
    # 【關鍵修復】: 形態學膨脹 (Dilation) - 自動補墨水
    # ----------------------------------------------------
    # 定義核 (Kernel)：2x2 的矩陣，數值越大筆劃越粗
    kernel = np.ones((2, 2), np.uint8) 
    
    # 因為是黑底白字運算比較方便，我們先假設這時候是黑字白底(255)，所以要侵蝕(Erosion)黑色
    # 但 PIL 轉出來通常是白的比較亮。
    # 簡單來說：我們要把「黑色」的區域擴大。
    # 在 OpenCV 裡，dilate 是擴張「亮」的區域（白色）。
    # 如果我們的字是黑的（數值低），背景是白的（數值高），那我們要用 erosion (侵蝕白色 = 擴張黑色)
    
    # 這裡我們直接用 "Erosion" (腐蝕白色背景 -> 字變粗)
    img_np = cv2.erode(img_np, kernel, iterations=1)
    
    # 轉回 PIL
    img = Image.fromarray(img_np, mode='L')
    
    # 增強對比度
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0) # 稍微加強就好，不用太暴力
    
    return img

def load_mac_font(size=110):
    font_candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
    ]
    for path in font_candidates:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size=size, index=0)
                print(f"✅ Loaded font skeleton: {path}")
                return font
            except:
                continue
    print("⚠️ Warning: No decent Chinese font found. Using default.")
    return ImageFont.load_default()

def draw_char(ch, font, size=128):
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)
    try:
        bbox = font.getbbox(ch)
        if bbox:
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = (size - w) // 2 - bbox[0]
            y = (size - h) // 2 - bbox[1] - 8 
        else:
            x, y = 0, 0
    except:
        x, y = 0, 0
    draw.text((x, y), ch, font=font, fill=0)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", type=str, required=True, help="Path to generator checkpoint")
    parser.add_argument("--ref", type=str, required=True, help="Path to reference images folder")
    parser.add_argument("--text", type=str, default="天地玄黃", help="Text to generate")
    parser.add_argument("--output", type=str, default="inference_dilated.png", help="Output filename")
    args = parser.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"🚀 Using device: {device}")

    # 建立模型
    gen_config = {
        "C_in": 1, "C": 32, "C_out": 1,
        "style_enc": { "norm": "in", "activ": "relu", "pad_type": "zero", "skip_scale_var": False },
        "experts": { "n_experts": 6, "norm": "in", "activ": "relu" },
        "emb_num": 2,
        "dec": { "norm": "in", "activ": "relu", "pad_type": "zero" }
    }
    
    gen = Generator(**gen_config).to(device)
    try:
        ckpt = torch.load(args.gen, map_location=device, weights_only=False)
        state_dict = ckpt['generator'] if 'generator' in ckpt else ckpt
        gen.load_state_dict(state_dict)
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return
    gen.eval()

    # 準備參考圖片
    ref_imgs_paths = []
    for root, _, files in os.walk(args.ref):
        for file in files:
            if file.endswith('.png'):
                ref_imgs_paths.append(os.path.join(root, file))
    
    if not ref_imgs_paths:
        print(f"❌ No PNG images found in {args.ref}")
        return
    
    import random
    selected_refs = random.sample(ref_imgs_paths, min(3, len(ref_imgs_paths)))
    while len(selected_refs) < 3: selected_refs.append(selected_refs[0])

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    style_imgs_stack = []
    style_debug_imgs = [] 
    
    print("🎨 Style Reference Images:")
    for path in selected_refs:
        print(f"  - {path}")
        img = Image.open(path).convert("L")
        style_debug_imgs.append(img.resize((128, 128)))
        style_imgs_stack.append(transform(img))
    
    style_tensor = torch.stack(style_imgs_stack).unsqueeze(0).to(device) 

    font = load_mac_font(size=110)

    print(f"✍️ Generating: {args.text}")
    
    source_debug_imgs = [] 
    result_imgs = []       

    with torch.no_grad():
        for char in args.text:
            # Source
            char_img = draw_char(char, font)
            source_debug_imgs.append(char_img)
            
            # 推論
            source_tensor = transform(char_img).unsqueeze(0).unsqueeze(1).to(device)
            out = gen.gen_from_style_char(style_tensor, source_tensor)
            
            # 轉圖片 (內含自動補墨水)
            out_img = tensor2im(out[0]) 
            result_imgs.append(out_img)

    # 組合大圖
    total_w = 128 * len(source_debug_imgs)
    source_strip = Image.new("L", (total_w, 128))
    for i, img in enumerate(source_debug_imgs):
        source_strip.paste(img, (i * 128, 0))
        
    result_strip = Image.new("L", (total_w, 128))
    for i, img in enumerate(result_imgs):
        result_strip.paste(img, (i * 128, 0))

    final_h = 128 * 3
    final_w = max(total_w, 128 * 3) 
    
    final_img = Image.new("L", (final_w, final_h), 255) # 白底
    
    for i, img in enumerate(style_debug_imgs):
        final_img.paste(img, (i * 128, 0))
        
    final_img.paste(source_strip, (0, 128))
    final_img.paste(result_strip, (0, 256))

    final_img.save(args.output)
    print(f"✅ Saved Enhanced Result to: {args.output}")

if __name__ == "__main__":
    main()