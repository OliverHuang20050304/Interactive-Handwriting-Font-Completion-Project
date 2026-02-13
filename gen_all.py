import os
import argparse
import torch
import sys
import numpy as np
import cv2
import time
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from torchvision import transforms

# 確保可以 import models
sys.path.append(os.getcwd())
try:
    from models.generator import Generator
except ImportError:
    print("❌ Error: Could not import 'models.generator'.")
    sys.exit(1)

# ================= 影像處理核心 =================
def tensor2im(var):
    var = var.cpu().detach().numpy()
    if var.ndim == 3:
        var = var.squeeze() 
    
    var = (var + 1) / 2
    var = np.clip(var, 0, 1)
    var = var * 255
    img_np = var.astype('uint8')
    
    # 自動補墨水 (保持剛剛成功的參數)
    kernel = np.ones((2, 2), np.uint8) 
    img_np = cv2.erode(img_np, kernel, iterations=1)
    
    img = Image.fromarray(img_np, mode='L')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    
    return img

def load_mac_font(size=110):
    font_candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc"
    ]
    for path in font_candidates:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size=size, index=0)
                return font
            except:
                continue
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

# ================= 主程式 =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", type=str, default="result/checkpoints/last.pth")
    parser.add_argument("--ref", type=str, default="png_data/target/train")
    parser.add_argument("--output_dir", type=str, default="output_images")
    args = parser.parse_args()

    # 1. 準備常用中文字表 (這裡列出最常用的 500 字測試，你可以隨時換成 3000 字)
    chars = "的一是在不了有和人這中大為上個國我以要他時來用們生到作地於出就分對成會可主發年動同工也能下過子說產種面而方後多定行學法所民得經十三之進著等部度家電力裡如水化高自二理起小物現實加量都兩體制機當使點從業本去把性好應開它合還因由其些然前外天政四日那社義事平形相全表間樣想向道命此位由實那"
    # 如果你有一個 txt 檔案包含所有字，可以用 open('chars.txt').read() 取代上面這行

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"🚀 Device: {device} | Output: {args.output_dir}")

    # 2. 載入模型
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
        print(f"❌ Load failed: {e}")
        return
    gen.eval()

    # 3. 鎖定風格 (這是關鍵！我們只抽一次風格，讓所有字看起來像同一套字體)
    ref_imgs_paths = [os.path.join(args.ref, f) for f in os.listdir(args.ref) if f.endswith('.png')]
    import random
    selected_refs = random.sample(ref_imgs_paths, min(3, len(ref_imgs_paths)))
    while len(selected_refs) < 3: selected_refs.append(selected_refs[0])
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])
    ])
    
    style_stack = []
    for p in selected_refs:
        img = Image.open(p).convert("L")
        style_stack.append(transform(img))
    style_tensor = torch.stack(style_stack).unsqueeze(0).to(device)
    
    print(f"🎨 Style locked using: {[os.path.basename(p) for p in selected_refs]}")

    # 4. 開始量產
    font = load_mac_font(size=110)
    print(f"🔥 Start generating {len(chars)} characters...")
    
    count = 0
    start_time = time.time()
    
    with torch.no_grad():
        for char in chars:
            # 略過特殊符號或空白
            if char.strip() == "": continue
            
            try:
                # 生成
                char_img = draw_char(char, font)
                source_tensor = transform(char_img).unsqueeze(0).unsqueeze(1).to(device)
                out = gen.gen_from_style_char(style_tensor, source_tensor)
                
                # 存檔 (以字元命名，例如 "我.png")
                final_img = tensor2im(out[0])
                final_img.save(os.path.join(args.output_dir, f"{char}.png"))
                
                count += 1
                if count % 50 == 0:
                    print(f"   ... generated {count} chars")
            except Exception as e:
                print(f"⚠️ Failed on {char}: {e}")

    print(f"✅ Done! Generated {count} images in {time.time()-start_time:.1f}s.")
    print(f"📁 Check the folder: {args.output_dir}")

if __name__ == "__main__":
    main()