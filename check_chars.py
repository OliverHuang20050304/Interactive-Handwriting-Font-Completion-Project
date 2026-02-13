import json
import os

# 你的 100 個訓練字
TRAIN_CHARS = (
    "永國酬標露騰驚護讓鬱靈響體屬觀豫蘭囊彎聽"
    "鹽爨釁鑲關鍵魔欄權讀艷驟疊隱龜龍慶麗舉憲"
    "應懷懸戀戇懿攪攀曬更書會最望期本朵機次止"
    "正此武比畢毛氣求汗江決沈沐沛河治法泛泰"
    "洋洞津洪活洽派流海浩浪浸涅消涉湧液涼淳"
    "淬清深淺添渾滋溉滑滿源濾演漫潘澡灌瀚火"
)

json_path = "data/chn_decomposition.json"

if not os.path.exists(json_path):
    print(f"❌ 找不到組件表，請確認路徑：{json_path}")
else:
    with open(json_path, 'r', encoding='utf-8') as f:
        decomp_data = json.load(f)
    
    missing = [c for c in TRAIN_CHARS if c not in decomp_data]
    
    if not missing:
        print("✅ 太棒了！100 個繁體字全部都有對應的組件資料，可以直接訓練。")
    else:
        print(f"⚠️ 注意！有 {len(missing)} 個字不在組件表中：")
        print("".join(missing))
        print("\n這會導致訓練報錯。建議：1. 用簡體字替換 2. 手動在 JSON 加入這些字的拆解。")