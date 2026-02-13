import json

json_path = "data/chn_decomposition.json"
fixed_json_path = "data/decomposition_fixed.json"

# 定義缺少的繁體字與其對應的簡體字（用來借用組件分解資料）
# 這樣模型就能知道這些字該用哪些「專家」來處理
mapping = {
    "讓": "让", "靈": "灵", "屬": "属", "鹽": "盐", "鑲": "镶",
    "關": "关", "艷": "艳", "隱": "隐", "慶": "庆", "憲": "宪",
    "派": "派", "深": "深", "添": "添", "滿": "满"
}

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for trad, simp in mapping.items():
    if simp in data:
        data[trad] = data[simp]
        print(f"✅ 已修復: {trad} -> 使用 {simp} 的組件資料")
    else:
        # 如果連簡體都沒有，就找一個結構最像的字代替，或直接複製一個常用的
        data[trad] = data.get("永", []) 
        print(f"⚠️ 警告: 找不到 {simp}, 暫時借用 '永' 的組件")

with open(fixed_json_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"\n🎉 修復完成！請在後續設定中使用: {fixed_json_path}")