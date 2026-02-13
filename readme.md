# Interactive CJK Handwriting Font Completion
## 互動式 CJK 手寫字體補全計畫

本專案結合深度學習 (MX-Font) 與互動式介面，協助使用者透過少量手寫樣本，快速生成高品質且具備個人風格的中韓字體。

## 🏆 最新進度 (2026.02.13 Updated)
- **模型訓練穩定化**：修復 Loss NaN 問題，完成個人手寫資料集 Fine-tuning。
- **推論品質增強**：引入 **OpenCV 形態學膨脹 (Dilation)** 與對比度增強技術，解決生成字體筆畫過細、斷裂問題，大幅提升手寫紮實度。
- **批量生成工具**：新增 `gen_all.py`，可自動化生成大量漢字圖片。

## 🚀 核心功能
1. **AI 自動生成**：基於改良版 MX-Font，少量樣本生成全字集。
2. **筆畫自動補全**：透過影像處理技術自動修補斷筆與鬼影。
3. **怪字偵測**：Confidence Scoring 系統（紅綠燈評分），定位需修正字形。
4. **互動修正**：支援粗細、斜度即時調整與手寫板重寫。

## 💻 如何執行 (Usage)
請先安裝影像處理依賴：
```bash
pip install opencv-python
```

### 1. 單字生成測試
生成指定文字並自動應用「補墨水」效果：
```bash
python inference.py --gen result/checkpoints/last.pth --ref png_data/target/train --text "天地玄黃"
```

### 2. 批量生成
生成常用中文字圖片至 `output_images/`：
```bash
python gen_all.py
```



## 👥 團隊與分工
- **Flora**：審美標準、資料集收集、怪字判定與評估。
- **Oliver**：AI 模型訓練、推論演算法優化 (OpenCV)、系統架構。

## 📅 開發時程
- [x] **Week 1-2**：建立 Baseline，完成模型訓練修正。
- [x] **Week 3**：推論引擎優化 (筆畫增強) 與批量生成工具。
- [ ] **Week 4-5**：UI/UX 設計、互動修正功能開發。
- [ ] **Week 6-8**：系統整合與最終發表。
