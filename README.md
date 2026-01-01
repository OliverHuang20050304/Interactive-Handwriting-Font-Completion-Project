# Interactive CJK Handwriting Font Completion 互動式 CJK 手寫字體補全計畫
這是一個為期 8 週的開發專案，旨在透過 AI 技術與互動式介面，協助使用者快速生成高品質且具備個人風格的中文與韓文字體 。



## 📌 專案概述 (Project Overview)
本專案結合了深度學習與互動式修正工具，解決手寫字體開發時繁瑣的流程 。使用者僅需提供少量的手寫樣本，系統即可自動生成全字集，並由 AI 偵測生成品質不佳的「怪字」，引導使用者進行精準修正 。



## 🚀 核心功能 (Core Features)

AI 自動字體生成：基於 MX-Font 模型架構，透過少量樣本生成完整的 CJK 字集 。



怪字偵測系統 (Confidence Scoring)：自動對生成字體進行評分（紅/黃/綠燈標籤），快速定位需要人工修正的字形 。



互動式修正面板：提供粗細 (Weight)、斜度 (Slant)、圓角 (Roundness) 的調整拉桿，並支援手寫板即時重寫 。


線上即時微調 (Online Fine-tuning)：系統能根據使用者的修正即時優化模型，提升後續生成的風格一致性 。


## 🛠️ 技術架構 (Technical Architecture)

模型 (Model)：MX-Font Baseline, Style-content Encoder, Radical-based Feature Extraction 。



評分機制：Confidence Scoring v1-v3 。


部署 (Deployment)：Vercel / HuggingFace 。

## 👥 團隊成員與分工 (Team)

Flora：字體審美、手寫資料集收集、語言結構分析、怪字判定標準與評估 。


Oliver：AI 模型訓練與微調、前後端架構開發、API 優化 。

## 📅 開發時程 (Timeline)

Week 1-2：建立 Baseline 與怪字判定標準 。


Week 3-5：UI/UX 原型設計、部件分析與 Online Fine-tuning 流程開發 。


Week 6-8：系統整合、盲測、效能優化與最終成果發表 。