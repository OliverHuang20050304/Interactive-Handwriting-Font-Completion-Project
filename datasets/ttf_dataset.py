import os
from pathlib import Path
from itertools import chain
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class TTFTrainDataset(Dataset):
    def __init__(self, data_dir, primals, decomposition, transform=None,
                 n_in_s=3, n_in_c=3, **kwargs):
        
        # 取得目前這個 ttf_dataset.py 檔案所在的絕對路徑
        # 然後往上推兩層回到 mxfont 根目錄
        current_file_path = Path(__file__).resolve()
        project_root = current_file_path.parent.parent 
        
        # 根據你的截圖，data 資料夾在 mxfont 內
        # 所以路徑應該是 project_root / data / target / train
        base_dir = project_root / "png_data"
        
        self.target_dir = base_dir / "target" / "train"
        self.source_dir = base_dir / "source" / "train"
        
        # 輸出目前嘗試讀取的路徑，方便除錯
        print(f"🔍 正在嘗試讀取資料路徑: {self.target_dir}")

        if not self.target_dir.exists():
            # 如果還是找不到，嘗試看看是不是在 data/ttfs 下 (對應你截圖中的 ttfs 資料夾)
            self.target_dir = base_dir / "ttfs" / "target" / "train"
            self.source_dir = base_dir / "ttfs" / "source" / "train"
            
        if not self.target_dir.exists():
            raise FileNotFoundError(
                f"❌ 依然找不到目錄。\n"
                f"目前偵測到的根目錄是: {project_root}\n"
                f"請確認你的 PNG 檔案是否放在: {project_root}/data/target/train/"
            )

        self.primals = primals
        self.decomposition = decomposition

        # 獲取所有圖片檔名
        self.filenames = sorted([f for f in os.listdir(self.target_dir) if f.endswith('.png')])
        
        if len(self.filenames) == 0:
            raise RuntimeError(f"❌ 在 {self.target_dir} 中找不到 PNG 檔案！")
            
        self.file_to_char = {f: chr(int(f.split('.')[0])) for f in self.filenames}
        self.chars = sorted([self.file_to_char[f] for f in self.filenames])

        self.transform = transform
        self.n_in_s = n_in_s
        self.n_in_c = n_in_c
        self.n_chars = len(self.chars)
        self.n_fonts = 1
    def __getitem__(self, index):
        trg_filename = self.filenames[index]
        char = self.file_to_char[trg_filename]
        
        # 1. 載入目標圖 (辰宇落雁體) 與 來源圖 (蘋方體)
        trg_img = self.transform(Image.open(self.target_dir / trg_filename).convert('L'))
        src_img = self.transform(Image.open(self.source_dir / trg_filename).convert('L'))
        
        # 2. 獲取組件標籤
        trg_dec = [self.primals.index(x) for x in self.decomposition[char]]

        # 3. 隨機抽取同風格的其他字 (Style Samples)
        style_filenames = random.sample([f for f in self.filenames if f != trg_filename], self.n_in_s)
        style_imgs = torch.stack([self.transform(Image.open(self.target_dir / f).convert('L')) for f in style_filenames])
        style_decs = [[self.primals.index(x) for x in self.decomposition[self.file_to_char[f]]] for f in style_filenames]

        # 4. 隨機抽取同內容的其他風格 (由於你只有一種 Target，這裡我們直接用 Source 代替)
        char_imgs = torch.stack([src_img] * self.n_in_c)
        char_decs = [trg_dec] * self.n_in_c
        char_fids = [0] * self.n_in_c # 只有一種字體

        return {
            "trg_imgs": trg_img,
            "trg_decs": trg_dec,
            "trg_fids": torch.LongTensor([0]),
            "trg_cids": torch.LongTensor([self.chars.index(char)]),
            "style_imgs": style_imgs,
            "style_decs": style_decs,
            "style_fids": torch.LongTensor([0] * self.n_in_s),
            "char_imgs": char_imgs,
            "char_decs": char_decs,
            "char_fids": torch.LongTensor(char_fids)
        }

    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def collate_fn(batch):
        _ret = {}
        for dp in batch:
            for key, value in dp.items():
                saved = _ret.get(key, [])
                _ret.update({key: saved + [value]})

        return {
            "trg_imgs": torch.stack(_ret["trg_imgs"]),
            "trg_decs": _ret["trg_decs"],
            "trg_fids": torch.cat(_ret["trg_fids"]),
            "trg_cids": torch.cat(_ret["trg_cids"]),
            "style_imgs": torch.stack(_ret["style_imgs"]),
            "style_decs": [*chain(*_ret["style_decs"])],
            "style_fids": torch.stack(_ret["style_fids"]),
            "char_imgs": torch.stack(_ret["char_imgs"]),
            "char_decs": [*chain(*_ret["char_decs"])],
            "char_fids": torch.stack(_ret["char_fids"])
        }

# 驗證集 (Validation) 也請依照相同邏輯簡化
class TTFValDataset(Dataset):
    def __init__(self, data_dir, source_font, char_filter, n_ref=4, n_gen=20, transform=None, **kwargs):
        # 取得專案根目錄
        current_file_path = Path(__file__).resolve()
        project_root = current_file_path.parent.parent 
        base_dir = project_root / "png_data"
        
        # 指向你的測試/驗證資料夾
        self.target_dir = base_dir / "target" / "test"
        self.source_dir = base_dir / "source" / "test"
        
        if not self.target_dir.exists():
            # 容錯：檢查 data/ttfs 下
            self.target_dir = base_dir / "ttfs" / "target" / "test"
            self.source_dir = base_dir / "ttfs" / "source" / "test"

        self.transform = transform
        
        # 獲取測試集的圖片檔名
        self.filenames = sorted([f for f in os.listdir(self.target_dir) if f.endswith('.png')])
        self.file_to_char = {f: chr(int(f.split('.')[0])) for f in self.filenames}
        
        # 為了讓模型驗證，我們需要定義參考風格字與待生成字
        # 這裡簡單處理：全部測試字都作為生成目標
        self.ref_filenames = random.sample(self.filenames, min(n_ref, len(self.filenames)))
        self.gen_filenames = self.filenames

        self.ref_chars = [self.file_to_char[f] for f in self.ref_filenames]
        self.gen_chars = [self.file_to_char[f] for f in self.gen_filenames]

    def __getitem__(self, index):
        trg_filename = self.gen_filenames[index]
        char = self.file_to_char[trg_filename]

        # 風格參考圖 (從測試集中選取)
        ref_imgs = torch.stack([self.transform(Image.open(self.target_dir / f).convert('L'))
                                for f in self.ref_filenames])

        # 來源圖與目標圖
        source_img = self.transform(Image.open(self.source_dir / trg_filename).convert('L'))
        trg_img = self.transform(Image.open(self.target_dir / trg_filename).convert('L'))

        return {
            "style_imgs": ref_imgs,
            "source_imgs": source_img,
            "fonts": "target_font",
            "chars": char,
            "trg_imgs": trg_img
        }

    def __len__(self):
        return len(self.gen_filenames)

    @staticmethod
    def collate_fn(batch):
        _ret = {}
        for dp in batch:
            for key, value in dp.items():
                saved = _ret.get(key, [])
                _ret.update({key: saved + [value]})

        return {
            "style_imgs": torch.stack(_ret["style_imgs"]),
            "source_imgs": torch.stack(_ret["source_imgs"]),
            "fonts": _ret["fonts"],
            "chars": _ret["chars"],
            "trg_imgs": torch.stack(_ret["trg_imgs"])
        }