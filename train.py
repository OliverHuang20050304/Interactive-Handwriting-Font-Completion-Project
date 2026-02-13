"""
MX-Font for M2 Mac (MPS)
Modified for Apple Silicon Compatibility
"""

import json
import sys
from pathlib import Path
import argparse

import torch
import torch.optim as optim
# 移除 cudnn，MPS 不需要
# import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torchvision import transforms
import numpy as np
from sconf import Config, dump_args
import utils
from utils import Logger

from models import Generator, disc_builder, aux_clf_builder
from models.modules import weights_init
from trainer import FactTrainer, Evaluator, load_checkpoint
from datasets import get_trn_loader, get_val_loader


def setup_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_paths", nargs="+", help="path/to/config.yaml")

    args, left_argv = parser.parse_known_args()

    cfg = Config(*args.config_paths, default="cfgs/defaults.yaml",
                 colorize_modified_item=True)
    cfg.argv_update(left_argv)

    # M2 Mac 建議關閉 DDP 與多線程 worker 以節省統一記憶體
    cfg.use_ddp = False 
    cfg.n_workers = 0 

    cfg.work_dir = Path(cfg.work_dir)
    (cfg.work_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    return args, cfg


def setup_transforms(cfg):
    # 修正：M2 Mac 的 Pillow 版本建議移除 fillcolor，改用 fill
    if cfg.dset_aug.random_affine:
        aug_transform = [
            transforms.ToPILImage(),
            transforms.RandomAffine(
                degrees=10, translate=(0.03, 0.03), scale=(0.9, 1.1), shear=10, fill=255
            )
        ]
    else:
        aug_transform = []

    tensorize_transform = [transforms.Resize((128, 128)), transforms.ToTensor()]
    if cfg.dset_aug.normalize:
        tensorize_transform.append(transforms.Normalize([0.5], [0.5]))
        cfg.g_args.dec.out = "tanh"

    trn_transform = transforms.Compose(aug_transform + tensorize_transform)
    val_transform = transforms.Compose(tensorize_transform)

    return trn_transform, val_transform


def train(args, cfg):
    # 自動偵測 M2 GPU (MPS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    cfg.device = device
    print(f"✅ 正在使用裝置: {device}")

    logger_path = cfg.work_dir / "log.log"
    logger = Logger.get(file_path=logger_path, level="info", colorize=True)

    image_scale = 0.5
    image_path = cfg.work_dir / "images"
    writer = utils.DiskWriter(image_path, scale=image_scale)
    cfg.tb_freq = -1

    args_str = dump_args(args)
    logger.info("Run Argv:\n> {}".format(" ".join(sys.argv)))
    logger.info("Args:\n{}".format(args_str))
    logger.info("Configs:\n{}".format(cfg.dumps()))

    logger.info("Get dataset ...")

    trn_transform, val_transform = setup_transforms(cfg)

    primals = json.load(open(cfg.primals))
    # 確保讀取你修復後的繁體字 JSON
    decomposition = json.load(open(cfg.decomposition))
    n_comps = len(primals)

    trn_dset, trn_loader = get_trn_loader(cfg.dset.train,
                                          primals,
                                          decomposition,
                                          trn_transform,
                                          use_ddp=False,
                                          batch_size=cfg.batch_size,
                                          num_workers=cfg.n_workers,
                                          shuffle=True)

    test_dset, test_loader = get_val_loader(cfg.dset.val,
                                            val_transform,
                                            batch_size=cfg.batch_size,
                                            num_workers=cfg.n_workers,
                                            shuffle=False)

    logger.info("Build model ...")
    
    # Generator
    g_kwargs = cfg.get("g_args", {})
    gen = Generator(1, cfg.C, 1, **g_kwargs)
    gen.to(device) # 修改：.cuda() -> .to(device)
    gen.apply(weights_init(cfg.init))

    # Discriminator
    d_kwargs = cfg.get("d_args", {})
    disc = disc_builder(cfg.C, trn_dset.n_fonts, trn_dset.n_chars, **d_kwargs)
    disc.to(device) # 修改：.cuda() -> .to(device)
    disc.apply(weights_init(cfg.init))

    # Aux Classifier
    aux_clf = aux_clf_builder(gen.feat_shape["last"], trn_dset.n_fonts, n_comps, **cfg.ac_args)
    aux_clf.to(device) # 修改：.cuda() -> .to(device)
    aux_clf.apply(weights_init(cfg.init))

    g_optim = optim.Adam(gen.parameters(), lr=cfg.g_lr, betas=cfg.adam_betas)
    d_optim = optim.Adam(disc.parameters(), lr=cfg.d_lr, betas=cfg.adam_betas)
    ac_optim = optim.Adam(aux_clf.parameters(), lr=cfg.ac_lr, betas=cfg.adam_betas)

    st_step = 0
    if cfg.resume:
        # 注意：load_checkpoint 內部也需要確保 map_location='cpu' 或 'mps'
        st_step, loss = load_checkpoint(cfg.resume, gen, disc, aux_clf, g_optim, d_optim, ac_optim, cfg.force_resume)
        logger.info("Resumed checkpoint from {} (Step {}, Loss {:7.3f})".format(cfg.resume, st_step, loss))

    evaluator = Evaluator(writer)

    trainer = FactTrainer(gen, disc, g_optim, d_optim,
                          aux_clf, ac_optim,
                          writer, logger,
                          evaluator, test_loader,
                          cfg)

    trainer.train(trn_loader, st_step, cfg.max_iter)


def main():
    args, cfg = setup_args_and_config()

    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    # 直接調用訓練，略過 DDP 邏輯
    train(args, cfg)


if __name__ == "__main__":
    main()