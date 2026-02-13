"""
MX-Font for M2 Mac (MPS)
Modified for Apple Silicon Compatibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_trainer import BaseTrainer
from .trainer_utils import cyclize, binarize_labels, expert_assign
from .hsic import RbfHSIC

import utils
from itertools import combinations


class FactTrainer(BaseTrainer):
    def __init__(self, gen, disc, g_optim, d_optim, aux_clf, ac_optim,
                 writer, logger, evaluator, test_loader, cfg):
        super().__init__(gen, disc, g_optim, d_optim, aux_clf, ac_optim,
                         writer, logger, evaluator, test_loader, cfg)
        
        # 從 cfg 中獲取在 train.py 定義好的 device (mps 或 cpu)
        self.device = cfg.device

    def sync_g_ema(self, style_imgs, char_imgs):
        org_train_mode = self.gen_ema.training
        with torch.no_grad():
            self.gen_ema.train()
            self.gen_ema.gen_from_style_char(style_imgs, char_imgs)
        self.gen_ema.train(org_train_mode)

    def train(self, loader, st_step=0, max_step=100000):

        self.gen.train()
        if self.disc is not None:
            self.disc.train()

        # loss stats: 確保這裡包含所有 plot 會用到的名稱
        losses = utils.AverageMeters("g_total", "pixel", "disc", "gen", "fm", "indp_exp", "indp_fact",
                                     "ac_s", "ac_c", "cross_ac_s", "cross_ac_c",
                                     "ac_gen_s", "ac_gen_c", "cross_ac_gen_s", "cross_ac_gen_c")
        
        discs = utils.AverageMeters("real_font", "real_uni", "fake_font", "fake_uni",
                                    "real_font_acc", "real_uni_acc",
                                    "fake_font_acc", "fake_uni_acc")
        
        stats = utils.AverageMeters("B", "ac_acc_s", "ac_acc_c", "ac_gen_acc_s", "ac_gen_acc_c")

        self.step = st_step
        self.clear_losses()

        self.logger.info("Start training ...")

        for batch in cyclize(loader):
            epoch = self.step // len(loader)
            if self.cfg.use_ddp and (self.step % len(loader)) == 0:
                loader.sampler.set_epoch(epoch)

            # 將所有 Tensor 送往 M2 MPS 裝置
            style_imgs = batch["style_imgs"].to(self.device)
            style_fids = batch["style_fids"].to(self.device)
            style_decs = batch["style_decs"]
            char_imgs = batch["char_imgs"].to(self.device)
            char_fids = batch["char_fids"].to(self.device)
            char_decs = batch["char_decs"]

            trg_imgs = batch["trg_imgs"].to(self.device)
            trg_fids = batch["trg_fids"].to(self.device)
            trg_cids = batch["trg_cids"].to(self.device)
            trg_decs = batch["trg_decs"]

            # --- Forward Logic ---
            B = len(trg_imgs)
            n_s = style_imgs.shape[1]
            n_c = char_imgs.shape[1]

            style_feats = self.gen.encode(style_imgs.flatten(0, 1))
            char_feats = self.gen.encode(char_imgs.flatten(0, 1))

            self.add_indp_exp_loss(torch.cat([style_feats["last"], char_feats["last"]]))

            style_facts_s = self.gen.factorize(style_feats, 0)
            style_facts_c = self.gen.factorize(style_feats, 1)
            char_facts_s = self.gen.factorize(char_feats, 0)
            char_facts_c = self.gen.factorize(char_feats, 1)

            self.add_indp_fact_loss(
                [style_facts_s["last"], style_facts_c["last"]], [style_facts_s["skip"], style_facts_c["skip"]],
                [char_facts_s["last"], char_facts_c["last"]], [char_facts_s["skip"], char_facts_c["skip"]],
            )

            mean_style_facts = {k: utils.add_dim_and_reshape(v, 0, (-1, n_s)).mean(1) for k, v in style_facts_s.items()}
            mean_char_facts = {k: utils.add_dim_and_reshape(v, 0, (-1, n_c)).mean(1) for k, v in char_facts_c.items()}
            gen_feats = self.gen.defactorize([mean_style_facts, mean_char_facts])
            gen_imgs = self.gen.decode(gen_feats)

            stats.updates({"B": B})

            # Discriminator Step
            real_font, real_uni, *real_feats = self.disc(trg_imgs, trg_fids, trg_cids, out_feats=self.cfg['fm_layers'])
            fake_font, fake_uni = self.disc(gen_imgs.detach(), trg_fids, trg_cids)
            self.add_gan_d_loss([real_font, real_uni], [fake_font, fake_uni])

            self.d_optim.zero_grad()
            self.d_backward()
            self.d_optim.step()

            # Generator Step
            fake_font, fake_uni, *fake_feats = self.disc(gen_imgs, trg_fids, trg_cids, out_feats=self.cfg['fm_layers'])
            self.add_gan_g_loss(fake_font, fake_uni)
            self.add_fm_loss(real_feats, fake_feats)

            def racc(x): return (x > 0.).float().mean().item()
            def facc(x): return (x < 0.).float().mean().item()

            discs.updates({
                "real_font": real_font.mean().item(), "real_uni": real_uni.mean().item(),
                "fake_font": fake_font.mean().item(), "fake_uni": fake_uni.mean().item(),
                'real_font_acc': racc(real_font), 'real_uni_acc': racc(real_uni),
                'fake_font_acc': facc(fake_font), 'fake_uni_acc': facc(fake_uni)
            }, B)

            self.add_pixel_loss(gen_imgs, trg_imgs)
            self.g_optim.zero_grad()

            self.add_ac_losses_and_update_stats(
                torch.cat([style_facts_s["last"], char_facts_s["last"]]),
                torch.cat([style_fids.flatten(), char_fids.flatten()]),
                torch.cat([style_facts_c["last"], char_facts_c["last"]]),
                style_decs + char_decs, gen_imgs, trg_fids, trg_decs, stats
            )
            self.ac_optim.zero_grad()
            self.ac_backward()
            self.ac_optim.step()

            self.g_backward()
            self.g_optim.step()

            loss_dic = self.clear_losses()
            losses.updates(loss_dic, B)

            self.accum_g()
            if self.is_bn_gen:
                self.sync_g_ema(style_imgs, char_imgs)

            # 訓練統計與保存
            if True: # 替代 cfg.gpu 判斷
                if self.step % self.cfg.tb_freq == 0:
                    self.plot(losses, discs, stats)

                if self.step % self.cfg.print_freq == 0:
                    self.log(losses, discs, stats)
                    losses.resets(); discs.resets(); stats.resets()
                    nrow = len(trg_imgs)
                    grid = utils.make_comparable_grid(trg_imgs.detach().cpu(), gen_imgs.detach().cpu(), nrow=nrow)
                    self.writer.add_image("last", grid)

                if self.step > 0 and self.step % self.cfg.val_freq == 0:
                    epoch = self.step / len(loader)
                    self.logger.info("Validation at Epoch = {:.3f}".format(epoch))
                    if not self.is_bn_gen:
                        self.sync_g_ema(style_imgs, char_imgs)
                    self.evaluator.comparable_val_saveimg(self.gen_ema, self.test_loader, self.step, n_row=self.test_n_row)
                    self.save(loss_dic['g_total'], self.cfg.save, self.cfg.get('save_freq', self.cfg.val_freq))

            if self.step >= max_step: break
            self.step += 1

        self.logger.info("Iteration finished.")

    # --- Override Plot 解決 AverageMeters 'ac' 錯誤 ---
    def plot(self, losses, discs, stats):
        tag_scalar_dic = {
            'train/g_total_loss': losses.g_total.val,
            'train/pixel_loss': losses.pixel.val,
            'train/indp_exp_loss': losses.indp_exp.val,
            'train/indp_fact_loss': losses.indp_fact.val,
        }
        # 使用 ac_s 和 ac_c 而不是父類別預期的 ac
        tag_scalar_dic.update({
            'train/ac_loss_s': losses.ac_s.val,
            'train/ac_loss_c': losses.ac_c.val,
            'train/ac_acc_s': stats.ac_acc_s.val,
            'train/ac_acc_c': stats.ac_acc_c.val
        })
        self.writer.add_scalars(tag_scalar_dic, self.step)

    # --- 其餘輔助方法保持你的邏輯 ---
    def add_indp_exp_loss(self, exps):
        # 【新增這段】如果權重是 0，直接跳出，連算都不要算！
        if self.cfg["indp_exp_w"] == 0.0:
            return
        
        exps = [F.adaptive_avg_pool2d(exps[:, i], 1).squeeze() for i in range(exps.shape[1])]
        exp_pairs = [*combinations(exps, 2)]
        crit = RbfHSIC(1)
        for pair in exp_pairs:
            self.add_loss(pair, self.g_losses, "indp_exp", self.cfg["indp_exp_w"], crit)

    def add_indp_fact_loss(self, *exp_pairs):
        # 【新增這段】同樣，權重是 0 就直接跳出
        if self.cfg["indp_fact_w"] == 0.0:
            return
        
        pairs = []
        for _exp1, _exp2 in exp_pairs:
            _pairs = [(F.adaptive_avg_pool2d(_exp1[:, i], 1).squeeze(),
                       F.adaptive_avg_pool2d(_exp2[:, i], 1).squeeze()) for i in range(_exp1.shape[1])]
            pairs += _pairs
        crit = RbfHSIC(1)
        for pair in pairs:
            self.add_loss(pair, self.g_losses, "indp_fact", self.cfg["indp_fact_w"], crit)

    def infer_comp_ac(self, fact_experts, comp_ids):
        B, n_experts = fact_experts.shape[:2]
        ac_logit_s_flat, ac_logit_c_flat = self.aux_clf(fact_experts.flatten(0, 1))
        
        # 1. Style AC 部分 (Style 分類損失)
        ac_prob_s_flat = nn.Softmax(dim=-1)(ac_logit_s_flat)
        
        # 【修改】改用 clamp，強制機率最小為 1e-7，徹底防止 log(0) 炸掉
        ac_prob_s_flat = torch.clamp(ac_prob_s_flat, min=1e-7, max=1.0)
        
        uniform_dist_s = torch.zeros_like(ac_prob_s_flat).fill_((1./ac_logit_s_flat.shape[-1])).to(self.device)
        uniform_loss_s = F.kl_div(ac_prob_s_flat.log(), uniform_dist_s, reduction="batchmean")

        # 2. Component AC 部分 (組件分類損失 - 易出錯區)
        ac_logit_c = ac_logit_c_flat.reshape((B, n_experts, -1))
        binary_comp_ids = binarize_labels(comp_ids, ac_logit_c.shape[-1]).to(self.device)
        ac_loss_c = torch.as_tensor(0.).to(self.device)
        accs = 0.

        for _b_comp_id, _logit in zip(binary_comp_ids, ac_logit_c):
            # 統一搬到 CPU 處理，徹底避開 M2 Mac 的裝置對齊問題
            _logit_cpu = _logit.detach().cpu()
            _b_comp_id_cpu = _b_comp_id.detach().cpu()
            
            _prob_cpu = nn.Softmax(dim=-1)(_logit_cpu)
            # 這裡用 nan_to_num 即可，因為不做 log 運算
            _prob_cpu = torch.nan_to_num(_prob_cpu, nan=0.0)
            
            T_probs = _prob_cpu.T[_b_comp_id_cpu]
            
            try:
                # 在 CPU 上執行專家分配 (scipy)
                cids, eids = expert_assign(T_probs)
                
                # 在 CPU 上計算 Cross Entropy 必要的索引
                _max_ids_cpu = torch.where(_b_comp_id_cpu)[0][cids]
                
                # 計算該樣本的 Loss 並搬回 self.device (MPS)
                sample_loss = F.cross_entropy(_logit_cpu[eids], _max_ids_cpu).to(self.device)
                ac_loss_c += sample_loss
                
                # 計算準確度
                acc = T_probs[cids, eids].sum() / n_experts
                accs += acc
            except Exception:
                # 如果矩陣運算失敗，跳過此樣本，避免訓練中斷
                continue

        # 回傳結果，確保 accs 格式正確
        return ac_loss_c / B, uniform_loss_s, (accs / B).item() if torch.is_tensor(accs) else accs / B
    def infer_style_ac(self, fact_experts, style_ids):
        B, n_experts = fact_experts.shape[:2]
        ac_in_flat = fact_experts.flatten(0, 1)
        style_ids_flat = style_ids.repeat_interleave(n_experts, dim=0)

        ac_logit_s_flat, ac_logit_c_flat = self.aux_clf(ac_in_flat)
        ac_loss_s = F.cross_entropy(ac_logit_s_flat, style_ids_flat)

        n_c = ac_logit_c_flat.shape[-1]
        
        # 1. Component 部分
        ac_prob_c_flat = nn.Softmax(dim=-1)(ac_logit_c_flat)
        
        # 【關鍵修改】使用 clamp 強制鎖定數值範圍，比 nan_to_num 更安全
        ac_prob_c_flat = torch.clamp(ac_prob_c_flat, min=1e-7, max=1.0)
        
        uniform_dist_c = torch.zeros_like(ac_prob_c_flat).fill_((1./n_c)).to(self.device)
        uniform_loss_c = F.kl_div(ac_prob_c_flat.log(), uniform_dist_c, reduction="batchmean")

        _, est_ids = ac_logit_s_flat.max(dim=-1)
        acc = (style_ids_flat == est_ids).float().mean().item()

        return ac_loss_s, uniform_loss_c, acc
    
    def add_ac_losses_and_update_stats(self, style_facts, style_ids, char_facts, comp_ids,
                                       gen_imgs, gen_style_ids, gen_comp_ids, stats):
        """
        計算輔助分類器 (Auxiliary Classifier) 的損失並更新訓練統計
        """
        # 1. 計算真實圖片的風格與組件損失
        ac_loss_s, cross_ac_loss_s, acc_s = self.infer_style_ac(style_facts, style_ids)
        ac_loss_c, cross_ac_loss_c, acc_c = self.infer_comp_ac(char_facts, comp_ids)

        # 2. 將計算結果存入 self.ac_losses 字典
        self.ac_losses["ac_s"] = ac_loss_s * self.cfg["ac_w"]
        self.ac_losses["ac_c"] = ac_loss_c * self.cfg["ac_w"]
        self.ac_losses["cross_ac_s"] = cross_ac_loss_s * self.cfg["ac_w"] * self.cfg["ac_cross_w"]
        self.ac_losses["cross_ac_c"] = cross_ac_loss_c * self.cfg["ac_w"] * self.cfg["ac_cross_w"]
        
        # 更新統計數據
        stats.ac_acc_s.update(acc_s, len(style_ids))
        stats.ac_acc_c.update(acc_c, sum([*map(len, comp_ids)]))

        # 3. 為了確保生成圖片的一致性，使用 Generator (EMA) 的特徵再計算一次
        with torch.no_grad():
            gen_feats = self.gen_ema.encode(gen_imgs)
            gen_style_facts = self.gen_ema.factorize(gen_feats, 0)["last"]
            gen_char_facts = self.gen_ema.factorize(gen_feats, 1)["last"]

        gen_ac_loss_s, gen_cross_ac_loss_s, gen_acc_s = self.infer_style_ac(gen_style_facts, gen_style_ids)
        gen_ac_loss_c, gen_cross_ac_loss_c, gen_acc_c = self.infer_comp_ac(gen_char_facts, gen_comp_ids)
        
        stats.ac_gen_acc_s.update(gen_acc_s, len(gen_style_ids))
        stats.ac_gen_acc_c.update(gen_acc_c, sum([*map(len, gen_comp_ids)]))

        # 儲存固定 (frozen) 的 AC 損失
        self.frozen_ac_losses['ac_gen_s'] = gen_ac_loss_s * self.cfg['ac_gen_w']
        self.frozen_ac_losses['ac_gen_c'] = gen_ac_loss_c * self.cfg['ac_gen_w']
        self.frozen_ac_losses['cross_ac_gen_s'] = gen_cross_ac_loss_s * self.cfg['ac_gen_w'] * self.cfg["ac_cross_w"]
        self.frozen_ac_losses['cross_ac_gen_c'] = gen_cross_ac_loss_c * self.cfg['ac_gen_w'] * self.cfg["ac_cross_w"]

    def log(self, L, D, S):
        """
        覆寫父類別的 log 函式，確保格式與 FactTrainer 的 Loss 名稱一致
        """
        self.logger.info(
            f"Step {self.step:7d}\n"
            f"{'|D':<12} {L.disc.avg:7.3f} {'|G':<12} {L.gen.avg:7.3f} {'|FM':<12} {L.fm.avg:7.3f} "
            f"{'|R_font':<12} {D.real_font_acc.avg:7.3f} {'|F_font':<12} {D.fake_font_acc.avg:7.3f}\n"
            f"{'|AC_s':<12} {L.ac_s.avg:7.3f} {'|AC_c':<12} {L.ac_c.avg:7.3f} "
            f"{'|AC_acc_s':<12} {S.ac_acc_s.avg:7.1%} {'|AC_acc_c':<12} {S.ac_acc_c.avg:7.1%}\n"
            f"{'|L1':<12} {L.pixel.avg:7.3f} {'|INDP_EXP':<12} {L.indp_exp.avg:7.4f} "
            f"{'|INDP_FACT':<12} {L.indp_fact.avg:7.4f}"
        )