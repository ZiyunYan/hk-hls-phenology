from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F


class VectorQuantizer2(nn.Module):
    def __init__(
            self, vocab_size, Cvae, using_znorm, beta: float = 0.25,
            default_qresi_counts=0, v_patch_nums=None, quant_resi=0.5, share_quant_resi=4,
    ):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.Cvae: int = Cvae
        self.using_znorm: bool = using_znorm
        self.v_patch_nums: Tuple[int] = v_patch_nums

        self.quant_resi_ratio = quant_resi
        if share_quant_resi == 0:  # 不共享：每个尺度有独立的 Phi 层
            self.quant_resi = PhiNonShared(
                [(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in
                 range(default_qresi_counts or len(self.v_patch_nums))])
        elif share_quant_resi == 1:  # 完全共享：所有尺度共享一个 Phi 层
            self.quant_resi = PhiShared(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
        else:  # 部分共享：share_quant_resi 个 Phi 层
            self.quant_resi = PhiPartiallyShared(nn.ModuleList(
                [(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in
                 range(share_quant_resi)]))

        self.register_buffer('ema_vocab_hit_SV', torch.full((len(self.v_patch_nums), self.vocab_size), fill_value=0.0))
        self.record_hit = 0

        self.beta: float = beta
        self.embedding = nn.Embedding(self.vocab_size, self.Cvae)

        self.prog_si = -1  # 渐进式训练：暂不支持

    def eini(self, eini):
        if eini > 0:
            nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0:
            self.embedding.weight.data.uniform_(-abs(eini) / self.vocab_size, abs(eini) / self.vocab_size)

    def extra_repr(self) -> str:
        return f'{self.v_patch_nums}, znorm={self.using_znorm}, beta={self.beta}  |  S={len(self.v_patch_nums)}, quant_resi={self.quant_resi_ratio}'

    def forward(self, f_BCT: torch.Tensor, ret_usages=False) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        dtype = f_BCT.dtype
        if dtype != torch.float32: f_BCT = f_BCT.float()
        B, C, T = f_BCT.shape
        f_no_grad = f_BCT.detach()

        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        with torch.cuda.amp.autocast(enabled=False):
            mean_vq_loss: torch.Tensor = 0.0
            vocab_hit_V = torch.zeros(self.vocab_size, dtype=torch.float, device=f_BCT.device)
            SN = len(self.v_patch_nums)
            for si, pn in enumerate(self.v_patch_nums):  # 从小到大遍历时间分辨率
                # 查找最近的嵌入向量
                if self.using_znorm:
                    rest_NC = F.interpolate(f_rest, size=pn, mode='linear').permute(0, 2, 1).reshape(-1, C) if (
                                si != SN - 1) else f_rest.permute(0, 2, 1).reshape(-1, C)
                    rest_NC = F.normalize(rest_NC, dim=-1)
                    idx_N = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
                else:
                    rest_NC = F.interpolate(f_rest, size=pn, mode='linear').permute(0, 2, 1).reshape(-1, C) if (
                                si != SN - 1) else f_rest.permute(0, 2, 1).reshape(-1, C)
                    d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(
                        self.embedding.weight.data.square(), dim=1, keepdim=False)
                    d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)
                    idx_N = torch.argmin(d_no_grad, dim=1)

                hit_V = idx_N.bincount(minlength=self.vocab_size).float()

                # 计算损失和量化表示
                idx_BT = idx_N.view(B, pn)
                h_BCT = F.interpolate(self.embedding(idx_BT).transpose(1, 2), size=T, mode='linear').contiguous() if (
                            si != SN - 1) else self.embedding(idx_BT).transpose(1, 2).contiguous()
                h_BCT = self.quant_resi[si / (SN - 1)](h_BCT)
                f_hat = f_hat + h_BCT
                f_rest -= h_BCT

                if self.training:
                    if self.record_hit == 0:
                        self.ema_vocab_hit_SV[si].copy_(hit_V)
                    elif self.record_hit < 100:
                        self.ema_vocab_hit_SV[si].mul_(0.9).add_(hit_V.mul(0.1))
                    else:
                        self.ema_vocab_hit_SV[si].mul_(0.99).add_(hit_V.mul(0.01))
                    self.record_hit += 1
                vocab_hit_V.add_(hit_V)
                mean_vq_loss += F.mse_loss(f_hat.data, f_BCT).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)

            mean_vq_loss *= 1. / SN
            f_hat = (f_hat.data - f_no_grad).add_(f_BCT)

        if ret_usages:
            usages = [(self.ema_vocab_hit_SV[si] >= 0).float().mean().item() * 100 for si in
                      range(len(self.v_patch_nums))]
        else:
            usages = None
        return f_hat, usages, mean_vq_loss

    def embed_to_fhat(self, ms_h_BCT: List[torch.Tensor], all_to_max_scale=True, last_one=False) -> Union[
        List[torch.Tensor], torch.Tensor]:
        ls_f_hat_BCT = []
        B = ms_h_BCT[0].shape[0]
        T = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        if all_to_max_scale:
            f_hat = ms_h_BCT[0].new_zeros(B, self.Cvae, T, dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums):
                h_BCT = ms_h_BCT[si]
                if si < len(self.v_patch_nums) - 1:
                    h_BCT = F.interpolate(h_BCT, size=T, mode='linear')
                h_BCT = self.quant_resi[si / (SN - 1)](h_BCT)
                f_hat.add_(h_BCT)
                if last_one:
                    ls_f_hat_BCT = f_hat
                else:
                    ls_f_hat_BCT.append(f_hat.clone())
        else:
            f_hat = ms_h_BCT[0].new_zeros(B, self.Cvae, self.v_patch_nums[0], dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums):
                f_hat = F.interpolate(f_hat, size=pn, mode='linear')
                h_BCT = self.quant_resi[si / (SN - 1)](ms_h_BCT[si])
                f_hat.add_(h_BCT)
                if last_one:
                    ls_f_hat_BCT = f_hat
                else:
                    ls_f_hat_BCT.append(f_hat)

        return ls_f_hat_BCT

    def f_to_idxBl_or_fhat(self, f_BCT: torch.Tensor, to_fhat: bool, v_patch_nums: Optional[Sequence[int]] = None) -> \
    List[Union[torch.Tensor, torch.LongTensor]]:
        B, C, T = f_BCT.shape
        f_no_grad = f_BCT.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)

        f_hat_or_idx_Bl: List[torch.Tensor] = []

        patch_nums = v_patch_nums or self.v_patch_nums
        assert patch_nums[-1] == T, f'{patch_nums[-1]=} != {T=}'

        SN = len(patch_nums)
        for si, pn in enumerate(patch_nums):
            if 0 <= self.prog_si < si: break
            z_NC = F.interpolate(f_rest, size=pn, mode='linear').permute(0, 2, 1).reshape(-1, C) if (
                        si != SN - 1) else f_rest.permute(0, 2, 1).reshape(-1, C)
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(
                    self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)
                idx_N = torch.argmin(d_no_grad, dim=1)

            idx_BT = idx_N.view(B, pn)
            h_BCT = F.interpolate(self.embedding(idx_BT).transpose(1, 2), size=T, mode='linear').contiguous() if (
                        si != SN - 1) else self.embedding(idx_BT).transpose(1, 2).contiguous()
            h_BCT = self.quant_resi[si / (SN - 1)](h_BCT)
            f_hat.add_(h_BCT)
            f_rest.sub_(h_BCT)
            f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, pn))

        return f_hat_or_idx_Bl


class Phi(nn.Conv1d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks // 2)
        self.resi_ratio = abs(quant_resi)

    def forward(self, h_BCT):
        return h_BCT * (1 - self.resi_ratio) + super().forward(h_BCT) * self.resi_ratio


class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi

    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        K = len(qresi)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())

    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'