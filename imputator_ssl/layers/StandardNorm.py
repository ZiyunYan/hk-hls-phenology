import torch
import torch.nn as nn


class Normalize(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=False, subtract_last=False, non_norm=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str, mask=None):
        if mode == 'norm':
            self._get_statistics(x, mask)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x, mask=None):
        dim2reduce = tuple(range(1, x.ndim - 1))

        if mask is not None:
            # 使用mask计算有效值的统计量
            if self.subtract_last:
                # 获取最后一个有效值，而不是简单地取最后一个时间步
                last_valid_idx = mask.sum(dim=1, keepdim=True) - 1  # [batch, 1, feature]
                batch_idx = torch.arange(x.size(0))[:, None, None].expand(-1, 1, x.size(-1))
                self.last = x[batch_idx, last_valid_idx, torch.arange(x.size(-1))[None, None, :]].unsqueeze(1)
            else:
                # 计算有效值的均值
                masked_sum = (x * mask).sum(dim=dim2reduce, keepdim=True)
                valid_counts = mask.sum(dim=dim2reduce, keepdim=True).clamp(min=1)  # 避免除零
                self.mean = (masked_sum / valid_counts).detach()

            # 计算有效值的标准差
            if not self.subtract_last:
                masked_var = ((x - self.mean) * mask) ** 2
                masked_var_sum = masked_var.sum(dim=dim2reduce, keepdim=True)
                self.stdev = torch.sqrt(masked_var_sum / valid_counts + self.eps).detach()
        else:
            # 原有的计算逻辑
            if self.subtract_last:
                self.last = x[:, -1, :].unsqueeze(1)
            else:
                self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
