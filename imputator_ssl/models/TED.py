import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
# 假设 these imports are available from your project setup
from layers.Self_layers import AttentionBlock, TimesPatchEmbed 
from layers.Embedding import PositionalEncoding
# from utils.losses import smooth_loss, mse_loss
from utils.tools import apply_mask, apply_scaling_and_noise, apply_channel_masking, patchify, unpatchify, random_patch_masking, create_smooth_target, get_student_input, get_teacher_input, generate_local_view_crop, generate_local_view_random_sample, random_patch_masking_dinov3_style, imputator_sliding_window_overlap
# Loss 计算在外部 DINOCriteria 中进行，Model 中不定义 loss 函数


class DINOHead(nn.Module):
    """
    根据dinov3源码优化的DINOHead结构
    结构：MLP (in_dim -> hidden_dim -> hidden_dim -> bottleneck_dim) -> L2 Normalize -> Linear (bottleneck_dim -> out_dim)
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = self._build_mlp(
            nlayers,
            in_dim,
            bottleneck_dim,
            hidden_dim=hidden_dim,
            use_bn=use_bn,
            bias=mlp_bias,
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def _build_mlp(self, nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
        """
        构建MLP层
        注意：对于时间序列场景，使用 LayerNorm 而不是 BatchNorm1d
        BatchNorm1d 在 batch 维度归一化，对于时间序列可能不合理
        LayerNorm 在特征维度归一化，更适合时间序列任务
        """
        if nlayers == 1:
            return nn.Linear(in_dim, bottleneck_dim, bias=bias)
        else:
            layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
            if use_bn:
                # 使用 LayerNorm 替代 BatchNorm1d（时间序列场景更合理）
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
                if use_bn:
                    # 使用 LayerNorm 替代 BatchNorm1d
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
            return nn.Sequential(*layers)

    def init_weights(self) -> None:
        """初始化权重"""
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """权重初始化函数"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, no_last_layer=False, only_last_layer=False):
        """
        前向传播
        Args:
            x: 输入特征 [..., in_dim]
            no_last_layer: 如果True，不应用最后一层
            only_last_layer: 如果True，只应用最后一层（需要先通过MLP）
        """
        if not only_last_layer:
            x = self.mlp(x)
            eps = 1e-6 if x.dtype == torch.float16 else 1e-12
            x = F.normalize(x, dim=-1, p=2, eps=eps)
        if not no_last_layer:
            x = self.last_layer(x)
        return x


# 保持向后兼容
ProtoHead = DINOHead

# ==========================================
# 2. Backbone (Updated)
# ==========================================

class Backbone(nn.Module):
    def __init__(self, configs):
        super(Backbone, self).__init__()
        self.seq_len = configs.seq_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.n_heads = configs.n_heads
        self.dropout = configs.dropout
        self.e_layers = configs.e_layers
        self.d_layers = configs.d_layers if hasattr(configs, 'd_layers') else 4
        self.num_patches = math.ceil((self.seq_len - self.patch_len + self.stride) / self.stride)
        
        # Storage tokens (register tokens) 数量，默认4个（参考dinov3配置）
        self.n_storage_tokens = getattr(configs, 'n_storage_tokens', 2)
        self.n_cls_tokens = max(1, int(getattr(configs, 'n_cls_tokens', 1)))
        self.imputator_segment_stride = getattr(configs, 'imputator_segment_stride', 244)

        # 1. Embedding
        # 主数据embedding：先patchify再embedding（模仿DINOv3的Conv2d方式）
        # patchify后：[B, T, C] -> [B, N, patch_len*C]
        # 然后对每个patch做embedding：[B, N, patch_len*C] -> [B, N, D]
        patch_dim = self.patch_len * self.enc_in
        self.embedding = nn.Linear(patch_dim, self.d_model)
        
        # Time mark和missing mask的独立embedding
        # time_mark: [B, T, 2] -> patchify -> [B, N, patch_len*2]
        # missing_mask: [B, T] -> patchify -> [B, N, patch_len*1]
        self.time_mark_embedding = nn.Linear(self.patch_len * 2, self.d_model)
        self.missing_mask_embedding = nn.Linear(self.patch_len * 1, self.d_model)
        
        self.position_encoding = PositionalEncoding(self.d_model, self.num_patches)
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        
        # CLS tokens：[1, K, D]，K>1 时在块后做 Qwen 式 attn-output sigmoid 门控加权融合再进 DINO head
        self.cls_token = nn.Parameter(torch.zeros(1, self.n_cls_tokens, self.d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.n_cls_tokens > 1:
            self.cls_attn_output_gate = nn.Linear(self.d_model, 1, bias=True)
        else:
            self.cls_attn_output_gate = None
        
        # Storage tokens (register tokens) - 在CLS token之后，patch tokens之前
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, self.n_storage_tokens, self.d_model))
            nn.init.normal_(self.storage_tokens, std=0.02)
        # 2. Encoder
        _attn_ckpt = not getattr(configs, 'no_attn_checkpoint', False)
        self.encoder = nn.ModuleList([
            AttentionBlock(
                self.d_model, self.n_heads, configs.d_ff // self.d_model, self.dropout,
                use_checkpoint=_attn_ckpt,
            )
            for l in range(self.e_layers)
        ])
        
        # [DINOv2 技巧] 独立 LayerNorm
        self.norm_student = nn.LayerNorm(self.d_model)
        self.norm_teacher = nn.LayerNorm(self.d_model)

        # 3. Decoder
        self.decoder = nn.ModuleList([
            AttentionBlock(
                self.d_model, self.n_heads, configs.d_ff // self.d_model, self.dropout,
                use_checkpoint=_attn_ckpt,
            )
            for l in range(self.d_layers)
        ])
        self.decoder_norm = nn.LayerNorm(self.d_model)
        # [新增] Decoder 专用位置编码 (提升重建清晰度)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.d_model))
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        # 4. Heads
        # # Latent Predictor
        # self.predictor = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.GELU(), nn.Linear(self.d_model, self.d_model))
        
        # DINO Head (用于CLS token) - 根据dinov3配置
        dino_head_n_prototypes = getattr(configs, 'dino_head_n_prototypes', 512)
        dino_head_hidden_dim = getattr(configs, 'dino_head_hidden_dim', 128)
        dino_head_bottleneck_dim = getattr(configs, 'dino_head_bottleneck_dim', 64)
        dino_head_nlayers = getattr(configs, 'dino_head_nlayers', 1)
        self.dino_head = DINOHead(
            in_dim=self.d_model,
            out_dim=dino_head_n_prototypes,
            hidden_dim=dino_head_hidden_dim,
            bottleneck_dim=dino_head_bottleneck_dim,
            nlayers=dino_head_nlayers,
        )
        
        # iBOT Head (用于Patch tokens) - 独立的head
        ibot_head_n_prototypes = getattr(configs, 'ibot_head_n_prototypes', 512)
        ibot_head_hidden_dim = getattr(configs, 'ibot_head_hidden_dim', 128)
        ibot_head_bottleneck_dim = getattr(configs, 'ibot_head_bottleneck_dim', 64)
        ibot_head_nlayers = getattr(configs, 'ibot_head_nlayers', 1)
        self.ibot_head = DINOHead(
            in_dim=self.d_model,
            out_dim=ibot_head_n_prototypes,
            hidden_dim=ibot_head_hidden_dim,
            bottleneck_dim=ibot_head_bottleneck_dim,
            nlayers=ibot_head_nlayers,
        )

        _cap = getattr(configs, 'fft_freq_bins_cap', None)
        if _cap is not None and int(_cap) > 0:
            self.fft_num_freq_bins = int(_cap)
        else:
            self.fft_num_freq_bins = self.num_patches // 2 + 1
        fft_head_n_prototypes = getattr(configs, 'fft_head_n_prototypes', 256)
        fft_head_hidden_dim = getattr(configs, 'fft_head_hidden_dim', 128)
        fft_head_bottleneck_dim = getattr(configs, 'fft_head_bottleneck_dim', 64)
        fft_head_nlayers = getattr(configs, 'fft_head_nlayers', 3)
        _ffm = getattr(configs, 'fft_feature_mode', 'mag')
        self.fft_feature_mode = _ffm if _ffm in ('mag', 're_im') else 'mag'
        self.use_fft_freq_pos_embed = bool(getattr(configs, 'use_fft_freq_pos_embed', 1))
        if self.fft_feature_mode == 're_im':
            self.fft_reim_proj = nn.Linear(2 * self.d_model, self.d_model)
        else:
            self.fft_reim_proj = None
        self.fft_bin_norm = nn.LayerNorm(self.d_model)
        if self.use_fft_freq_pos_embed:
            self.fft_freq_pos_embed = nn.Parameter(torch.zeros(1, self.fft_num_freq_bins, self.d_model))
            nn.init.trunc_normal_(self.fft_freq_pos_embed, std=0.02)
        else:
            self.register_parameter('fft_freq_pos_embed', None)
        self.fft_head = DINOHead(
            in_dim=self.d_model,
            out_dim=fft_head_n_prototypes,
            hidden_dim=fft_head_hidden_dim,
            bottleneck_dim=fft_head_bottleneck_dim,
            nlayers=fft_head_nlayers,
        )

        # 保持向后兼容
        self.proj_head = self.dino_head

        # Pixel Decoder
        self.pixel_decoder = nn.Linear(self.d_model, self.patch_len * self.c_out)

    def _fuse_cls_tokens(self, z_seq):
        """
        z_seq: LayerNorm 后全序列 [B, K+R+N, D]
        返回 z_cls [B, D]（DINO / z_global）、z_cls_bank [B, K, D]
        K>1：与 Qwen headwise gated attention 一致，对每路输出做 out * sigmoid(gate)（gate 由该路 D 维向量产生标量），再对 K 路 gated 向量求和得到单一 CLS 表征。
        """
        z_cls_bank = z_seq[:, : self.n_cls_tokens]
        if self.n_cls_tokens == 1 or self.cls_attn_output_gate is None:
            return z_cls_bank[:, 0], z_cls_bank
        # Qwen headwise: attn_out * sigmoid(gate)；此处每路 CLS 一个标量门，再对 gated 向量求和（不做归一）
        g = torch.sigmoid(self.cls_attn_output_gate(z_cls_bank))
        z_cls = (z_cls_bank * g).sum(dim=1)
        return z_cls, z_cls_bank

    def forward(self, x_enc, missing_mask, time_mark, mask_map=None, is_student=True):
        """
        增加 is_student 参数来控制 Norm 的选择
        """
        # 1. Patchify + Embedding（模仿DINOv3的Conv2d方式）
        # 先patchify，再对每个patch做embedding
        
        # 主数据：先patchify再embedding
        x_patches = patchify(x_enc, self.patch_len, self.stride)  # [B, N, patch_len*C]
        x_embed = self.embedding(x_patches)  # [B, N, D]
        B, N, D = x_embed.shape
        
        # Time mark和missing mask：先patchify再embedding
        if time_mark is not None:
            time_mark_patches = patchify(time_mark, self.patch_len, self.stride)  # [B, N, patch_len*2]
            time_mark_embed = self.time_mark_embedding(time_mark_patches)  # [B, N, D]
        else:
            time_mark_embed = torch.zeros(B, N, D, device=x_enc.device, dtype=x_embed.dtype)
        
        missing_mask_patches = patchify(missing_mask.unsqueeze(-1).float(), self.patch_len, self.stride)  # [B, N, patch_len*1]
        missing_mask_embed = self.missing_mask_embedding(missing_mask_patches)  # [B, N, D]
        
        # 相加
        x_input = x_embed + time_mark_embed + missing_mask_embed

        # 2. Masking
        if mask_map is not None:
            mask_expand = mask_map.unsqueeze(-1).expand(-1, -1, D).type_as(x_embed)
            mask_tokens = self.mask_token.expand(B, N, -1)
            x_input = x_input * (1 - mask_expand) + mask_tokens * mask_expand
        

        # Position encoding：需要处理不同长度的输入（global: 122, local: 16）
        # PositionalEncoding通常返回[1, max_len, D]或[B, N, D]
        # 我们需要确保维度匹配
        pos_embed = self.position_encoding(x_input)
        # 如果pos_embed的长度大于N，截取前N个
        if pos_embed.shape[1] > N:
            pos_embed = pos_embed[:, :N, :]
        # 如果pos_embed的长度小于N，需要padding（但这种情况不应该发生）
        elif pos_embed.shape[1] < N:
            # padding with zeros
            pad_len = N - pos_embed.shape[1]
            pos_embed = torch.cat([pos_embed, torch.zeros(B, pad_len, D, device=pos_embed.device, dtype=pos_embed.dtype)], dim=1)
        x_input = x_input + pos_embed
        
        # 准备tokens: [CLS, Storage Tokens, Patch Tokens]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens.expand(B, -1, -1)
            x_input = torch.cat([cls_tokens, storage_tokens, x_input], dim=1)
        else:
            x_input = torch.cat([cls_tokens, x_input], dim=1)
        # 3. Encoder
        # 注意：为了保持训练和验证时行为一致（正确计算loss），
        # 这里统一不收集attention（loss计算不需要attention）
        for layer in self.encoder:
            x_input, _ = layer(x_input, is_causal=False, return_attn=False)

        # 4. Independent Norm & Output
        if is_student:
            z = self.norm_student(x_input)
        else:
            z = self.norm_teacher(x_input)
        
        # 分离 tokens: CLS(融合), Storage, Patch
        z_cls, _ = self._fuse_cls_tokens(z)
        off = self.n_cls_tokens
        if self.n_storage_tokens > 0:
            z_storage = z[:, off : off + self.n_storage_tokens]
            z_patch = z[:, off + self.n_storage_tokens :]
        else:
            z_storage = None
            z_patch = z[:, off:]

        # 5. Proto Logits (CLS & Patch) - 使用独立的head
        # 注意：iBOT head在Model层应用（只对masked patches），这里只返回pre-head features
        # 这样可以避免对所有patches都计算head（参考DINOv3的优化）
        if is_student:
            logits_global = self.dino_head(z_cls)  # DINO head用于CLS
            # iBOT head：在Model层对masked patches应用，这里不计算
            # 返回 None，表示需要在 Model 层根据 mask_indices_list 选择 masked patches 后再应用
            logits_patch = None
        else:
            with torch.no_grad():
                logits_global = self.dino_head(z_cls)
                # Teacher 分支跳过 iBOT head，避免重复计算；只返回预 head 的 patch 特征
                logits_patch = None
        
        # # Predictor (只用于 Student Latent 对齐，可选)
        # z_patch_pred = self.predictor(z_patch)

        # 6. Decoder (只用于 Reconstruction / Student)
        # 支持动态长度，decoder_pos_embed 按当前 N 截断/填充
        # local views（N 很小）不使用 decoder，因为 reconstruction 仅对 global views
        rec_patches = None
        if is_student and N > 0:
            if self.decoder_pos_embed.shape[1] >= N:
                dec_pos = self.decoder_pos_embed[:, :N, :]
            else:
                pad_len = N - self.decoder_pos_embed.shape[1]
                dec_pos = torch.cat(
                    [self.decoder_pos_embed, torch.zeros(1, pad_len, self.d_model, device=self.decoder_pos_embed.device, dtype=self.decoder_pos_embed.dtype)],
                    dim=1
                )
            z_dec = z_patch + dec_pos  # 重新注入位置信息
            for layer in self.decoder:
                z_dec, _ = layer(z_dec, is_causal=False, return_attn=False)
            z_dec = self.decoder_norm(z_dec)
            rec_patches = self.pixel_decoder(z_dec)

        return {
            'logits_global': logits_global,
            'logits_patch': logits_patch,
            'rec_patches': rec_patches,
            'z_cls': z_cls,
            'z_storage': z_storage,  # Storage tokens
            'z_global': z_cls,
            'z_patch_enc': z_patch,
            # 'all_attns': all_attns if output_attentions else None
        }

    def forward_fft_logits(self, z_patch):
        xc = torch.fft.rfft(z_patch.float(), dim=1, norm='ortho')
        fb = xc.shape[1]
        fmax = self.fft_num_freq_bins
        if fb < fmax:
            xc = F.pad(xc, (0, 0, 0, fmax - fb, 0, 0))
        elif fb > fmax:
            xc = xc[:, :fmax, :]
        if self.fft_feature_mode == 're_im' and self.fft_reim_proj is not None:
            h = self.fft_reim_proj(torch.cat([xc.real, xc.imag], dim=-1))
        else:
            h = torch.abs(xc)
        h = self.fft_bin_norm(h)
        if self.use_fft_freq_pos_embed and self.fft_freq_pos_embed is not None:
            h = h + self.fft_freq_pos_embed
        return self.fft_head(h)

    def encode(self, xEnc, missing_mask, time_mark, imputator=None, use_student_norm=False, output_attentions=False):
        """
        编码方法
        Args:
            xEnc: 输入数据 [B, T, C]
            missing_mask: 缺失mask [B, T]
            time_mark: 时间标记 [B, T, 2]
            imputator: 插值器（可选），如果提供则用于填充缺失值
            use_student_norm: 是否使用student的layernorm（默认False，使用teacher norm）
            output_attentions: 是否输出attention权重（用于可视化）
        """
        B = xEnc.shape[0]
        device = xEnc.device
        
        # 处理 imputator：与 _forward 一致；超长用滑动窗口重叠（见 imputator_sliding_window_overlap）
        if imputator is not None:
            missing_mask_orig = missing_mask.clone()
            x_clean_filled = xEnc.nan_to_num(0.0)
            T = xEnc.shape[1]
            max_imp_len = getattr(imputator, 'pred_len', 366)
            stride_imp = getattr(self, 'imputator_segment_stride', 244)

            with torch.no_grad():
                imp_device = next(imputator.parameters()).device
                imputed_out = imputator_sliding_window_overlap(
                    xEnc,
                    time_mark,
                    missing_mask_orig.bool(),
                    imputator,
                    window_len=max_imp_len,
                    stride=stride_imp,
                    device=device,
                    imp_device=imp_device,
                )

            mask_expanded = missing_mask_orig.unsqueeze(-1).float()
            xEnc = x_clean_filled * (1 - mask_expanded) + imputed_out * mask_expanded
            missing_mask = torch.zeros_like(missing_mask_orig)
        
        # 主数据：先patchify再embedding
        x_patches = patchify(xEnc, self.patch_len, self.stride)  # [B, N, patch_len*C]
        x_embed = self.embedding(x_patches)  # [B, N, D]
        N = x_embed.shape[1]
        
        # Time mark和missing mask：先patchify再embedding
        if time_mark is not None:
            time_mark_patches = patchify(time_mark, self.patch_len, self.stride)  # [B, N, patch_len*2]
            time_mark_embed = self.time_mark_embedding(time_mark_patches)  # [B, N, D]
        else:
            time_mark_embed = torch.zeros(B, N, self.d_model, device=xEnc.device, dtype=x_embed.dtype)
        
        missing_mask_patches = patchify(missing_mask.unsqueeze(-1).float(), self.patch_len, self.stride)  # [B, N, patch_len*1]
        missing_mask_embed = self.missing_mask_embedding(missing_mask_patches)  # [B, N, D]
        
        # 相加
        x_input_embed = x_embed + time_mark_embed + missing_mask_embed
        
        # Position encoding：处理不同长度的输入
        pos_embed = self.position_encoding(x_input_embed)
        if pos_embed.shape[1] > N:
            pos_embed = pos_embed[:, :N, :]
        elif pos_embed.shape[1] < N:
            pad_len = N - pos_embed.shape[1]
            pos_embed = torch.cat([pos_embed, torch.zeros(B, pad_len, self.d_model, device=pos_embed.device, dtype=pos_embed.dtype)], dim=1)
        x_input = x_input_embed + pos_embed
        cls_tokens = self.cls_token.expand(B, -1, -1)
        if self.n_storage_tokens > 0:
            storage_tokens_in = self.storage_tokens.expand(B, -1, -1)
            x_input = torch.cat([cls_tokens, storage_tokens_in, x_input], dim=1)
        else:
            x_input = torch.cat([cls_tokens, x_input], dim=1)
        
        # Encoder: 根据 output_attentions 决定是否收集 attention
        all_attns = []
        if output_attentions:
            for layer in self.encoder:
                x_input, attn = layer(x_input, is_causal=False, return_attn=True)
                all_attns.append(attn if attn is not None else None)
        else:
            for layer in self.encoder:
                x_input, _ = layer(x_input, is_causal=False, return_attn=False)
        
        # 根据 use_student_norm 选择使用哪个 norm
        if use_student_norm:
            x_out = self.norm_student(x_input)  # 使用 Student Norm
        else:
            x_out = self.norm_teacher(x_input)  # 使用 Teacher Norm
        
        cls_token_fused, _ = self._fuse_cls_tokens(x_out)
        off = self.n_cls_tokens
        if self.n_storage_tokens > 0:
            storage_tokens = x_out[:, off : off + self.n_storage_tokens]
            patch_tokens = x_out[:, off + self.n_storage_tokens :]
        else:
            storage_tokens = None
            patch_tokens = x_out[:, off:]
        
        result = {'cls_token': cls_token_fused, 'storage_tokens': storage_tokens, 'patch_tokens': patch_tokens}
        
        # 如果 output_attentions=True，添加 attention 和 cos 相似度信息
        if output_attentions:
            # 计算融合 CLS 与所有 token 的 cos 相似度
            z_cls_expanded = cls_token_fused.unsqueeze(1)  # [B, 1, D]
            cls_cos_sim_all = F.cosine_similarity(z_cls_expanded, x_out, dim=-1)  # [B, 1+R+N]
            cls_cos_sim_patch = F.cosine_similarity(z_cls_expanded, patch_tokens, dim=-1)  # [B, N]
            
            result.update({
                'all_attns': all_attns,
                'cls_cos_sim_all': cls_cos_sim_all,
                'cls_cos_sim_patch': cls_cos_sim_patch,
            })
        
        return result

    def visualize(self, xEnc, missing_mask, time_mark, imputator=None, use_student_norm=False):
        """
        可视化方法：返回每层的 attention 权重、CLS token 与所有 token 的 cos 相似度等信息
        
        Args:
            xEnc: 输入数据 [B, T, C]
            missing_mask: 缺失mask [B, T]
            time_mark: 时间标记 [B, T, 2]
            imputator: 插值器（可选），如果提供则用于填充缺失值
            use_student_norm: 是否使用student的layernorm（默认False，使用teacher norm）
        
        Returns:
            包含以下信息的字典：
            - all_attns: 每层的 attention 权重列表
            - cls_cos_sim_all: CLS token 与所有 token 的 cos 相似度 [B, 1+R+N]
            - cls_cos_sim_patch: CLS token 与 patch tokens 的 cos 相似度 [B, N]
            - cls_token: CLS token 特征 [B, D]
            - patch_tokens: Patch tokens 特征 [B, N, D]
            - storage_tokens: Storage tokens 特征 [B, R, D] 或 None
        """
        # 调用 encode 方法，设置 output_attentions=True
        return self.encode(xEnc, missing_mask, time_mark, imputator=imputator, use_student_norm=use_student_norm, output_attentions=True)

# ==========================================
# 3. Model (Updated with EMA & New Losses)
# ==========================================

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # 1. Student & Teacher Backbones
        self.backbone = Backbone(configs) # Student
        self.teacher = Backbone(configs)  # Teacher (EMA)
        
        # 初始化 Teacher 为 Student 的副本并冻结
        self.teacher.load_state_dict(self.backbone.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False
            
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.c_out = configs.c_out
        # 与 Backbone 中的 patchify/unfold 逻辑保持一致：
        # N = ceil((T - patch_len) / stride) + 1 （当 T > patch_len）
        self.num_patches = math.ceil((configs.seq_len - configs.patch_len + configs.stride) / configs.stride)
        self.local_view_patch_divisor = max(1, int(getattr(configs, "local_view_patch_divisor", 8)))
        self.num_local_patches = max(1, self.num_patches // self.local_view_patch_divisor)
        # 2. Loss 计算在外部 DINOCriteria 中进行，Model 只负责前向传播
        
        # 3. Weights（对齐DINOv3官方配置，用于外部 loss 计算）
        # 保留 base 值，便于在训练中根据 epoch 做调度
        self.lambda_recon = configs.lambda_recon if hasattr(configs, 'lambda_recon') else 0.5
        self.lambda_recon_base = self.lambda_recon
        # FFT recon 仅在前若干 epoch 使用并线性衰减到 0（0 表示不使用调度）
        self.fft_recon_warm_epochs = getattr(configs, 'fft_recon_warm_epochs', 0)
        self.lambda_fft_align = getattr(configs, 'lambda_fft_align', 0.05)  # FFT 对齐正则，建议保持较小避免与 iBOT 冲突
        self.lambda_cls_proto = configs.lambda_cls_proto if hasattr(configs, 'lambda_cls_proto') else 1.0
        self.lambda_patch_proto = configs.lambda_patch_proto if hasattr(configs, 'lambda_patch_proto') else 1.0  # 关键修复：从0.2增加到1.0
        self.lambda_koleo = configs.lambda_koleo if hasattr(configs, 'lambda_koleo') else 0.2  # 从0.1增加到0.2（对齐DINOv3实际权重）
        self.lambda_temporal = configs.lambda_temporal if hasattr(configs, 'lambda_temporal') else 0.2  # 从0.3降到0.2（平衡）
        # DINOv3-style patch masking 中 block 部分的比例（其余为 random），允许外部控制
        self.block_mask_ratio = getattr(configs, 'block_mask_ratio', 0.8)
        self.mask_sample_probability = getattr(configs, 'mask_sample_probability', 0.5)
        # 对 imputator 填补位置的 patch loss 做降权（1.0 表示不降权）
        self.imputed_patch_weight = getattr(configs, 'imputed_patch_weight', 1.0)
        # Raw–Imputed CLS 一致性：同序列 raw / imputed 两视角 CLS 对齐权重（0 表示关闭）
        self.lambda_cls_cons = getattr(configs, 'lambda_cls_cons', 0.05)
        self.lambda_fft_proto = getattr(configs, 'lambda_fft_proto', 0.0)
        self.lambda_fft_proto_base = self.lambda_fft_proto
        self.fft_proto_warm_epochs = getattr(configs, 'fft_proto_warm_epochs', 0)
        self.fft_align_min_valid_ratio = float(
            getattr(configs, "fft_align_min_valid_ratio", 0.25)
        )

        # Teacher temperature for sinkhorn-knopp
        self.teacher_temp = getattr(configs, 'teacher_temp', 0.07)
        
        # 4. 训练状态跟踪
        self.step_counter = 0
        self.log_interval = 10  # 备用步数日志间隔（目前未使用）
        self.last_log_time = time.time()
        self.log_interval_seconds = 300  # 两次日志之间至少间隔 5 分钟（300秒）
        # 记录当前 epoch（用于 lambda_recon 调度），默认 0
        self._current_epoch = 0
        
        # 5. 验证相关阈值
        self.valid_patch_threshold = 0.5  # 用于判断 patch 是否有效的阈值
        self.ibot_patch_reliable_mode = getattr(
            configs, 'ibot_patch_reliable_mode', 'filter'
        )

        # 6. 渐进学习策略配置
        # 'fast': 前10%只用1-3年，之后1-6年全范围（推荐，适合余弦退火）
        # 'balanced': 0-15%: 1-3年, 15-35%: 1-4年, 35-55%: 1-5年, 55-100%: 1-6年
        # 'conservative': 0-20%: 1-3年, 20-40%: 1-4年, 40-60%: 1-5年, 60-100%: 1-6年
        # 'none': 不使用渐进学习，直接1-6年全范围
        self.curriculum_strategy = getattr(configs, 'curriculum_strategy', 'fast')
        
        # 7. Imputator 使用策略
        # 'full'         : Teacher 全用 imputator，Student 一半 imputator 一半 raw，
        #                  Teacher 侧 missing_mask 置为全 0，所有 patch 视为可靠，重建用 perfect target
        # 'recon_only'   : Teacher / Student 视图都不用 imputator（都看 raw），
        #                  missing_mask / patch 过滤保持原始逻辑，但重建目标仍用 imputator 生成的 perfect target
        # 'mixed_teacher': Student 仍是 50% imputator 视图，Teacher 也 50% imputator 视图，
        #                  missing_mask / patch 过滤保持原始逻辑，重建目标用 perfect target
        # 'woMask'       : Teacher 全用 imputator，Student 一半 imputator 一半 raw，
        #                  且所有视图传入 backbone 的 missing_mask 视为全 0（不使用 missing mask embedding）
        # 'wMask'        : Teacher 全用 imputator，Student 都不用 imputator（都用 raw），
        #                  Teacher / Student 都保留原始 missing_mask（基于物理缺失）
        self.imputator_mode = getattr(configs, 'imputator_mode', 'full')
        self.imputator_segment_stride = getattr(configs, 'imputator_segment_stride', 244)
        


    def _update_teacher(self, m=None):
        """ EMA Update: Teacher = m * Teacher + (1-m) * Student """
        # 使用DINOv3的默认momentum（0.992），而不是0.996
        # 0.996更新太慢，可能导致teacher跟不上student，loss不收敛
        if m is None:
            m = 0.992  # DINOv3默认值

        # 关键修复：
        #  - Student 只使用 backbone.norm_student，Teacher 只使用 teacher.norm_teacher
        #  - 之前的 zip 会让 teacher.norm_teacher 跟踪 student.norm_teacher（训练中几乎不用），
        #    导致 Teacher 的归一化层没有正确跟随 Student。
        #  - 这里显式将 student.norm_student 的权重 EMA 到 teacher.norm_teacher，
        #    其他参数仍按同名映射。
        with torch.no_grad():
            teacher_params = dict(self.teacher.named_parameters())
            for name_s, param_s in self.backbone.named_parameters():
                # 跳过 decoder 分支
                if "decoder" in name_s:
                    continue

                # 默认：同名参数
                target_name = name_s
                # 特例：norm_student -> norm_teacher
                if "norm_student" in name_s:
                    target_name = name_s.replace("norm_student", "norm_teacher")

                param_t = teacher_params.get(target_name, None)
                if param_t is None:
                    continue

                param_t.data.mul_(m).add_((1 - m) * param_s.data)

    def _forward(self, x_enc, time_mark, mask_rate_v1, mask_rate_v2=None, mode='train', imputator=None):
        """
        Args:
            x_enc: 输入数据
            time_mark: 时间标记
            mask_rate_v1: mask率1
            mask_rate_v2: mask率2
            mode: 模式
            imputator: 插值器（可选）
        """
        B, T, C = x_enc.shape
        device = x_enc.device
        
        # 1. 数据准备 (Data Prep)
        # 【关键修改】：保存原始的物理缺失 mask，用于那些不经过 imputator 的 raw student views
        missing_mask_orig = torch.isnan(x_enc).any(dim=-1)
        
        # 初始化逻辑 missing_mask，默认等于原始 mask
        # 如果后续用了 imputator，这个 mask 会被置为全 0
        missing_mask = missing_mask_orig.clone()  # clone 以确保安全（后续可能会修改）
        
        # 先对完整序列填充，后续再做裁剪
        x_clean_filled = x_enc.nan_to_num(0.0)
        
        # 2. Imputator Logic: pred_len 窗口 + 步长滑动重叠，缺失处多窗预测取均值（见 imputator_sliding_window_overlap）
        x_target_perfect = None
        imputator_mode = getattr(self, 'imputator_mode', 'full')
        if imputator is not None:
            with torch.no_grad():
                imp_device = next(imputator.parameters()).device
                max_imp_len = getattr(imputator, 'pred_len', 366)
                stride_imp = getattr(self, 'imputator_segment_stride', 244)
                imputed_out = imputator_sliding_window_overlap(
                    x_enc,
                    time_mark,
                    missing_mask_orig,
                    imputator,
                    window_len=max_imp_len,
                    stride=stride_imp,
                    device=device,
                    imp_device=imp_device,
                )
            
            # 【关键修改】：合成 perfect target 时使用 missing_mask_orig (指示哪里需要补全)
            mask_expanded = missing_mask_orig.unsqueeze(-1).float()
            x_target_perfect = x_clean_filled * (1 - mask_expanded) + imputed_out * mask_expanded
            
            # 【模式1：full】
            # 认为 imputator 生成的目标在所有位置都可靠：
            # - downstream 视角中 missing_mask 视为全有效
            # - patch 过滤逻辑中也不再区分缺失严重的 patch
            #
            # 【模式2：recon_only / wMask】
            # 保留 missing_mask = missing_mask_orig，让 patch 过滤等逻辑仍然基于原始观测质量。
            #
            # 【模式3：mixed_teacher】
            # 在 missing_mask 中对填补位置打 0.5，表示“半缺失半有效”，用于后续 patch 可靠性估计；
            # 对于 Teacher/Student 视图的 missing_mask 传递，由各自逻辑决定使用 missing_mask 或 missing_mask_orig。
            if imputator_mode in ['full', 'woMask']:
                missing_mask = torch.zeros_like(missing_mask_orig)
            elif imputator_mode == 'mixed_teacher':
                missing_mask = torch.where(
                    missing_mask_orig,
                    torch.full_like(missing_mask_orig, 0.5, dtype=torch.float32),
                    torch.zeros_like(missing_mask_orig, dtype=torch.float32),
                )
        else:
            # 没有 imputator，主 missing_mask 保持为 missing_mask_orig
            # x_target_perfect = create_smooth_target(x_clean_filled, missing_mask_orig, kernel_size=5)
            x_target_perfect = x_clean_filled

        # 3. 动态 / 固定 序列长度采样
        # 如果 curriculum_strategy == 'fixed'，则完全使用原始长度 T，不做任何裁剪。
        curriculum_strategy = getattr(self, 'curriculum_strategy', 'fast')
        if curriculum_strategy == 'fixed':
            chosen_len = T
            start_idx = 0
            end_idx = T
        else:
            # 动态序列长度采样（1年/2年/3年/4年/5年/6年），在 imputator 处理完成后进行裁剪
            #
            # 【方案1：渐进学习策略（当前实现）】
            # 训练过程中逐步增加长度范围，提高训练稳定性
            # 优点：简单易实现，训练稳定，loss下降快
            # 缺点：需要知道训练总步数
            #
            # 【方案2：Batch内分组（备选）】
            # 将batch分成多个组，每组使用不同长度，然后分别处理
            # 优点：每个batch都能看到多种长度，无需知道训练步数
            # 缺点：实现复杂，需要处理变长序列或分组forward
            # 实现思路：
            #   - 将batch分成3-4组
            #   - 每组独立选择长度（1-3年、2-4年、3-5年、4-6年）
            #   - 分别forward，loss加权平均
            #
            # 【方案3：每个样本独立长度（备选）】
            # 每个样本独立选择长度，使用padding到最大长度
            # 优点：最灵活
            # 缺点：padding浪费计算，实现复杂
            #
            # 假设 T 是6年长度，按等分定义1-6年
            base_year_len = max(self.patch_len, T // 6)
            
            # 获取当前训练进度（iteration/epoch）
            # 可以通过 self.step_counter 或外部传入的 iteration 参数控制
            current_iteration = getattr(self, '_current_iteration', 0)
            total_iterations = getattr(self, '_total_iterations', 100000)  # 默认10万步
            
            # 【优化后的渐进学习策略】：加快进度，让长序列在较高学习率时就开始训练
            # 考虑到余弦退火，学习率在前期较高，后期较低
            # 因此加快渐进学习进度，让模型在较高学习率时就能接触到长序列
            progress = current_iteration / max(total_iterations, 1)
            
            if curriculum_strategy == 'none':
                # 不使用渐进学习，直接1-6年全范围
                max_year = 6
            elif curriculum_strategy == 'fast':
                # 快速策略：前10%只用1-3年，之后就是1-6年全范围（推荐，适合余弦退火）
                if progress < 0.1:
                    max_year = 3
                else:
                    max_year = 6
            elif curriculum_strategy == 'balanced':
                # 平衡策略：0-15%: 1-3年, 15-35%: 1-4年, 35-55%: 1-5年, 55-100%: 1-6年
                if progress < 0.15:
                    max_year = 3
                elif progress < 0.35:
                    max_year = 4
                elif progress < 0.55:
                    max_year = 5
                else:
                    max_year = 6
            elif curriculum_strategy == 'conservative':
                # 保守策略：0-20%: 1-3年, 20-40%: 1-4年, 40-60%: 1-5年, 60-100%: 1-6年
                if progress < 0.2:
                    max_year = 3
                elif progress < 0.4:
                    max_year = 4
                elif progress < 0.6:
                    max_year = 5
                else:
                    max_year = 6
            else:
                # 其它字符串（例如历史上的 'mixed_batch'）默认使用快速策略
                if progress < 0.1:
                    max_year = 3
                else:
                    max_year = 6
            
            # 生成当前阶段允许的长度选择
            length_choices = sorted(set([
                base_year_len,               # 1年
                min(2 * base_year_len, T),   # 2年
                min(3 * base_year_len, T),   # 3年
            ]))
            if max_year >= 4:
                length_choices.append(min(4 * base_year_len, T))  # 4年
            if max_year >= 5:
                length_choices.append(min(5 * base_year_len, T))  # 5年
            if max_year >= 6:
                length_choices.append(T)  # 6年（完整长度）
            
            length_choices = torch.tensor(length_choices, device=device)
            # 随机选择一个长度
            chosen_len = int(length_choices[torch.randint(len(length_choices), (1,), device=device)])
            # 在原序列上随机起始位置裁剪到 chosen_len
            start_idx = 0
            if chosen_len < T:
                max_start = max(T - chosen_len, 0)
                start_idx = int(torch.randint(max_start + 1, (1,), device=device))
            end_idx = start_idx + chosen_len
        
        # === 时间长度裁剪（在得到 perfect/clean 后执行） ===
        if chosen_len < T:
            slice_fn = lambda t: t[:, start_idx:end_idx] if t is not None else None
            x_target_perfect = slice_fn(x_target_perfect)
            x_clean_filled = slice_fn(x_clean_filled)
            missing_mask = slice_fn(missing_mask)
            missing_mask_orig = slice_fn(missing_mask_orig)
            time_mark = slice_fn(time_mark)
            T = chosen_len  # 更新当前序列长度

        # 动态当前 patch 数（后续 mask 与 decoder 需要用到）
        num_patches_cur = math.ceil((T - self.patch_len + self.stride) / self.stride)
        num_local_patches_cur = max(1, num_patches_cur // self.local_view_patch_divisor)

        # 根据裁剪后的长度重算 is_target_reliable 与 is_patch_reliable_for_ibot
        # - full 模式下：认为所有 patch 目标都可靠（imputator 提供了完全填补），不过滤
        # - 其它情况：ibot_patch_reliable_mode 为 filter（默认）或 keep_all
        # 同时预先计算每个 patch 是否包含 imputator 填补位置（用于后续 patch loss 降权）
        orig_valid = (~missing_mask_orig).float().unsqueeze(-1)
        orig_valid_map_flat = patchify(orig_valid, self.patch_len, self.stride)  # [B, N_cur, patch_len]
        patch_valid_ratio_orig = orig_valid_map_flat.mean(dim=-1)  # [B, N_cur]，1 表示该 patch 全为原始观测
        has_imputed_patch = patch_valid_ratio_orig < 1.0  # True 表示该 patch 至少包含 1 个原始缺失点

        if imputator is not None and imputator_mode == 'full':
            is_target_reliable = torch.ones(B, num_patches_cur, device=device, dtype=torch.bool)
            is_patch_reliable_for_ibot = torch.ones(B, num_patches_cur, device=device, dtype=torch.bool)
        else:
            # is_target_reliable 使用 missing_mask（mixed 下为 0.5）
            missing_mask_float = missing_mask.float()
            is_valid_pixel = 1.0 - missing_mask_float
            valid_map_flat = patchify(is_valid_pixel.unsqueeze(-1), self.patch_len, self.stride)
            is_target_reliable = valid_map_flat.mean(dim=-1) > self.valid_patch_threshold
            if getattr(self, 'ibot_patch_reliable_mode', 'filter') == 'keep_all':
                is_patch_reliable_for_ibot = torch.ones(
                    B, num_patches_cur, device=device, dtype=torch.bool
                )
            else:
                is_patch_reliable_for_ibot = patch_valid_ratio_orig > 0

        # 2. 构建多视角 (Multi-view) - 优化版本：批量生成
        
        # 使用完整的views配置（训练和验证保持一致，确保loss有效性）
        n_teacher_views = 2
        n_global_student = 2
        n_local_student = 8

        # 根据 imputator_mode 决定 Student 全局视图是否使用 imputator
        # - full         : [True, False]（一半用 imputator，一半用 raw）
        # - mixed_teacher: [True, False]（一半用 imputator，一半用 raw）
        # - recon_only   : [False, False]（都用 raw）
        # - woMask       : [True, False]（一半用 imputator，一半用 raw，但 missing_mask 统一视为 0）
        # - wMask        : [False, False]（都用 raw，保留原始 missing_mask）
        if imputator is not None and imputator_mode in ['full', 'mixed_teacher', 'woMask']:
            x_student_global_use_imputator = [True, False]
        else:
            x_student_global_use_imputator = [False, False]
        
        # Teacher Views: weak augmentation views
        # - full         : Teacher 视图全部使用 imputator；Student 一半 imp 一半 raw
        # - recon_only   : Teacher 视图全部使用 raw（x_clean_filled）
        # - mixed_teacher: Teacher 视图全部使用 imputator；Student 一半 imp 一半 raw
        teacher_sources = []
        if imputator is not None:
            if imputator_mode == 'full':
                teacher_sources = ['imp', 'imp']
            elif imputator_mode == 'recon_only':
                teacher_sources = ['raw', 'raw']
            else:  # 'mixed_teacher' 以及其它字符串默认视为 mixed
                teacher_sources = ['imp', 'imp']
        else:
            teacher_sources = ['raw', 'raw']

        x_teacher_bases = []
        teacher_masks_list = []
        for src in teacher_sources:
            if src == 'imp':
                x_teacher_bases.append(x_target_perfect)
                # 视为已经无缺失
                teacher_masks_list.append(missing_mask)
            else:
                x_teacher_bases.append(x_clean_filled)
                teacher_masks_list.append(missing_mask_orig)

        x_teacher_base = torch.cat(x_teacher_bases, dim=0)  # [n_teacher*B, T, C]
        x_teacher_views_batch = get_teacher_input(x_teacher_base, 'weak', is_train=True)
        
        # Student Global Views: 强增强，带随机patch mask
        x_global_bases = []
        for use_imp in x_student_global_use_imputator:
            x_base = x_target_perfect if use_imp else x_clean_filled
            x_global_bases.append(x_base)
        x_global_bases_batch = torch.cat(x_global_bases, dim=0)  # [n_global*B, T, C]
        x_student_global_views_batch = get_student_input(x_global_bases_batch, 'strong', is_train=True)
        # 优化：避免列表推导式，直接使用 batch（如果需要单独访问，可以用 view）
        # x_student_global_views = [x_student_global_views_batch[i*B:(i+1)*B] for i in range(n_global_student)]  # 已不再使用，保留注释
        
        # 3. Teacher Forward (EMA) - 批处理所有teacher views（需要在生成 local views 之前完成）
        teacher_was_training = self.teacher.training
        self.teacher.eval()
        
        teacher_temp = getattr(self, '_current_teacher_temp', getattr(self, 'teacher_temp', 0.07))
        
        # Teacher 使用相应的 missing_mask（对于基于 imputator 的视图，mask 可为全 0）
        missing_mask_teacher = torch.cat(teacher_masks_list, dim=0).float()  # [n_teacher*B, T]
        time_mark_teacher = time_mark.repeat(n_teacher_views, 1, 1) if time_mark is not None else None
        
        with torch.no_grad():
            out_teacher_batch = self.teacher(
                x_teacher_views_batch, missing_mask_teacher, time_mark_teacher, 
                mask_map=None, is_student=False
            )
            
            t_logits_global_raw = out_teacher_batch['logits_global'].view(n_teacher_views, B, -1).detach()
            # 注意：Teacher 的 ibot_head 现在不在 backbone 中计算，只返回 normalized features
            t_z_cls_batch = out_teacher_batch['z_cls'].view(n_teacher_views, B, -1).detach()
            t_z_patch_batch = out_teacher_batch['z_patch_enc'].view(n_teacher_views, B, -1, out_teacher_batch['z_patch_enc'].shape[-1]).detach()
        
        t_logits_global = t_logits_global_raw
        # t_logits_patch 不再在这里计算，将在后面根据 mask_indices_list 选择 masked patches 后应用 ibot_head
        
        self.teacher.train(teacher_was_training)

        # 4. Student Local Views: 从 x_target_perfect 中生成，然后做 weak augmentation
        local_view_types = ['crop'] * 6 + ['random'] * 2
        x_student_local_views = []
        local_time_mark_list = []  # 保存每个 local view 对应的 time_mark
        local_missing_mask_list = []  # 保存每个 local view 对应的 missing_mask
        
        if n_local_student > 0:
            local_token_num = num_local_patches_cur
            local_time_steps = local_token_num * self.patch_len
            
            # 【关键修改】：从 x_target_perfect 中生成 local views，然后做 weak augmentation
            # x_target_perfect 是原始数据（已填充/插值，无NaN），与 teacher views 的输入一致
            # local views 独立生成，然后做 weak augmentation，保持与 teacher 的一致性
            crop_indices = [i for i, t in enumerate(local_view_types) if t == 'crop']
            random_indices = [i for i, t in enumerate(local_view_types) if t == 'random']
            
            # 从 x_target_perfect 中生成 local views（需要扩展到 n_local_student * B）
            # x_target_perfect: [B, T, C]，需要 repeat 到足够数量
            base_for_local = x_target_perfect.repeat(n_local_student, 1, 1)  # [n_local_student*B, T, C]
            
            # 准备 time_mark 和 missing_mask（用于后续 crop/random）
            time_mark_for_local = time_mark.repeat(n_local_student, 1, 1) if time_mark is not None else None  # [n_local_student*B, T, 2] 或 None
            missing_mask_for_local = missing_mask.repeat(n_local_student, 1)  # [n_local_student*B, T]
            
            if len(crop_indices) > 0:
                # 从 x_target_perfect 中 crop
                # 需要为每个 crop view 生成独立的 crop（总共 len(crop_indices) * B 个样本）
                n_crop_samples = len(crop_indices) * B
                x_crop_base = base_for_local[:n_crop_samples]  # [n_crop_samples, T, C]
                x_crop_batch, crop_start_indices = generate_local_view_crop(x_crop_base, local_time_steps, device)
                
                # 【关键修复】：同时 crop time_mark 和 missing_mask
                # 优化：复用 batch_indices 和 time_indices，避免重复计算
                batch_indices = torch.arange(n_crop_samples, device=device).unsqueeze(1)
                time_indices = crop_start_indices.unsqueeze(1) + torch.arange(local_time_steps, device=device).unsqueeze(0)
                
                if time_mark_for_local is not None:
                    time_mark_crop = time_mark_for_local[:n_crop_samples][batch_indices, time_indices]  # [n_crop_samples, local_time_steps, 2]
                else:
                    time_mark_crop = None
                
                # missing_mask：从 x_target_perfect 生成，missing_mask 可能不全为0（取决于是否有 imputator）
                missing_mask_crop = missing_mask_for_local[:n_crop_samples][batch_indices, time_indices]  # [n_crop_samples, local_time_steps]
                
                # 【关键修改】：对 crop 后的 local views 应用 weak augmentation
                x_crop_aug = get_student_input(x_crop_batch, 'weak_local', is_train=True)
                
                # 分配 crop views
                for idx, crop_idx in enumerate(crop_indices):
                    x_student_local_views.append((crop_idx, x_crop_aug[idx*B:(idx+1)*B]))
                    # 保存对应的 time_mark 和 missing_mask
                    if time_mark_crop is not None:
                        local_time_mark_list.append(time_mark_crop[idx*B:(idx+1)*B])
                    else:
                        local_time_mark_list.append(None)
                    local_missing_mask_list.append(missing_mask_crop[idx*B:(idx+1)*B])
            
            if len(random_indices) > 0:
                # 从 x_target_perfect 中随机采样
                # 需要为每个 random view 生成独立的采样（总共 len(random_indices) * B 个样本）
                n_random_samples = len(random_indices) * B
                x_random_base = base_for_local[len(crop_indices)*B:len(crop_indices)*B+n_random_samples]  # [n_random_samples, T, C]
                x_random_patches = patchify(x_random_base, self.patch_len, self.stride)
                x_random_patches_sampled, token_indices = generate_local_view_random_sample(x_random_patches, local_token_num, device)
                x_random_local = unpatchify(x_random_patches_sampled, local_time_steps, self.patch_len, self.c_out)
                
                # 【关键修复】：time_mark / missing_mask 必须与 x_random_local 的构造严格一致
                # x_random_local 是通过 patchify -> 采样 token -> unpatchify 得到的，因此这里也对 time_mark/missing_mask
                # 走相同的 patchify -> 采样 token -> unpatchify 流程，避免“中心点近似索引”带来的错位。
                start = len(crop_indices) * B
                end = start + n_random_samples
                time_mark_random_base = time_mark_for_local[start:end] if time_mark_for_local is not None else None
                missing_mask_random_base = missing_mask_for_local[start:end]

                # 使用与 x_random_local 完全相同的一组 token_indices 来选择 time_mark/missing_mask 对应的 token，
                # 避免再次随机采样导致的错位。
                batch_indices = torch.arange(n_random_samples, device=device).unsqueeze(1)

                if time_mark_random_base is not None:
                    # time_mark: [n_random_samples, T, 2] -> patchify -> [n_random_samples, N, patch_len*2]
                    tm_patches = patchify(time_mark_random_base, self.patch_len, self.stride)
                    tm_patches_sampled = tm_patches[batch_indices, token_indices]  # [n_random_samples, local_token_num, patch_len*2]
                    time_mark_random = unpatchify(tm_patches_sampled, local_time_steps, self.patch_len, 2)  # [n_random_samples, local_time_steps, 2]
                else:
                    time_mark_random = None

                # missing_mask: [n_random_samples, T] -> [n_random_samples, T, 1] -> patchify -> [n_random_samples, N, patch_len]
                mm_patches = patchify(missing_mask_random_base.unsqueeze(-1).float(), self.patch_len, self.stride)
                mm_patches_sampled = mm_patches[batch_indices, token_indices]  # [n_random_samples, local_token_num, patch_len]
                missing_mask_random = unpatchify(mm_patches_sampled, local_time_steps, self.patch_len, 1).squeeze(-1)  # [n_random_samples, local_time_steps]
                
                # 【关键修改】：对 random 采样后的 local views 应用 weak augmentation
                x_random_aug = get_student_input(x_random_local, 'weak_local', is_train=True)
                
                # 分配 random views
                for idx, random_idx in enumerate(random_indices):
                    x_student_local_views.append((random_idx, x_random_aug[idx*B:(idx+1)*B]))
                    # 保存对应的 time_mark 和 missing_mask
                    if time_mark_random is not None:
                        local_time_mark_list.append(time_mark_random[idx*B:(idx+1)*B])
                    else:
                        local_time_mark_list.append(None)
                    local_missing_mask_list.append(missing_mask_random[idx*B:(idx+1)*B])
            
            # 同步排序：按照 crop_idx/random_idx 排序
            sorted_pairs = sorted(zip(x_student_local_views, local_time_mark_list, local_missing_mask_list), key=lambda x: x[0][0])
            x_student_local_views = [pair[0][1] for pair in sorted_pairs]  # 提取数据部分
            local_time_mark_list = [pair[1] for pair in sorted_pairs]
            local_missing_mask_list = [pair[2] for pair in sorted_pairs]

        # 5. Student Forward (Backbone) - 处理多个student views
        
        # 使用 DINOv3 风格的 patch 掩码（包含 block + random 结构）
        # 掩码率区间由 mask_rate_v1 / mask_rate_v2 控制，提供外部接口而不是写死。
        # 如果只给了一个值（mask_rate_v2 为空），则使用固定掩码率。
        if mask_rate_v2 is None:
            mask_ratio_tuple = (mask_rate_v1, mask_rate_v1)
        else:
            mask_ratio_tuple = (min(mask_rate_v1, mask_rate_v2), max(mask_rate_v1, mask_rate_v2))
        mask_sample_probability = float(self.mask_sample_probability)

        collated_masks_global, mask_indices_list_global, masks_weight_global = random_patch_masking_dinov3_style(
            B * n_global_student,
            mask_ratio_tuple,
            mask_sample_probability,
            num_patches_cur,
            device,
            block_ratio=self.block_mask_ratio,
        )
        collated_masks_global = collated_masks_global.view(n_global_student, B, num_patches_cur)
        
        # Global student views（带mask）- 批量处理
        # 【关键修改】：动态构建 missing_mask_global
        global_masks_list = []
        for use_imp in x_student_global_use_imputator:
            if use_imp:
                # 使用 imputator 的视图，mask 为全 0 (即 missing_mask，此时已被置为0)
                global_masks_list.append(missing_mask)
            else:
                # 不使用 imputator 的视图，默认使用原始缺失情况
                global_masks_list.append(missing_mask_orig)
        
        missing_mask_global = torch.cat(global_masks_list, dim=0).float() # [n_global*B, T]
        # woMask 模式下，不使用 missing_mask embedding，统一视为全 0
        if imputator_mode == 'woMask':
            missing_mask_global = torch.zeros_like(missing_mask_global)
        
        time_mark_global = time_mark.repeat(n_global_student, 1, 1) if time_mark is not None else None
        mask_map_global_batch = collated_masks_global.view(n_global_student * B, num_patches_cur)
        
        # 一次性forward所有global student views
        out_student_global_batch = self.backbone(
            x_student_global_views_batch, missing_mask_global, time_mark_global,
            mask_map=mask_map_global_batch, is_student=True
        )
        
        s_logits_global = out_student_global_batch['logits_global'].view(n_global_student, B, -1)
        s_z_cls_global_list = out_student_global_batch['z_cls'].view(n_global_student, B, -1)
        # 注意：ibot_head 现在不在 backbone 中计算，只返回 normalized features
        s_z_patch_global = out_student_global_batch['z_patch_enc'].view(n_global_student, B, -1, out_student_global_batch['z_patch_enc'].shape[-1])
        
        # 注意：ibot_head 对 masked patches 的应用将在后面根据 valid_sample_indices 筛选后进行
        # 这里先保存 pre-head features
        
        s_z_patch_enc = s_z_patch_global[0]  # [B, N, D]
        s_z_global = s_z_cls_global_list[0]  # [B, D]
        
        # 【修改】：使用所有global student views的重建结果，而不仅仅是第一个
        if out_student_global_batch['rec_patches'] is not None:
            # rec_patches形状: [n_global_student * B, N, patch_len * c_out]
            # 现在使用所有global student views的重建结果
            s_rec_patches = out_student_global_batch['rec_patches']  # [n_global_student * B, N, patch_len * c_out]
        else:
            s_rec_patches = None
        
        # Local student views（不带mask）- 批量处理
        if n_local_student > 0 and len(x_student_local_views) > 0:
            x_student_local_batch = torch.cat(x_student_local_views, dim=0)
            B_local, T_local, C_local = x_student_local_views[0].shape
            
            # 【关键修复】：使用对应 crop/random 位置的 time_mark 和 missing_mask
            # 优化：过滤 None 后拼接（如果 time_mark 为 None，local_time_mark_list 中会有 None）
            if len(local_time_mark_list) > 0 and local_time_mark_list[0] is not None:
                time_mark_local = torch.cat([tm for tm in local_time_mark_list if tm is not None], dim=0)
            else:
                time_mark_local = None
            
            # missing_mask：从 x_target_perfect 生成，missing_mask 可能不全为0（取决于是否有 imputator）
            if len(local_missing_mask_list) > 0:
                missing_mask_local = torch.cat(local_missing_mask_list, dim=0)  # [n_local_student * B, T_local]
            else:
                missing_mask_local = torch.zeros(n_local_student * B, T_local, device=device, dtype=torch.bool)
            
            out_student_local_batch = self.backbone(
                x_student_local_batch, missing_mask_local.float(), time_mark_local,
                mask_map=None, is_student=True
            )
            s_logits_local = out_student_local_batch['logits_global'].view(n_local_student, B, -1)
        else:
            s_logits_local = None

        # 6. Loss Calculation Preparation
        
        # 辅助函数：根据 Valid Sample 筛选（基于原始观测质量，而不是 imputator 后的 missing_mask）
        # 说明：
        # - 即使使用 imputator 填充了缺失值，如果原始有效观测点极少，该样本的 pseudo target 仍不可靠，
        #   不应参与 DINO / iBOT 的 self-distillation loss。
        # - 因此这里使用 missing_mask_orig 来衡量样本有效观测比例。
        valid_ratio_per_sample = (~missing_mask_orig).float().mean(dim=-1)
        valid_sample_threshold = getattr(self, "valid_sample_threshold", 0.1)
        valid_sample_indices = torch.where(valid_ratio_per_sample >= valid_sample_threshold)[0]
        
        if len(valid_sample_indices) == 0:
            # 如果没有有效样本，返回空字典 (保持原样)
            thr_ff = float(getattr(self, "fft_align_min_valid_ratio", 0.25))
            n_ps = int(s_z_patch_global.shape[2])
            n_pt = int(t_z_patch_batch.shape[2])
            n_pa = min(n_ps, n_pt)
            if thr_ff <= 0:
                fft_idx = torch.arange(B, device=device, dtype=torch.long)
            else:
                fft_idx = torch.where(valid_ratio_per_sample >= thr_ff)[0]
            fft_idx = fft_idx[(fft_idx >= 0) & (fft_idx < B)]
            if len(fft_idx) == 0 or n_pa <= 0:
                s_patch_all = None
                t_patch_all = None
            else:
                n_pair_ff = min(int(n_global_student), int(n_teacher_views))
                s_rows_ff = []
                t_rows_ff = []
                for g in range(n_pair_ff):
                    s_rows_ff.append(s_z_patch_global[g, fft_idx, :n_pa, :])
                    t_rows_ff.append(t_z_patch_batch[g, fft_idx, :n_pa, :].detach())
                s_patch_all = torch.cat(s_rows_ff, dim=0)
                t_patch_all = torch.cat(t_rows_ff, dim=0)
            # 根据当前 epoch 计算 FFT recon 的动态权重（仅在训练阶段使用）
            current_lambda_recon = self.lambda_recon
            if self.fft_recon_warm_epochs > 0:
                epoch = getattr(self, "_current_epoch", 0)
                if epoch >= self.fft_recon_warm_epochs:
                    current_lambda_recon = 0.0
                else:
                    factor = 1.0 - float(epoch) / float(max(1, self.fft_recon_warm_epochs))
                    current_lambda_recon = self.lambda_recon_base * factor

            _epoch = getattr(self, "_current_epoch", 0)
            if self.fft_proto_warm_epochs > 0 and _epoch < self.fft_proto_warm_epochs:
                current_lambda_fft_proto = 0.0
            else:
                current_lambda_fft_proto = self.lambda_fft_proto_base

            return {
                'z_global': s_z_global,
                'valid_sample_indices': valid_sample_indices,
                'recon_data': {'rec_seq_valid': None, 'target_seq_valid': None},
                'spectral_data': {'s_patch_tokens_all': s_patch_all, 't_patch_tokens_all': t_patch_all},
                'cls_data': {'s_logits_global_valid': None, 's_logits_local_valid': None, 't_logits_global_valid': None, 'dino_global_scale': None, 'dino_local_scale': None},
                'patch_data': {'s_patch_masked': None, 't_patch_masked_centered': None, 'masks_weight_global_valid': None, 'collated_masks_global_valid': None, 'mask_indices_list_global_valid': None, 'ibot_denom_rows': None},
                'koleo_data': {'s_z_cls_flat': None},
                'cls_consistency_data': None,
                'fft_proto_data': {'s_logits_fft_valid': None, 't_logits_fft_valid': None},
                'temporal_data': {'s_z_patch_enc_valid': None},
                'lambda_weights': {
                    'lambda_recon': current_lambda_recon,
                    'lambda_fft_align': self.lambda_fft_align,
                    'lambda_cls_proto': self.lambda_cls_proto,
                    'lambda_patch_proto': self.lambda_patch_proto,
                    'lambda_koleo': self.lambda_koleo,
                    'lambda_temporal': self.lambda_temporal,
                    'lambda_cls_cons': self.lambda_cls_cons,
                    'lambda_fft_proto': current_lambda_fft_proto,
                },
                'teacher_temp': teacher_temp,
            }

        def filter_batch(tensor, indices):
            """
            按样本索引筛选 batch 维上的数据，保证空集时返回形状兼容的空张量。
            """
            if tensor is None:
                return None

            # 仅当张量维度 >=3 时认为 batch 在 dim1；2D/1D 视为 batch 在 dim0
            batch_dim = 1 if tensor.dim() >= 3 else 0
            max_idx = tensor.size(batch_dim)
            if max_idx == 0:
                return tensor

            valid_indices = indices[(indices >= 0) & (indices < max_idx)]
            if len(valid_indices) == 0:
                # 构造与原张量同维度的空张量（batch 维长度为 0）
                shape = list(tensor.shape)
                shape[batch_dim] = 0
                return tensor.new_empty(shape)

            if batch_dim == 1:
                return tensor[:, valid_indices]
            else:
                return tensor[valid_indices]

        # A. Reconstruction Loss 数据准备
        # 所有 global student views 都参与 FFT 重建，目标为 perfect target。
        rec_seq_valid = None
        target_seq_valid = None
        if s_rec_patches is not None:
            # s_rec_patches形状: [n_global_student * B, N, patch_len * c_out]
            rec_seq = unpatchify(s_rec_patches, T, self.patch_len, self.c_out)  # [n_global_student * B, T, C]
            
            # perfect target 形状: [B, T, C]，扩展到 [n_global_student * B, T, C]
            target_perfect_expanded = x_target_perfect.repeat(n_global_student, 1, 1)
            
            if len(valid_sample_indices) > 0:
                all_valid_indices = []
                for g_idx in range(n_global_student):
                    offset = g_idx * B
                    global_valid_indices = valid_sample_indices + offset
                    all_valid_indices.append(global_valid_indices)
                
                all_valid_indices_flat = torch.cat(all_valid_indices, dim=0)
                max_batch_idx = rec_seq.shape[0] - 1
                all_valid_indices_flat = all_valid_indices_flat[all_valid_indices_flat <= max_batch_idx]
                
                if len(all_valid_indices_flat) > 0:
                    rec_seq_valid = rec_seq[all_valid_indices_flat]
                    target_seq_valid = target_perfect_expanded[all_valid_indices_flat]

        # B. CLS Loss (DINO) 数据准备
        max_batch_size = s_logits_global.shape[1] if len(s_logits_global.shape) >= 2 else s_logits_global.shape[0]
        valid_sample_indices_safe = valid_sample_indices[valid_sample_indices < max_batch_size]
        
        if len(valid_sample_indices_safe) == 0:
            s_logits_global_valid = s_logits_global[:, :0, :] if len(s_logits_global.shape) == 3 else s_logits_global[:0]
            s_logits_local_valid = s_logits_local[:, :0, :] if len(s_logits_local.shape) == 3 else s_logits_local[:0]
            t_logits_global_valid = t_logits_global[:, :0, :] if len(t_logits_global.shape) == 3 else t_logits_global[:0]
            dino_global_scale = None
            dino_local_scale = None
        else:
            s_logits_global_valid = filter_batch(s_logits_global, valid_sample_indices_safe)
            s_logits_local_valid = filter_batch(s_logits_local, valid_sample_indices_safe)
            t_logits_global_valid = filter_batch(t_logits_global, valid_sample_indices_safe)
            
            if len(valid_sample_indices_safe) > 0 and s_logits_global_valid.shape[1] > 0:
                n_global_crops = s_logits_global_valid.shape[0]
                n_local_crops = s_logits_local_valid.shape[0] if s_logits_local_valid is not None else 0
                if n_local_crops > 0:
                    dino_global_terms = n_global_crops * (n_global_crops - 1)
                    dino_local_terms = n_global_crops * n_local_crops
                    dino_global_scale = dino_global_terms / (dino_global_terms + dino_local_terms)
                    dino_local_scale = dino_local_terms / (dino_global_terms + dino_local_terms)
                else:
                    dino_global_scale = 1.0
                    dino_local_scale = 0.0
            else:
                dino_global_scale = None
                dino_local_scale = None

        # 根据当前 epoch 计算 FFT recon 的动态权重（仅在训练阶段使用）
        current_lambda_recon = self.lambda_recon
        if self.fft_recon_warm_epochs > 0:
            epoch = getattr(self, "_current_epoch", 0)
            if epoch >= self.fft_recon_warm_epochs:
                current_lambda_recon = 0.0
            else:
                factor = 1.0 - float(epoch) / float(max(1, self.fft_recon_warm_epochs))
                current_lambda_recon = self.lambda_recon_base * factor

        _epoch = getattr(self, "_current_epoch", 0)
        if self.fft_proto_warm_epochs > 0 and _epoch < self.fft_proto_warm_epochs:
            current_lambda_fft_proto = 0.0
        else:
            current_lambda_fft_proto = self.lambda_fft_proto_base

        # C. Patch Loss (iBOT) 数据准备
        # 注意：现在 ibot_head 只对 masked patches 应用（与 DINOv3 一致）
        s_patch_masked = None
        t_patch_masked = None
        masks_weight_global_valid = None
        collated_masks_global_valid = None
        mask_indices_list_global_valid = None
        ibot_denom_rows = None

        if len(mask_indices_list_global) > 0:
            collated_masks_global_valid = collated_masks_global[:, valid_sample_indices_safe, :]
            
            # 重新计算 valid 样本的 mask_indices（基于筛选后的 valid_sample_indices_safe）
            valid_mask_flatten = collated_masks_global_valid.flatten()  # [n_global_student * B_valid * N]
            mask_indices_list_global_valid = valid_mask_flatten.nonzero(as_tuple=False).squeeze(1)
            
            # 【Patch 可靠性过滤】排除"0 匹配 0"的无意义 patch：至少 1 个有效观测点才参与 iBOT loss
            # full 模式下 is_patch_reliable_for_ibot 全为 True，不过滤
            if len(mask_indices_list_global_valid) > 0:
                n_gs, B_valid, N_patch = collated_masks_global_valid.shape
                is_patch_reliable_valid = is_patch_reliable_for_ibot[valid_sample_indices_safe, :]  # [B_valid, N]
                flat_rest = mask_indices_list_global_valid % (B_valid * N_patch)
                patch_sample_idx = flat_rest // N_patch
                patch_patch_idx = flat_rest % N_patch
                reliable_flags = is_patch_reliable_valid[patch_sample_idx, patch_patch_idx]  # [n_masked]
                mask_indices_list_global_valid = mask_indices_list_global_valid[reliable_flags]
            
            if len(mask_indices_list_global_valid) > 0:
                # Student: 选择 masked patches 的 pre-head features，然后应用 ibot_head
                s_patch_flat_valid = s_z_patch_global[:, valid_sample_indices_safe, :, :].flatten(0, 1).flatten(0, 1)  # [n_global_student * B_valid * N, D]
                s_masked_patches_pre_head = torch.index_select(s_patch_flat_valid, dim=0, index=mask_indices_list_global_valid)
                s_patch_masked = self.backbone.ibot_head(s_masked_patches_pre_head)  # [n_masked_valid, K]
                
                # Teacher: 选择 masked patches 的 pre-head features，然后应用 ibot_head
                t_patch_all_global = t_z_patch_batch.view(n_teacher_views, B, -1, t_z_patch_batch.shape[-1])  # [n_teacher, B, N, D]
                t_patch_all_global_valid = t_patch_all_global[:, valid_sample_indices_safe, :, :]  # [n_teacher, B_valid, N, D]
                t_patch_flat_valid = t_patch_all_global_valid.flatten(0, 1).flatten(0, 1)  # [n_teacher * B_valid * N, D]
                t_masked_patches_pre_head = torch.index_select(t_patch_flat_valid, dim=0, index=mask_indices_list_global_valid)
                # Teacher 使用 no_grad 和 detach
                with torch.no_grad():
                    t_patch_masked = self.teacher.ibot_head(t_masked_patches_pre_head).detach()  # [n_masked_valid, K]
                
                # 计算 masks_weight（按样本平衡 masked patch 数量）
                masks_per_sample = collated_masks_global_valid.sum(-1)  # [n_global_student, B_valid]
                masks_weight_per_sample = (1 / masks_per_sample.clamp(min=1.0))
                masks_weight_expanded = masks_weight_per_sample.unsqueeze(-1).expand_as(collated_masks_global_valid)  # [n_global_student, B_valid, N]
                masks_weight_flatten = masks_weight_expanded.flatten()  # [n_global_student * B_valid * N]
                masks_weight_global_valid = masks_weight_flatten[mask_indices_list_global_valid]  # [n_masked_valid]

                # 方案四：对包含 imputator 填补位置的 patch 做降权（仅当启用 imputator 且权重 < 1 时生效）
                if imputator is not None and self.imputed_patch_weight != 1.0:
                    n_gs, B_valid, N_patch = collated_masks_global_valid.shape
                    has_imputed_patch_valid = has_imputed_patch[valid_sample_indices_safe, :]  # [B_valid, N_patch]
                    # 复用前面计算的 sample / patch 索引，将其映射到 has_imputed_patch 上
                    flat_rest = mask_indices_list_global_valid % (B_valid * N_patch)
                    patch_sample_idx = flat_rest // N_patch
                    patch_patch_idx = flat_rest % N_patch
                    has_imputed_for_masked = has_imputed_patch_valid[patch_sample_idx, patch_patch_idx]  # [n_masked_valid]
                    # 对包含 imputed 的 patch 乘以 imputed_patch_weight，其余保持 1
                    weight_factor = torch.ones_like(masks_weight_global_valid)
                    weight_factor = torch.where(
                        has_imputed_for_masked,
                        torch.full_like(weight_factor, self.imputed_patch_weight),
                        weight_factor,
                    )
                    masks_weight_global_valid = masks_weight_global_valid * weight_factor
                
                n_masked = len(t_patch_masked)
                if masks_weight_global_valid is not None and len(masks_weight_global_valid) != n_masked:
                    if len(masks_weight_global_valid) > n_masked:
                        masks_weight_global_valid = masks_weight_global_valid[:n_masked]
                    else:
                        padding = torch.ones(n_masked - len(masks_weight_global_valid), device=masks_weight_global_valid.device, dtype=masks_weight_global_valid.dtype)
                        masks_weight_global_valid = torch.cat([masks_weight_global_valid, padding])

                # iBOT 标量分母：过滤后仍至少贡献 1 个 masked patch 的 (global_view, sample) 行数
                _, _, N_patch_denom = collated_masks_global_valid.shape
                row_ids = mask_indices_list_global_valid // N_patch_denom
                ibot_denom_rows = float(max(1, int(torch.unique(row_ids).numel())))

        # D. Koleo Regularization
        s_z_cls_global_valid = filter_batch(s_z_cls_global_list, valid_sample_indices_safe)
        s_z_cls_flat = None
        if s_z_cls_global_valid.shape[0] > 0 and s_z_cls_global_valid.shape[1] > 0:
            s_z_cls_flat = s_z_cls_global_valid.flatten(0, 1)

        # D'. Raw–Imputed CLS 一致性：仅当存在 imputator 且 student 同时有 raw 与 imputed 两路 global view 时提供
        cls_consistency_data = None
        if (
            imputator is not None
            and len(x_student_global_use_imputator) >= 2
            and sum(x_student_global_use_imputator) == 1
            and s_z_cls_global_valid.shape[0] >= 2
            and s_z_cls_global_valid.shape[1] > 0
        ):
            cls_consistency_data = {
                's_z_cls_per_view': s_z_cls_global_valid,
                'use_imputator_per_view': x_student_global_use_imputator,
            }

        s_logits_fft_valid = None
        t_logits_fft_valid = None
        if current_lambda_fft_proto > 0 and len(valid_sample_indices_safe) > 0:
            s_fft_list = []
            for g in range(n_global_student):
                s_z = s_z_patch_global[g, valid_sample_indices_safe]
                s_fft_list.append(self.backbone.forward_fft_logits(s_z))
            s_logits_fft_valid = torch.stack(s_fft_list, dim=0)
            t_fft_list = []
            t_z_bt = t_z_patch_batch.view(n_teacher_views, B, -1, t_z_patch_batch.shape[-1])
            for g in range(n_teacher_views):
                t_z = t_z_bt[g, valid_sample_indices_safe]
                with torch.no_grad():
                    t_fft_list.append(self.teacher.forward_fft_logits(t_z))
            t_logits_fft_valid = torch.stack(t_fft_list, dim=0)

        # E. Temporal Loss
        s_z_patch_enc_valid = filter_batch(s_z_patch_enc, valid_sample_indices_safe)

        # F. FFT Align：频域 Gram；仅 raw 有效比例 >= fft_align_min_valid_ratio 的样本参与（在 valid_sample 子集内）
        n_patch_student = int(s_z_patch_global.shape[2])
        n_patch_teacher = int(t_z_patch_batch.shape[2])
        n_patch_align = min(n_patch_student, n_patch_teacher)
        thr_ff = float(getattr(self, "fft_align_min_valid_ratio", 0.25))
        if n_patch_align <= 0:
            s_patch_tokens_all = None
            t_patch_tokens_all = None
        else:
            if n_patch_student != n_patch_teacher:
                if not hasattr(self, "_fft_align_patch_warned"):
                    self._fft_align_patch_warned = False
                if not self._fft_align_patch_warned:
                    print(
                        f"[FFT align patch mismatch] student={n_patch_student}, teacher={n_patch_teacher}, align={n_patch_align}",
                        flush=True,
                    )
                    self._fft_align_patch_warned = True
            n_pair = min(int(n_global_student), int(n_teacher_views))
            s_rows = []
            t_rows = []
            for g in range(n_pair):
                if len(valid_sample_indices_safe) == 0:
                    continue
                s_sub = s_z_patch_global[g, valid_sample_indices_safe, :n_patch_align, :]
                t_sub = t_z_patch_batch[g, valid_sample_indices_safe, :n_patch_align, :].detach()
                if thr_ff <= 0:
                    keep = torch.ones(len(valid_sample_indices_safe), device=device, dtype=torch.bool)
                else:
                    keep = valid_ratio_per_sample[valid_sample_indices_safe] >= thr_ff
                if not bool(keep.any()):
                    continue
                s_rows.append(s_sub[keep])
                t_rows.append(t_sub[keep])
            if len(s_rows) == 0:
                s_patch_tokens_all = None
                t_patch_tokens_all = None
            else:
                s_patch_tokens_all = torch.cat(s_rows, dim=0)
                t_patch_tokens_all = torch.cat(t_rows, dim=0)

        return {
            'z_global': s_z_global,
            'valid_sample_indices': valid_sample_indices_safe,
            'recon_data': {'rec_seq_valid': rec_seq_valid, 'target_seq_valid': target_seq_valid},
            'spectral_data': {'s_patch_tokens_all': s_patch_tokens_all, 't_patch_tokens_all': t_patch_tokens_all},
            'cls_data': {
                's_logits_global_valid': s_logits_global_valid,
                's_logits_local_valid': s_logits_local_valid,
                't_logits_global_valid': t_logits_global_valid,
                'dino_global_scale': dino_global_scale,
                'dino_local_scale': dino_local_scale,
            },
            'patch_data': {
                's_patch_masked': s_patch_masked,  # [n_masked_valid, K] - 已应用 ibot_head
                't_patch_masked': t_patch_masked,  # [n_masked_valid, K] - 已应用 ibot_head
                # 注意：t_logits_patch_full 不再需要，因为 ibot_head 只对 masked patches 应用
                'masks_weight_global_valid': masks_weight_global_valid,
                'collated_masks_global_valid': collated_masks_global_valid,
                'mask_indices_list_global_valid': mask_indices_list_global_valid,
                'ibot_denom_rows': ibot_denom_rows,
            },
            'koleo_data': {'s_z_cls_flat': s_z_cls_flat},
            'temporal_data': {'s_z_patch_enc_valid': s_z_patch_enc_valid},
            'cls_consistency_data': cls_consistency_data,
            'fft_proto_data': {
                's_logits_fft_valid': s_logits_fft_valid,
                't_logits_fft_valid': t_logits_fft_valid,
            },
            'lambda_weights': {
                'lambda_recon': current_lambda_recon,
                'lambda_fft_align': self.lambda_fft_align,
                'lambda_cls_proto': self.lambda_cls_proto,
                'lambda_patch_proto': self.lambda_patch_proto,
                'lambda_koleo': self.lambda_koleo,
                'lambda_temporal': self.lambda_temporal,
                'lambda_cls_cons': self.lambda_cls_cons,
                'lambda_fft_proto': current_lambda_fft_proto,
            },
            'teacher_temp': teacher_temp,
        }

    def forward(self, x_enc, time_mark=None, valid_mask=None, next_x_enc=None, mode='train', mask_rate_v1=0.3, mask_rate_v2=0.6, imputator=None, teacher_temp=None, iteration=None, total_iterations=None, current_epoch=None):
        # 如果传入了动态 teacher_temp，保存它
        if teacher_temp is not None:
            self._current_teacher_temp = teacher_temp
        
        # 保存 iteration / epoch 信息用于渐进学习和 FFT recon 调度
        if iteration is not None:
            self._current_iteration = iteration
        if total_iterations is not None:
            self._total_iterations = total_iterations
        if current_epoch is not None:
            self._current_epoch = int(current_epoch)
        
        if mode == 'train':
            return self._forward(x_enc, time_mark, mask_rate_v1, mask_rate_v2, mode='train', imputator=imputator)
        
        # # Eval Mode: Just run Student (Backbone)
        # # 复用 backbone 的 eval 返回逻辑
        # missing_mask = torch.isnan(x_enc).any(dim=-1).float()
        # x_in = x_enc.nan_to_num(0.0)
        # out = self.backbone(x_in, missing_mask, time_mark, mask_map=None, is_student=True)
        
        # # 构造兼容旧 eval 循环的字典
        # return {
        #     'dec_out': unpatchify(out['rec_patches'], x_enc.shape[1], self.patch_len, self.c_out),
        #     'cls_token': out['cls_token'],
        #     'all_attns': out['all_attns'],
        #     'inter_cos': out['cls_token_cos']
        # }

    def encode(self, xEnc, timeMark=None, imputator=None):
        """
        编码方法
        Args:
            xEnc: 输入数据 [B, T, C]
            timeMark: 时间标记 [B, T, 2]（可选）
            imputator: 插值器（可选），如果提供则用于填充缺失值
        Returns:
            包含 cls_token, storage_tokens, patch_tokens 的字典
        """
        try:
            # 与训练阶段一致地构造 missing_mask / imputator_mode 逻辑
            missing_mask_orig = torch.isnan(xEnc).any(dim=-1).float()
            imputator_mode = getattr(self, "imputator_mode", "full")
            
            if imputator is not None:
                if imputator_mode in ["full", "woMask"]:
                    # full / woMask: 训练阶段对 teacher 视图视为无缺失
                    missing_mask = torch.zeros_like(missing_mask_orig)
                elif imputator_mode == "mixed_teacher":
                    # mixed_teacher: 填补位置 0.5，其余 0
                    missing_mask = torch.where(
                        missing_mask_orig > 0,
                        torch.full_like(missing_mask_orig, 0.5, dtype=torch.float32),
                        torch.zeros_like(missing_mask_orig, dtype=torch.float32),
                    )
                else:  # recon_only 及其它：保持物理缺失
                    missing_mask = missing_mask_orig
            else:
                # 没有 imputator 时，训练阶段 missing_mask = missing_mask_orig
                missing_mask = missing_mask_orig

            x_in = xEnc.nan_to_num(0.0)
            if timeMark is None:
                timeMark = torch.zeros(xEnc.shape[0], xEnc.shape[1], 1, device=xEnc.device)
            
            # 统一改为 Teacher 分支编码；imputator 在 backbone.encode 内按 pred_len 分段填补，与训练 _forward 一致
            if imputator is not None:
                return self.teacher.encode(x_in, missing_mask, timeMark, imputator=imputator, use_student_norm=False)
            else:
                return self.teacher.encode(x_in, missing_mask, timeMark, imputator=None, use_student_norm=False)
        except Exception as e:
            print(f"Encode Error: {e}")
            return {'cls_token': torch.zeros(xEnc.shape[0], self.backbone.d_model, device=xEnc.device), 'patch_tokens': None}

    def visualize(self, xEnc, timeMark=None, imputator=None):
        """
        可视化方法：返回每层的 attention 权重、CLS token 与所有 token 的 cos 相似度等信息
        
        Args:
            xEnc: 输入数据 [B, T, C]
            timeMark: 时间标记 [B, T, 2]（可选）
            imputator: 插值器（可选），如果提供则用于填充缺失值
        
        Returns:
            包含以下信息的字典：
            - all_attns: 每层的 attention 权重列表
            - cls_cos_sim_all: CLS token 与所有 token 的 cos 相似度 [B, 1+R+N]
            - cls_cos_sim_patch: CLS token 与 patch tokens 的 cos 相似度 [B, N]
            - cls_token: CLS token 特征 [B, D]
            - patch_tokens: Patch tokens 特征 [B, N, D]
            - storage_tokens: Storage tokens 特征 [B, R, D] 或 None
        """
        try:
            missing_mask_orig = torch.isnan(xEnc).any(dim=-1).float()
            imputator_mode = getattr(self, "imputator_mode", "full")

            if imputator is not None:
                if imputator_mode in ["full", "woMask"]:
                    missing_mask = torch.zeros_like(missing_mask_orig)
                elif imputator_mode == "mixed_teacher":
                    missing_mask = torch.where(
                        missing_mask_orig > 0,
                        torch.full_like(missing_mask_orig, 0.5, dtype=torch.float32),
                        torch.zeros_like(missing_mask_orig, dtype=torch.float32),
                    )
                else:
                    missing_mask = missing_mask_orig
            else:
                missing_mask = missing_mask_orig

            x_in = xEnc.nan_to_num(0.0)
            if timeMark is None: 
                timeMark = torch.zeros(xEnc.shape[0], xEnc.shape[1], 2, device=xEnc.device)
            
            # 使用 Teacher 分支进行可视化（更稳定）
            # 调用 teacher 的 visualize 方法
            return self.teacher.visualize(x_in, missing_mask, timeMark, imputator=imputator, use_student_norm=False)
        except Exception as e:
            print(f"Visualize Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                'all_attns': [],
                'cls_cos_sim_all': None,
                'cls_cos_sim_patch': None,
                'cls_token': torch.zeros(xEnc.shape[0], self.backbone.d_model, device=xEnc.device),
                'patch_tokens': None,
                'storage_tokens': None,
            }
