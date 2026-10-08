import argparse
import os
import torch
import torch.distributed as dist  # 必须导入这个
from exp.exp_ssl import Exp_SSL
from utils.tools import clamp_experiment_setting_for_checkpoint
import random
import numpy as np

# 固定随机种子
fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# GPU 性能优化设置
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

parser = argparse.ArgumentParser(description='TimeMixer')

# basic config
parser.add_argument(
    '--task_name',
    type=str,
    default='imputation',
    help="must be 'imputation' (historical CLI flag; training uses Exp_SSL only; other task exps removed)",
)
parser.add_argument('--mode', type=str, default='test', help='train, fine-tune, test, pred and visualize')
parser.add_argument('--model_id', type=str, default='RSTS-pixel', help='model id')
parser.add_argument('--model', type=str, default='RSTS_Pixel',
                    help='model name')
parser.add_argument('--pretrain_model', type=str, default=None, help='use pretrain model')

# data loader
parser.add_argument('--data', type=str, default='HLS', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='rs',
                    help='freq for time features encoding')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument(
    '--resume_checkpoint',
    type=str,
    default=None,
    help='resume training state path (file or directory). Supports full train-state resume including optimizer/scaler/RNG.',
)
parser.add_argument('--save_dir', type=str, default='./output_images/', help='directory to save prediction outputs')

# train data subsampling (for Exp_SSL / run.py training ablations)
parser.add_argument(
    '--train_data_ratio',
    type=float,
    default=1.0,
    help='ratio of training data to use (0 < ratio <= 1). Only affects flag=train in data provider.',
)

# forecasting task
parser.add_argument('--seq_len', type=int, default=610, help='input sequence length')
parser.add_argument('--label_len', type=int, default=610, help='start token length')
parser.add_argument('--pred_len', type=int, default=122, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

# model define
parser.add_argument('--embed_type', type=int, default=3, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--enc_in', type=int, default=5, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=5, help='decoder input size')
parser.add_argument('--c_out', type=int, default=5, help='output size')
parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--e_layers', type=int, default=6, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=2, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.0, help='dropout (MHA/FFN); TED_modular 主干默认 0')
parser.add_argument(
    '--drop_path', '--drop_depth',
    type=float,
    default=0.1,
    help='stochastic depth 概率（按层、按样本随机跳过整层残差更新；与 --drop_depth 同义；默认 0.1 与历史稳定 HLS 配置一致，更强正则可显式传 0.2）',
)
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', default=True, action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--channel_independence', type=int, default=1,
                    help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
parser.add_argument('--down_sampling_layers', type=int, default=3, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default='avg',
                    help='down sampling method, only support avg, max, conv')
parser.add_argument('--use_future_temporal_feature', type=int, default=0,
                    help='whether to use future_temporal_feature; True 1 False 0')


parser.add_argument('--use_rope', type=int, default=1, help='1: use RoPE only; 0: use APE only (若 RoPE 精度下降可设为 0 恢复原行为)')
parser.add_argument('--rope_base', type=float, default=100.0, help='RoPE base (DINOv3 用 100；语言模型常用 10000，序列较短可试 100)')
parser.add_argument(
    '--curriculum_strategy',
    type=str,
    default='mixed_batch',
    help=(
        "curriculum / batch: "
        "'mixed_batch' splits each GPU's batch into micro-batches, each forward draws its own random "
        "sequence length and (TED_modular) local-view patch ratio; backward uses ssl_num_valid_samples "
        "weighting when --mixed_batch_weight_by_valid_samples 1; grad accumulation then one optimizer step. "
        "'none'/'fast'/... single forward per GPU per step (faster; one random length per GPU per step)."
    ),
)
parser.add_argument('--curriculum_length_jitter', type=int, default=61,
                    help='mixed_batch 起点抖动半径（步）；选定年对齐长度后窗口起点 ±jitter 随机偏移，0 关闭')
parser.add_argument('--curriculum_jitter_probability', type=float, default=0.5,
                    help='起点抖动触发概率（0-1），默认 0.5')
parser.add_argument(
    '--disable_view_augmentation',
    action='store_true',
    help='关闭 teacher/student 视图的幅度 scale、平移 shift、加性 noise（及 student global 的 channel mask）',
)
parser.add_argument('--global_shift_steps', type=int, default=0,
                    help='第二个 global view 相对 anchor 窗的偏移步数；与 global_shift_ratio 二选一。'
                         '若 steps/ratio 都为 0，则触发时改为独立随机采样另一个同长度时间窗')
parser.add_argument('--global_shift_jitter_steps', type=int, default=0,
                    help='平移步数在 base shift 上的抖动半径（± 步），0 表示不用步数抖动（可用 jitter_ratio）')
parser.add_argument('--global_shift_ratio', type=float, default=0.0,
                    help='用序列长度比例定义第二个 global view 的基准偏移；>0 时优先于 global_shift_steps')
parser.add_argument('--global_shift_jitter_ratio', type=float, default=0.0,
                    help='相对序列长度的平移抖动比例；0 则仅用 global_shift_jitter_steps')
parser.add_argument('--global_shift_probability', type=float, default=1.0,
                    help='每个训练 forward 让第二个 global view 使用不同时间窗的概率（0–1）；0 表示两份 global 共用 anchor 窗')
parser.add_argument('--global_shift_min_overlap_ratio', type=float, default=0.6,
                    help='两个 global 时间窗的最小重叠比例（相对窗口长度）；默认 0.6，保证仍属于同一语义段')
parser.add_argument(
    '--global_shift_mode',
    type=str,
    default='base_jitter',
    choices=['uniform', 'base_jitter'],
    help='uniform: 围绕基准偏移均匀采样；base_jitter: 基准偏移加抖动。仅当 ratio/steps > 0 时生效',
)
parser.add_argument('--mixed_batch_groups', type=int, default=2,
                    help='curriculum_strategy=mixed_batch 时每卡拆分的子 batch 数，默认 2（速度与变长混合折中）')
parser.add_argument(
    '--mixed_batch_local_patch_divisors',
    type=str,
    default='8,6,4',
    help='mixed_batch 训练时 local view 的 patch 比例：local_token_num=max(1,N_patches//d)，'
         '约等于全窗 patch 数的 1/d；逗号分隔，与 mixed_batch_local_patch_divisor_probs 等长',
)
parser.add_argument(
    '--mixed_batch_local_patch_divisor_probs',
    type=str,
    default='0.5,0.25,0.25',
    help='与 divisors 同长度的采样概率（自动归一化）；仅在 curriculum_strategy=mixed_batch 且 train 时生效',
)
parser.add_argument(
    '--mixed_batch_weight_by_valid_samples',
    type=int,
    default=1,
    help='1：mixed_batch 子 batch 反向时用「过 valid_sample_threshold 的样本数」加权，'
         '再在 optimizer 步前按总有效样本数归一（对齐「整批一次 forward」的样本均值梯度）；'
         '0：保持旧行为（子 batch 等权 1/G）',
)
parser.add_argument('--imputator_mode', type=str, default='full',
                    help="imputator 使用策略: "
                         "'full' (Teacher 全用 imputator, Student 一半用), "
                         "'recon_only' (Teacher/Student 都用 raw, 只在重建目标用 imputator), "
                         "'mixed_teacher' (Teacher/Student 都混合使用 imputator/raw, patch 过滤保留)")
parser.add_argument(
    '--imputator_segment_stride',
    type=int,
    default=244,
    help='超长序列时滑动窗口起点步长（如 244≈两年）；窗口长=imputator.pred_len(默认366)，重叠区对缺失预测取均值',
)
parser.add_argument(
    '--use_lon_lat_embed',
    type=int,
    default=1,
    help='1: Backbone 使用多频 sin/cos 经纬度嵌入（WGS84 度）；0: 关闭，兼容无地理坐标场景',
)
parser.add_argument(
    '--lon_lat_n_fourier_freqs',
    type=int,
    default=4,
    help='经纬度傅里叶特征频数；嵌入维=4*该值再 Linear 到 d_model（4 足够区分气候区，无需更高）',
)
parser.add_argument(
    '--geo_dropout_p',
    type=float,
    default=0.5,
    help='训练时丢弃经纬度嵌入概率（CFG 式）；0=总使用；默认 0.5 减弱对地理捷径依赖',
)
parser.add_argument(
    '--missing_mask_embed_dropout',
    type=float,
    default=0.5,
    help='训练时按样本整段丢弃 missing_mask 嵌入概率（CFG 式）；0=关闭；默认 0.5 减弱对缺失通道捷径依赖',
)
parser.add_argument(
    '--use_missing_mask_embed',
    type=int,
    default=1,
    help='1: 将缺失掩码线性嵌入加到 patch（默认）；0: 关闭该项（仍保留线性层权重以兼容旧 checkpoint），仅与时间/数值等相加',
)
parser.add_argument('--n_storage_tokens', type=int, default=0, help='number of storage tokens')
parser.add_argument(
    '--n_cls_tokens',
    type=int,
    default=1,
    help='CLS token 数量；>1 时在 encoder 后对各 CLS 做 Qwen 式 sigmoid 门控加权融合再进 DINO head（默认 1 与旧 checkpoint 一致）',
)
parser.add_argument('--evidence_gap_distill', type=int, default=1, help='1: use single-teacher evidence-gap CLS distillation; 0: use legacy global/local DINO path')
parser.add_argument('--evidence_gap_teacher_lengths', type=str, default='61,122,183,244,366,488,732', help='comma-separated teacher window lengths for evidence-gap distillation')
parser.add_argument('--evidence_gap_student_ratio_min', type=float, default=0.1, help='minimum student/teacher patch-token ratio in evidence-gap distillation')
parser.add_argument('--evidence_gap_student_ratio_max', type=float, default=0.9, help='maximum student/teacher patch-token ratio in evidence-gap distillation')
parser.add_argument('--evidence_gap_cls_bins', type=str, default='5,10,21,41,82', help='comma-separated token-gap upper bounds; N bounds create N+1 CLS groups')
parser.add_argument('--evidence_gap_condition', type=int, default=1, help='1: use relation-conditioned CLS readout for short evidence-gap views; 0: use legacy gap-bin CLS routing')
parser.add_argument('--evidence_gap_condition_alpha', type=float, default=0.1, help='residual scale for relation-conditioned evidence-gap readout')
parser.add_argument('--evidence_gap_condition_view_embed_dim', type=int, default=8, help='view-type embedding dimension for relation condition')
parser.add_argument('--evidence_gap_condition_scalar_embed_dim', type=int, default=8, help='embedding dimension for each scalar relation condition')
parser.add_argument('--evidence_gap_condition_scalar_n_freqs', type=int, default=4, help='number of Fourier frequencies for scalar relation condition embeddings')
parser.add_argument('--evidence_gap_condition_hidden_dim', type=int, default=0, help='hidden dimension of relation condition adapter/MLP; 0 uses a small default')
parser.add_argument(
    '--evidence_gap_condition_readout',
    type=str,
    default='adapter',
    choices=[
        'adapter', 'direction', 'gate', 'film', 'cond_mlp', 'cond_sum_mlp', 'cond_film_mlp',
        'cond_res_mlp', 'cond_res_film_mlp', 'cond_gate_bottleneck', 'cond_mul_bottleneck',
        'cond_xattn_bottleneck', 'cond_blend_mlp',
    ],
    help='short-view condition readout; cond_* bottleneck variants reuse dino_head.mlp base',
)
parser.add_argument(
    '--evidence_gap_condition_drop_p',
    type=float,
    default=0.0,
    help='per-sample prob to skip condition readout on short views and use raw z->dino_head (train only); 0 disables',
)
parser.add_argument('--evidence_gap_student_aug', type=str, default='strong', choices=['none', 'weak', 'weak_local', 'strong'], help='student augmentation mode for evidence-gap distillation; strong is noise + channel masking')
parser.add_argument('--evidence_gap_n_short_crop', type=int, default=4, help='number of short contiguous crop students in evidence-gap distillation')
parser.add_argument('--evidence_gap_n_short_random', type=int, default=2, help='number of short random-token students in evidence-gap distillation')
parser.add_argument('--evidence_gap_short_outside_teacher', type=int, default=0, help='1: sample crop/random shorts from timeline regions outside the teacher window (global stays inside teacher)')
parser.add_argument('--evidence_gap_independent_fullseq', type=int, default=0, help='1: crop/random independently random-sample on full timeline; global stays in teacher window')
parser.add_argument('--evidence_gap_same_short_multi_teacher_prob', type=float, default=0.25, help='train-time probability to add same-short multi-teacher anchor rows per step')
parser.add_argument('--evidence_gap_same_short_multi_teacher_count', type=int, default=1, help='number of alternate teacher windows paired with a reused crop short z per anchor step')
parser.add_argument('--evidence_gap_same_short_anchor_from_crop_only', type=int, default=1, help='1: anchor same-short multi-teacher only from crop short views; 0: allow any short row (crop only implemented for now)')
parser.add_argument('--evidence_gap_dual_teacher_cross', type=int, default=0, help='1: sample two same-length teacher windows and DINO-style cross-view pairing (2 global + n_short_crop+n_short_random shorts per side)')
parser.add_argument('--evidence_gap_dual_teacher_short_per_side', type=int, default=4, help='legacy fallback: crop-only shorts per teacher window when dual_teacher_cross=1 and n_short_crop=n_short_random=0')
parser.add_argument('--evidence_gap_dual_teacher_patch_cross', type=int, default=1, help='1: iBOT patch loss uses cross-window teacher targets (default dualT2); 0: same-window patch pairing while CLS may stay cross')
parser.add_argument('--evidence_gap_version', type=str, default='v2', choices=['v2', 'v2.5', 'v3', 'v4'], help='condition builder: v2=ratio+position in teacher window; v2.5=ratio+timeline offset; v4=teacher scale + timeline offset')
parser.add_argument('--dino_head_n_prototypes', type=int, default=256, help='number of prototypes in DINO head')
parser.add_argument('--dino_head_hidden_dim', type=int, default=128, help='hidden dimension in DINO head')
parser.add_argument('--dino_head_bottleneck_dim', type=int, default=64, help='bottleneck dimension in DINO head')
parser.add_argument('--dino_head_nlayers', type=int, default=3, help='number of layers in DINO head')
parser.add_argument('--ibot_head_n_prototypes', type=int, default=256, help='number of prototypes in iBOT head')
parser.add_argument('--ibot_head_hidden_dim', type=int, default=128, help='hidden dimension in iBOT head')
parser.add_argument('--ibot_head_bottleneck_dim', type=int, default=64, help='bottleneck dimension in iBOT head')
parser.add_argument('--ibot_head_nlayers', type=int, default=3, help='number of layers in iBOT head')
parser.add_argument('--teacher_temp', type=float, default=0.07, help='teacher temperature (final value)')
parser.add_argument('--warmup_teacher_temp', type=float, default=0.04, help='warmup teacher temperature (initial value)')
parser.add_argument('--warmup_teacher_temp_epochs', type=int, default=30, help='warmup epochs for teacher temperature')
parser.add_argument('--student_temp', type=float, default=0.1, help='student temperature (fixed)')
parser.add_argument(
    '--cls_global_loss_mode',
    type=str,
    default='dino',
    choices=['dino', 'overlap_compat'],
    help='global CLS loss: dino = cross-view CE with diagonal ignored; '
    'overlap_compat = self-view CE plus overlap-weighted cross-view CE for shifted temporal globals',
)
parser.add_argument(
    '--cls_global_compat_self_weight',
    type=float,
    default=1.0,
    help='overlap_compat: weight for same-window student/teacher global CE',
)
parser.add_argument(
    '--cls_global_compat_cross_floor',
    type=float,
    default=0.25,
    help='overlap_compat: minimum cross-window CE weight at the configured minimum overlap',
)
parser.add_argument(
    '--cls_global_compat_min_overlap',
    type=float,
    default=-1.0,
    help='overlap_compat: overlap mapped to the cross-weight floor; <0 uses global_shift_min_overlap_ratio',
)
parser.add_argument(
    '--cls_local_loss_mode',
    type=str,
    default='per_view',
    choices=['per_view', 'bag', 'crop_set'],
    help='local CLS vs teacher: '
    'per_view = standard DINOLoss over local×teacher crops; '
    'bag = logmeanexp bag over crop locals vs mean teacher prob (see cls_local_bag_loss); '
    'crop_set = set-posterior pool over crop views + optional per-crop anchor (see crop_view_loss). '
    'bag/crop_set auto-split crop vs random locals when cls_data provides counts.',
)
parser.add_argument(
    '--cls_local_crop_gamma',
    type=float,
    default=2.0,
    help='crop_set mode: generalized-mean sharpness for pooling crop views into set_prob',
)
parser.add_argument(
    '--cls_local_crop_lambda_set',
    type=float,
    default=0.5,
    help='crop_set mode: weight for set-level posterior loss vs teacher globals',
)
parser.add_argument(
    '--cls_local_crop_lambda_ind',
    type=float,
    default=1.0,
    help='crop_set mode: weight for per-crop anchor loss vs teacher globals',
)
parser.add_argument(
    '--cls_local_cross_teacher_beta',
    type=float,
    default=0.0,
    help='optional weak local-to-non-parent teacher CE weight; 0 keeps parent-only local supervision',
)
parser.add_argument(
    '--cls_local_cross_teacher_normalize',
    type=int,
    default=1,
    help='1 normalizes parent + weak cross local CE by its summed weights to preserve local loss scale',
)
parser.add_argument(
    '--lambda_cls_local_contrib',
    type=float,
    default=0.0,
    help='bag mode only: weight for weak per-local contribution regularizer (0 = disabled)',
)
parser.add_argument(
    '--cls_local_contrib_margin',
    type=float,
    default=0.0,
    help='bag mode: margin in relu(margin - contrib)^2 when lambda_cls_local_contrib > 0',
)
parser.add_argument('--momentum_teacher', type=float, default=0.992, help='EMA momentum (initial value)')
parser.add_argument('--final_momentum_teacher', type=float, default=1.0, help='EMA momentum (final value)')
parser.add_argument('--empty_cache_interval', type=int, default=0, help='every N steps call torch.cuda.empty_cache() (0=disabled, default; set e.g. 300 if fighting fragmentation/OOM)')
parser.add_argument('--weight_decay_end', type=float, default=None, help='final weight decay (default: weight_decay * 10)')
parser.add_argument('--min_lr', type=float, default=None, help='minimum learning rate (default: lr * 1e-6)')
parser.add_argument('--freeze_last_layer_epochs', type=int, default=1, help='epochs to freeze last layer')
parser.add_argument('--schedule_trunc_extra', type=float, default=0.0, help='schedule truncation extra')
parser.add_argument(
    '--scheduler_version',
    type=str,
    default='cosine',
    choices=['cosine', 'dinov3_v2'],
    help='LR/WD/momentum/teacher_temp 调度：cosine=CosineScheduler（默认）；dinov3_v2=DINOv3 cfg.schedules v2 的 linear_warmup_cosine_decay',
)
parser.add_argument(
    '--sched_lr_start',
    type=float,
    default=None,
    help='dinov3_v2 only: LR schedule start（默认 0）',
)
parser.add_argument(
    '--sched_lr_peak',
    type=float,
    default=None,
    help='dinov3_v2 only: LR cosine peak（默认 None=scaling 后的 learning_rate）',
)
parser.add_argument(
    '--sched_lr_end',
    type=float,
    default=None,
    help='dinov3_v2 only: LR 末端（默认 None=scaling 后的 min_lr）；可与 peak 设为相同得到平顶',
)
parser.add_argument(
    '--sched_lr_warmup_epochs',
    type=int,
    default=None,
    help='dinov3_v2 only: LR 线性 warmup epoch 数（默认 None=warmup_epochs）',
)
parser.add_argument(
    '--sched_lr_cosine_epochs',
    type=int,
    default=None,
    help='dinov3_v2 only: LR 余弦段 epoch 数（默认 None=填满剩余 iter）',
)
parser.add_argument(
    '--sched_wd_warmup_epochs',
    type=int,
    default=0,
    help='dinov3_v2 only: weight_decay 线性 warmup epoch 数',
)
parser.add_argument(
    '--sched_wd_cosine_epochs',
    type=int,
    default=None,
    help='dinov3_v2 only: WD 余弦段 epoch 数（默认 None）',
)
parser.add_argument(
    '--sched_momentum_warmup_epochs',
    type=int,
    default=0,
    help='dinov3_v2 only: EMA momentum warmup epoch 数',
)
parser.add_argument(
    '--sched_momentum_cosine_epochs',
    type=int,
    default=None,
    help='dinov3_v2 only: momentum 余弦段 epoch 数（默认 None）',
)
parser.add_argument(
    '--sched_teacher_temp_end',
    type=float,
    default=None,
    help='dinov3_v2 only: teacher temperature 末端（默认=teacher_temp）',
)
parser.add_argument(
    '--sched_teacher_temp_cosine_epochs',
    type=int,
    default=None,
    help='dinov3_v2 only: teacher_temp 余弦段 epoch 数（默认 None）',
)

# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.0, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=31, help='patch length')
parser.add_argument('--stride', type=int, default=31, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=1, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=0, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

# RSTS
parser.add_argument('--traning_mode', type=str, default='all', help='all, imputation, forecast, anomaly_detection')
parser.add_argument('--fine_tune_mode', type=str, default='all', help='all, imputation, forecast, anomaly_detection')
parser.add_argument('--dispersive_loss', type=str, default='False', help='True or False')
parser.add_argument('--mlp_ratio', type=int, default=2, help='mlp ratio')
parser.add_argument('--num_register_tokens', type=int, default=4, help='num of register tokens')

# imputation task
parser.add_argument('--mask_rate', type=float, default=0.8, help='mask ratio (for input-level masking)')
parser.add_argument('--mask_rate_v1', type=float, default=0.3, help='min patch mask ratio for TED (DINOv3-style masking)')
parser.add_argument('--mask_rate_v2', type=float, default=0.6, help='max patch mask ratio for TED (DINOv3-style masking)')
parser.add_argument(
    '--mask_sample_probability',
    type=float,
    default=0.5,
    help='fraction of global student rows that use patch masking (DINO-style: not every view is masked); set 1.0 to mask all rows',
)
parser.add_argument(
    '--local_view_patch_divisor',
    type=int,
    default=8,
    help='TED/TED_modular local views (crop & random patch): '
         'local_token_num = max(1, num_patches_in_teacher_win // divisor) ≈ 1/divisor of patches. '
         'Must be >= 1 (default 8 matches previous hard-coded behavior). '
         'Independent of n_local_student (number of local views).',
)
parser.add_argument(
    '--ted_modular_n_local_student',
    type=int,
    default=8,
    help='TED_modular only: number of student local views (temporal crop + random-token views). '
         'Default 8 matches previous behavior.',
)
parser.add_argument(
    '--ted_modular_n_local_random_views',
    type=int,
    default=2,
    help='TED_modular only: how many of the local views use random patch-token sampling; '
         'the rest use temporal crop. Default 2 => 6 crop + 2 random. Set 0 for all crop (no random local views).',
)
parser.add_argument('--dynamic_sequence', type=int, default=0, help='training and validating with dynamic sequence (0/1)')
parser.add_argument('--sampling_stride', type=int, default=610, help='when time_step is large, how to sample')
parser.add_argument('--delay', type=int, default=610, help='token length for forecast')
parser.add_argument('--time_mark', type=int, default=1, help='use time_mark (0/1)')


# RS application task
parser.add_argument('--probe', type=int, default=0, help='whether to run KNN probe evaluation (0/1)')
parser.add_argument('--probe_interval', type=int, default=10, help='run KNN probe every N epochs (only rank 0 runs probe, reduces GPU 0 memory spike)')
parser.add_argument('--use_pretrained_imputator', type=int, default=0, help='whether to use pretrained imputator (0/1)')
parser.add_argument('--pretrained_imputator_path', type=str, default='./checkpoints/models/Transformer_Imputator.pth', help='path to pretrained imputator')
parser.add_argument('--imp_d_model', type=int, default=256, help='imputator d_model (used when loading pretrained imputator inside TED)')
parser.add_argument('--imp_n_heads', type=int, default=8, help='imputator n_heads')
parser.add_argument('--imp_e_layers', type=int, default=6, help='imputator e_layers')
parser.add_argument('--imp_d_ff', type=int, default=1024, help='imputator d_ff')
parser.add_argument(
    '--imp_n_storage_tokens',
    type=int,
    default=2,
    help='Imputator storage token 数（默认 2，与集群 Imputator 训练一致）；设为 -1 时改为跟随 n_storage_tokens（>0）否则 2。加载预训练到 TED 须与 checkpoint 一致。与 backbone 的 --n_storage_tokens 无关。',
)
parser.add_argument(
    '--imp_rec_loss',
    type=str,
    default='mse',
    choices=['mse', 'huber', 'mae'],
    help='Imputator (Transformer) reconstruction term: mse | huber | mae',
)
parser.add_argument('--imp_huber_delta', type=float, default=1.0, help='Huber delta when imp_rec_loss=huber')
parser.add_argument('--imp_rec_alpha', type=float, default=1.0, help='weight for reconstruction term in Imputator cal_rec_loss')
parser.add_argument('--imp_smooth_beta', type=float, default=0.5, help='weight for smooth_loss term in Imputator cal_rec_loss')
parser.add_argument(
    '--imp_smooth_mode',
    type=str,
    default='dy2',
    choices=['dy1', 'dy2'],
    help='Imputator smooth_loss: dy1=一阶差分平方均值, dy2=二阶差分(默认)',
)
parser.add_argument(
    '--imp_mask_min_p',
    type=float,
    default=0.4,
    help='Imputator 训练时 apply_mask 随机掩码率的下界 [min_p, mask_rate]（use_random_p=True）；原先默认 0.25',
)
parser.add_argument(
    '--imp_trim_topk_per_seq',
    type=int,
    default=0,
    help='每个样本在重建监督点中忽略误差最大的 topK 点；0 表示关闭（建议先试 3~10）',
)
parser.add_argument(
    '--imp_trim_min_keep',
    type=int,
    default=8,
    help='topK 忽略后每个样本最少保留的监督点数，避免监督被清空',
)
parser.add_argument('--downstream_data_root', type=str, default='/intelnvme01/ziyun/DownStreamTasks',
                    help='Base directory for all downstream task datasets')
parser.add_argument(
    '--downstream_group',
    type=str,
    default='classification',
    help="which downstream KNN tasks to run in RS probe: 'classification', 'segmentation', or 'all'",
)
parser.add_argument('--use_alphaearth', type=int, default=0, help='whether to use AlphaEarth embeddings in RS applications (0/1)')
parser.add_argument('--alphaearth_path', type=str, default=None, help='optional override path to AlphaEarth embeddings npz file (per-dataset has its own default)')
parser.add_argument('--alphaearth_years', type=str, default=None, help='optional year range to select from AlphaEarth embeddings, format: start-end (e.g., 2017-2021)')
parser.add_argument('--save_tsne_embeddings', type=int, default=0,
                    help='是否在 RS 探针评估时额外保存用于 t-SNE 的少量样本 CLS/AlphaEarth 表达 (0/1)')

# visualization task
parser.add_argument('--visualize_checkpoint_path', type=str, default=None, help='checkpoint path for visualization (if None, use best model from setting)')
parser.add_argument('--visualize_num_samples', type=int, default=10, help='number of samples to visualize')
parser.add_argument('--imputator_vali_save_dir', type=str, default='./imputator_val_plots',
                    help='output directory for mode=vali_imputator PNG plots')
parser.add_argument('--imputator_vali_num_samples', type=int, default=100,
                    help='number of sequences to plot in vali_imputator mode')
parser.add_argument('--imputator_vali_seed', type=int, default=2026,
                    help='RNG seed for reproducible vali_imputator sample selection (forces num_workers=0 internally)')

# anomaly detection task
parser.add_argument('--anomaly_ratio', type=float, default=0, help='prior anomaly ratio (%)')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument(
    '--no_attn_checkpoint',
    action='store_true',
    help='disable gradient checkpoint inside AttentionBlock (faster step time, higher VRAM; PyTorch TED/Patch_* only)',
)
parser.add_argument(
    '--compile_backbone',
    action='store_true',
    help='torch.compile Backbone+teacher in TED_modular (PyTorch 2.x; dynamic seq; first steps slow; measure on your GPU)',
)
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=400, help='train epochs')
parser.add_argument(
    '--max_train_steps',
    type=int,
    default=0,
    help='>0: stop after this many successful optimizer steps (smoke/debug); 0=disabled',
)
parser.add_argument('--warmup_epochs', type=int, default=1, help='warmup epochs')
parser.add_argument(
    '--batch_size',
    type=int,
    default=256,
    help='per-process (per-GPU) train batch size; global batch is this × world_size under DDP (DINOv3: batch_size_per_gpu)',
)
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='base learning rate (will be scaled by batch size)')
parser.add_argument(
    '--max_grad_norm',
    type=float,
    default=1.0,
    help='梯度裁剪阈值；默认 1.0（沿用原行为）。对齐 DINOv3 yaml 中 clip_grad≈30 时可传 --max_grad_norm 30',
)
parser.add_argument('--scaling_rule', type=str, default='sqrt_wrt_1024', help='LR scaling rule: sqrt_wrt_1024, linear_wrt_256, or none')
parser.add_argument('--des', type=str, default='Exp', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument(
    '--amp_dtype',
    type=str,
    default='auto',
    choices=['auto', 'float16', 'bfloat16'],
    help='With --use_amp: autocast dtype; auto uses bfloat16 when CUDA supports it (e.g. Hopper/Ampere), else float16.',
)
parser.add_argument('--comment', type=str, default='none', help='com')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')

# loss weight params
parser.add_argument('--lambda_recon', type=float, default=1, help='lambda patch')
parser.add_argument('--lambda_fft_align', type=float, default=0, help=(
    'Peak FFT Gram coefficient when fft_align_warmup_epochs>0 (value at end of warmup); '
    'otherwise base lambda (optionally gated by fft_align_epoch_*).'
))
parser.add_argument('--fft_align_epoch_start', type=int, default=-1,
                    help='enable fft align only from this 1-based epoch (inclusive); <=0 disables epoch gating')
parser.add_argument('--fft_align_epoch_end', type=int, default=-1,
                    help='enable fft align until this 1-based epoch (inclusive); '
                    'if fft_align_epoch_start>0 and this is <=0, run through final epoch (train_epochs); '
                    'if both start and end <=0, no epoch gating (use lambda_fft_align for all epochs)')
parser.add_argument('--fft_align_lambda_active', type=float, default=None,
                    help='fft align lambda inside [fft_align_epoch_start, fft_align_epoch_end]; None=use lambda_fft_align')
parser.add_argument('--fft_align_lambda_inactive', type=float, default=0.0,
                    help='fft align lambda outside gated epoch range (default 0.0)')
parser.add_argument(
    '--fft_align_warmup_epochs',
    type=int,
    default=0,
    help='If >0: ramp lambda_fft_align from fft_align_lambda_start to lambda_fft_align (peak) over '
    'this many 1-based epochs, then decay linearly to fft_align_lambda_end by train_epochs; '
    'ignores fft_align_epoch_start/end. If 0: use legacy epoch gate or constant lambda_fft_align.',
)
parser.add_argument(
    '--fft_align_lambda_start',
    type=float,
    default=0.0,
    help='FFT Gram coefficient at epoch 1 when fft_align_warmup_epochs>0 (before ramp to peak).',
)
parser.add_argument(
    '--fft_align_lambda_end',
    type=float,
    default=0.0,
    help='FFT Gram coefficient at final epoch after warmup when fft_align_warmup_epochs>0.',
)
parser.add_argument(
    '--fft_align_all_patches',
    action='store_true',
    help='Use legacy all-patch FFT Gram over the aligned student/teacher windows. '
    'Default (flag off): only masked patches (same support as iBOT / patch loss), as a spectral regularizer.',
)
parser.add_argument(
    '--fft_align_gram_mse_f_ref_bins',
    type=int,
    default=0,
    help='FFT Gram MSE 重标定：0=按 seq_len/patch/stride 得满长 F_ref，loss 乘 min(1,(F_act/F_ref)^2)，'
    '与 mean 随 F 变化解耦、长短 micro-step 更可比，满长 scale=1，不额外放大更长窗；<0 关闭',
)
parser.add_argument(
    '--fft_align_freq_keep_ratio',
    type=float,
    default=1.0,
    help='ratio-based low-frequency keep for FFT align Gram, applied on active bins F_act after DC removal (0,1]; 1.0 keeps all',
)
parser.add_argument(
    '--fft_align_freq_min_bins',
    type=int,
    default=1,
    help='minimum kept frequency bins for FFT align after ratio truncation',
)
parser.add_argument(
    '--fft_align_freq_max_bins',
    type=int,
    default=0,
    help='maximum kept frequency bins for FFT align after ratio truncation; <=0 means no upper bound',
)
parser.add_argument('--lambda_cls_proto', type=float, default=1, help='lambda cls proto')
parser.add_argument('--lambda_patch_proto', type=float, default=0.2, help='lambda patch proto')
parser.add_argument('--lambda_koleo', type=float, default=0.05, help='lambda koleo')
parser.add_argument('--lambda_temporal', type=float, default=0.05, help='lambda temporal')

# FFT / masking / imputator 相关额外控制
parser.add_argument(
    '--fft_align_min_valid_ratio',
    type=float,
    default=0.0,
    help='only compute FFT Gram (lambda_fft_align) when per-sample raw valid ratio '
    '(~missing_mask_orig) >= this; <=0 disables gating (use all valid_sample rows)',
)
parser.add_argument(
    '--valid_sample_threshold',
    type=float,
    default=0.0,
    help='TED_modular: per-sample raw valid ratio (~missing_mask_orig.mean) must be >= this '
    'to join DINO / iBOT / koleo / temporal losses; 0 keeps all samples',
)
# 方案二：控制 FFT 重建中使用的频率截止比例（RobustFreqLoss.cutoff_ratio）
parser.add_argument(
    '--fft_cutoff_ratio',
    type=float,
    default=0.5,
    help='frequency cutoff ratio for RobustFreqLoss (0-1, default 0.5 means use lowest half of frequency bins)',
)
# 方案三：DINOv3 风格 patch mask 中 block masking 的比例（剩余为随机 masking）
parser.add_argument(
    '--block_mask_ratio',
    type=float,
    default=0.8,
    help='ratio of block-masked patches in DINOv3-style masking (default 0.8)',
)
# 方案四：对 imputator 填补位置的 patch loss 降权
parser.add_argument(
    '--imputed_patch_weight',
    type=float,
    default=1.0,
    help='relative weight for patches that contain imputed (originally missing) positions in patch loss (1.0 = no reweighting)',
)
# iBOT：全缺 patch 是否进 loss（无 Imputator 或 imputator_mode 非 full 时；full 仍全参与）
parser.add_argument(
    '--ibot_patch_reliable_mode',
    type=str,
    default='filter',
    choices=['filter', 'keep_all'],
    help='filter: 仅 patch 内存在至少一处原始有效观测时参与 iBOT (patch_valid_ratio_orig>0，与旧版一致); keep_all: 不过滤',
)
parser.add_argument(
    '--lambda_cls_cons',
    type=float,
    default=0,
    help='weight for raw-imputed CLS consistency loss (same sequence, align CLS from raw view and imputed view); 0 to disable',
)

# supervised classification split / early stopping (LCMAP & GlanceTraining)
parser.add_argument('--cls_train_ratio', type=float, default=0.8,
                    help='train split ratio for supervised classification (LCMAP / GlanceTraining)')
parser.add_argument('--cls_split_seed', type=int, default=42,
                    help='random seed for train/test split in supervised classification')
parser.add_argument('--cls_few_shot_k', type=int, default=None,
                    help='few-shot probe: k samples per class for train (1=one-shot, 5=five-shot). None=use cls_train_ratio')
parser.add_argument('--cls_min_samples_per_class', type=int, default=10,
                    help='exclude classes with total count < this from probe training and metrics (default 10)')
parser.add_argument('--cls_patience', type=int, default=10,
                    help='early stopping patience (epochs) on train accuracy plateau')

# 下游 KNN 探针的统一 K（ExpProbe 中所有 knn_probe_* 将读取该值）
parser.add_argument('--probe_knn_k', type=int, default=1,
                    help='K used in all downstream KNN probes (classification + segmentation). Default 1 for few-shot.')
parser.add_argument(
    '--probe_cls_mode',
    type=str,
    default='gap0',
    choices=[
        'gap0',
        'auto_gap',
        'gap_soft',
        'attn_readout',
        'gap0_plus_mean',
        'mean',
        'concat',
        'fused',
        'knn_ensemble',
    ],
    help=(
        'Multi-CLS downstream readout for KNN: '
        'gap0=CLS#0 only; auto_gap=hard pick by token gap; '
        'gap_soft=soft Gaussian mix over 6 CLS (recommended fusion for full 732); '
        'attn_readout=patch-mean attention over CLS bank; '
        'gap0_plus_mean=concat [CLS#0, mean(bank)]; '
        'knn_ensemble=6 separate KNNs + majority vote; '
        'mean/concat/fused=baselines.'
    ),
)
parser.add_argument(
    '--probe_cls_fusion_tau',
    type=float,
    default=1.0,
    help='Temperature for gap_soft / attn_readout CLS fusion; larger -> more uniform mixing.',
)
parser.add_argument(
    '--probe_classification_tasks',
    type=str,
    default='lcmap',
    help=(
        "classification probe task subset: 'all' or comma-separated "
        "subset from {lcmap,glance,globaltree,cdl,cropharvest}, e.g. 'lcmap,glance,cropharvest'."
    ),
)
parser.add_argument(
    '--probe_segmentation_tasks',
    type=str,
    default='hansen,wildfire',
    help=(
        "segmentation probe task subset: 'all' or comma-separated "
        "subset from {hansen,wildfire,lcmapchange}, e.g. 'hansen,wildfire'."
    ),
)

# 模拟多云：在 probe 阶段 encode 前随机掩码一定比例的有效时间步（将整步置为 NaN）
parser.add_argument('--probe_cloud_mask_ratio', type=float, default=0.0,
                    help='simulate clouds: randomly mask this ratio of valid time steps before encode in KNN probe (set to NaN). Default 0.')
parser.add_argument('--probe_cloud_mask_seed', type=int, default=2026,
                    help='random seed for probe_cloud_mask_ratio. Used to make the cloud masking reproducible.')

# 训练吞吐：默认偏重墙钟速度；需要密日志或可恢复快照时把它们调回 1
parser.add_argument(
    '--train_log_interval',
    type=int,
    default=25,
    help=(
        'imputation TED 训练：每 N 个 batch 打印一行含各 loss 分量的明细（减少每步 CUDA 同步）。'
        '1 表示逐步打印（旧行为）；建议 25–100。train_loss_epoch 仍为每步一个标量用于 epoch 均值。'
    ),
)
parser.add_argument(
    '--train_step_log_enable',
    type=int,
    default=1,
    help='1: write per-step training metrics to JSONL file asynchronously (rank0 only). 0: disable.',
)
parser.add_argument(
    '--train_step_log_file',
    type=str,
    default='auto',
    help='per-step JSONL path; "auto" writes to logs/<model_id>_train_step_metrics.jsonl (rank0 only).',
)
parser.add_argument(
    '--train_step_console_log_enable',
    type=int,
    default=0,
    help='1: also print per-step detailed losses to stdout; 0: keep per-step logs only in JSONL.',
)
parser.add_argument(
    '--train_step_tensor_item_interval',
    type=int,
    default=25,
    help='convert Tensor log_vars to Python scalars every N steps (1=every step). Larger N reduces .item()/sync overhead; default 25.',
)
parser.add_argument(
    '--progress_log_interval',
    type=int,
    default=200,
    help=(
        'imputation TED 训练：每 N 个 batch 打印一次进度（LR/速度/剩余时间）。'
        '用于低频观察训练态；建议 100–500。'
    ),
)
parser.add_argument(
    '--checkpoint_epoch_save_interval',
    type=int,
    default=1,
    help=(
        'imputation 训练：rank0 每隔 N epoch 写入 checkpoint_epoch_*.pth（默认 1=每 epoch）；'
        'train loss 刷新 best 时的 checkpoint.pth 不受影响。大跑可设 5+ 减磁盘。'
    ),
)
parser.add_argument(
    '--train_state_save_interval',
    type=int,
    default=5,
    help=(
        'imputation 训练：每 N epoch 在各 rank 写 train_state_rank*.pth / train_state.pth；'
        '首 epoch 末与最后一个 epoch 必写。设为 1 即每 epoch 写完整恢复状态。'
    ),
)
parser.add_argument(
    '--ddp_find_unused_parameters',
    type=int,
    default=0,
    help='DDP（imputation）：1=启用 find_unused_parameters（更慢，排 unused 梯度）；0=关闭（推荐速度）。backward 报错时再设 1。',
)

# GPU
# 【修改】建议用 int 控制 bool 选项，更稳健
parser.add_argument('--use_gpu', type=int, default=1, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', type=int, default=1, help='use multiple gpus')
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

# de-stationary projector params
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                    help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

args = parser.parse_args()

# 转换 bool
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
args.use_multi_gpu = True if torch.cuda.is_available() and args.use_multi_gpu else False
args.probe = bool(args.probe)
args.use_pretrained_imputator = bool(args.use_pretrained_imputator)
args.save_tsne_embeddings = bool(getattr(args, 'save_tsne_embeddings', 0))

# --- DDP 初始化核心逻辑 ---
if args.use_multi_gpu:
    # [DDP 模式] 检查是否在分布式环境中
    # 如果使用 torchrun 启动，环境变量 RANK 和 LOCAL_RANK 会被自动设置
    if "RANK" in os.environ and "LOCAL_RANK" in os.environ:
        # [DDP 模式] torchrun 会自动注入 LOCAL_RANK
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        args.device = torch.device("cuda", local_rank)
        args.local_rank = local_rank
    else:
        # 如果没有使用 torchrun，回退到单卡模式
        print("Warning: use_multi_gpu is True but RANK/LOCAL_RANK not set. Falling back to single GPU mode.")
        print("To use multi-GPU training, please use: torchrun --nproc_per_node=N run.py ...")
        args.use_multi_gpu = False  # 回退到单卡模式
        args.local_rank = 0
        if args.use_gpu and torch.cuda.is_available():
            device_id = args.gpu
            torch.cuda.set_device(device_id)
            args.device = torch.device(f"cuda:{device_id}")
        else:
            args.device = torch.device("cpu")
else:
    # [单卡模式] 读取 --gpu 参数指定的 ID
    args.local_rank = 0
    if args.use_gpu and torch.cuda.is_available():
        # 这里使用 args.gpu (默认为0，或者由命令行指定)
        device_id = args.gpu
        torch.cuda.set_device(device_id) # 最好也加上这句，设置当前默认设备
        args.device = torch.device(f"cuda:{device_id}")
    else:
        args.device = torch.device("cpu")

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    # DDP 模式下 args.gpu 参数其实用处不大了，因为用 local_rank 控制
    args.gpu = args.local_rank 

# 打印 GPU 信息（仅主进程打印）
if args.local_rank == 0:
    print('cuda.is_available:', torch.cuda.is_available())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print('Args in experiment:')
    print(args)

if args.task_name != 'imputation':
    raise ValueError(
        f"Unsupported task_name={args.task_name!r}. "
        "Only task_name='imputation' (Exp_SSL) is supported. "
        "Removed entry points: long_term_forecast, short_term_forecast, "
        "classification (Exp_Classification), anomaly_detection. "
        "For supervised LCMAP-style Patch_Masked_Cls training or RF baselines, "
        "see run_classification_comparison.py or restore deleted exp modules from git history."
    )
Exp = Exp_SSL

# 设置实验记录
setting = '{}_{}_{}_{}_{}_sl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
    args.task_name,
    args.model_id,
    args.comment,
    args.model,
    args.data,
    args.seq_len,
    args.pred_len,
    args.d_model,
    args.n_heads,
    args.e_layers,
    args.d_layers,
    args.d_ff,
    args.factor,
    args.embed,
    args.distil,
    args.des, 0)

_setting_raw = setting
setting = clamp_experiment_setting_for_checkpoint(setting)
if setting != _setting_raw:
    args.checkpoint_setting_full = _setting_raw
else:
    args.checkpoint_setting_full = None

exp = Exp(args)  # set experiments

# 根据 mode 执行相应操作；退出前统一销毁 DDP 进程组，避免 NCCL 资源泄漏
try:
    if args.mode == 'train':
        for ii in range(args.itr):
            # 更新 setting 中的迭代编号
            setting = setting.rsplit('_', 1)[0] + f'_{ii}'
            if args.local_rank == 0:
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            
            exp.train(setting)
            
            if int(getattr(args, "max_train_steps", 0) or 0) > 0:
                if args.local_rank == 0:
                    print(
                        "[Smoke] max_train_steps>0: skip post-train test() (no full checkpoint expected).",
                        flush=True,
                    )
            else:
                if args.local_rank == 0:
                    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                
                exp.test(setting)
            torch.cuda.empty_cache()

    elif args.mode == 'fine-tune':
        for ii in range(args.itr):
            setting = setting.rsplit('_', 1)[0] + f'_{ii}'
            if args.local_rank == 0:
                print('>>>>>>>start fine-tuning : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            
            exp.fine_tuning(setting)
            
            if args.local_rank == 0:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            
            exp.test(setting)
            torch.cuda.empty_cache()

    elif args.mode == 'test':
        if args.local_rank == 0:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()

    elif args.mode == 'pred':
        if args.local_rank == 0:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.pred(setting)
        torch.cuda.empty_cache()

    elif args.mode == 'visualize':
        # 可视化模式只在主进程执行（避免多进程冲突）
        if args.local_rank == 0:
            print('>>>>>>>visualizing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.visualize(
                setting,
                checkpoint_path=args.visualize_checkpoint_path,
                num_samples=args.visualize_num_samples
            )
        torch.cuda.empty_cache()

    elif args.mode == 'vali_imputator':
        # 仅用于验证 imputator 质量（绘图），只在 rank 0 执行核心逻辑
        if args.local_rank == 0:
            print('>>>>>>>vali_imputator (imputator visualization) : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # 仅支持 imputation 任务
            if args.task_name != 'imputation':
                raise ValueError("mode=vali_imputator 目前仅支持 task_name='imputation'")
            exp.expValiImputator(
                numSamples=args.imputator_vali_num_samples,
                saveDir=args.imputator_vali_save_dir,
                seed=args.imputator_vali_seed,
            )
        # 其他 rank 同步后直接退出
        torch.cuda.empty_cache()

    else:
        raise ValueError(f"Invalid mode: {args.mode}. Expected 'train', 'fine-tune', 'test', 'pred', 'visualize', or 'vali_imputator'.")
finally:
    # DDP：程序退出前销毁进程组，避免 "destroy_process_group() was not called" 警告与资源泄漏
    if args.use_multi_gpu and dist.is_initialized():
        dist.destroy_process_group()
