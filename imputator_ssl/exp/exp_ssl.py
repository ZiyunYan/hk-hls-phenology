import os
import json
import random
import time
import warnings
import threading
import queue

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.cuda.amp import GradScaler

from utils.amp_utils import amp_autocast_ctx, resolve_amp_dtype
from torch.nn.parallel import DistributedDataParallel as DDP

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from exp.exp_rs_application import ExpProbe
from models.Transformer import load_pretrained_imputator
from utils.losses import DINOCriteria
from utils.schedulers import apply_optim_scheduler, build_schedulers
from utils.tools import save_model_periodically, EarlyStopping, resolve_fft_align_lambda_for_epoch


warnings.filterwarnings('ignore')

"""
自监督 / 表征预训练主循环（TED_modular、Patch_Masked 等）。
与「插值」任务无必然关系；CLI 仍沿用 task_name='imputation' 仅为历史兼容。
"""


def get_total_params(model):
    """计算模型总参数量"""
    # 处理 nn.DataParallel 包裹的模型
    model = model.module if isinstance(model, nn.DataParallel) else model
    return sum(p.numel() for p in model.parameters())

class Exp_SSL(Exp_Basic):
    def __init__(self, args):
        super(Exp_SSL, self).__init__(args)
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.probe = ExpProbe(self.device)
        
        # --- 1. 初始化 Loss 模块 (全局唯一，共享状态) ---
        self.criterion = DINOCriteria(self.args, self.device)

    def _compute_loss(self, outputs):
        """
        根据模型输出计算 loss。若为 Patch_Masked / Patch_NTP_TED 等 SSL 模型，
        直接使用 outputs['ssl_loss']；否则使用 DINOCriteria。
        """
        if 'ssl_loss' in outputs:
            loss = outputs['ssl_loss']
            log_vars = outputs.get('log_vars', {})
            return loss, log_vars
        return self.criterion(outputs)

    def _build_model(self):
        # 1. 初始化模型
        model = self.model_dict[self.args.model].Model(self.args).float()
        
        # 2. 移动到指定 GPU
        model.to(self.device)

        # 3. DDP 包裹
        if self.args.use_multi_gpu:
            # SyncBN (如果有 BatchNorm 层)
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            fu = bool(int(getattr(self.args, 'ddp_find_unused_parameters', 0)))
            if self.args.local_rank == 0 and not fu:
                print('[DDP] find_unused_parameters=False (speed); set --ddp_find_unused_parameters 1 if backward complains)')
            model = DDP(model, device_ids=[self.args.local_rank], find_unused_parameters=fu)

        return model

    @staticmethod
    def _normalize_compiled_state_dict_keys(state_dict):
        """
        兼容 torch.compile 产生的 `_orig_mod` 前缀。
        仅处理编译相关前缀，不主动处理 DDP 的 `module.`（由调用方按场景决定）。
        """
        if not isinstance(state_dict, dict):
            return state_dict, False

        normalized = {}
        changed = False
        for k, v in state_dict.items():
            nk = k
            if nk.startswith('_orig_mod.'):
                nk = nk[len('_orig_mod.'):]
                changed = True
            while '._orig_mod.' in nk:
                nk = nk.replace('._orig_mod.', '.')
                changed = True
            normalized[nk] = v
        return normalized, changed

    def _get_data(self, flag, data=None):
        data_set, data_loader = data_provider(self.args, flag, data)
        return data_set, data_loader

    def _select_optimizer(self):
        # 分组参数：Regular (decay) vs No-Decay (bias, norm, head)
        regular_groups = []
        noreg_groups = []
        last_layer_groups = []  # 最后一层（需要特殊处理）
        
        # 这里的 self.model 已经被 DDP 包裹，所以要用 .named_parameters() 遍历
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            is_last_layer = "last_layer.weight" in name
            
            # 1. Bias, LayerNorm, BatchNorm 不 decay
            if name.endswith(".bias") or ".bn" in name or ".norm" in name:
                if is_last_layer:
                    last_layer_groups.append({
                        'params': [param],
                        'is_last_layer': True,
                        'weight_decay': 0.0,
                        'scheduled_weight_decay': False,
                    })
                else:
                    noreg_groups.append(param)
            # 2. DINO/iBOT Head 的最后一层 (Prototypes) 绝不能 decay
            elif is_last_layer:
                last_layer_groups.append({
                    'params': [param],
                    'is_last_layer': True,
                    'weight_decay': 0.0,
                    'scheduled_weight_decay': False,
                })
            # 3. Embedding 和 Token 也不 decay
            elif "embedding" in name or "cls_token" in name or "mask_token" in name:
                noreg_groups.append(param)
            else:
                regular_groups.append(param)
        
        param_groups = [
            {
                'params': regular_groups,
                'weight_decay': self.args.weight_decay,
                'is_last_layer': False,
                'scheduled_weight_decay': True,
            },
            {
                'params': noreg_groups,
                'weight_decay': 0.0,
                'is_last_layer': False,
                'scheduled_weight_decay': False,
            },
        ]
        
        # 添加最后一层组
        param_groups.extend(last_layer_groups)
        
        # 推荐使用 AdamW 以正确应用 Weight Decay
        model_optim = optim.AdamW(param_groups, lr=self.args.learning_rate) 
        return model_optim

    def load_pretrained_model(self, checkpoint_path=None):
        """
        加载预训练模型权重。
        
        Args:
            checkpoint_path: checkpoint文件路径（可选）
                - 如果为None，则从 args.pretrain_model 目录下查找 'checkpoint.pth'
                - 如果为字符串，则直接使用该路径（可以是文件路径或目录路径）
                - 如果是目录，会在目录下查找 'checkpoint.pth'
                - 如果是文件，直接加载该文件
        """
        # 确定checkpoint路径
        if checkpoint_path is None:
            if self.args.pretrain_model is None:
                raise ValueError("args.pretrain_model 未设置，无法加载预训练模型")
            
            # 检查是文件还是目录
            if os.path.isfile(self.args.pretrain_model):
                checkpoint_path = self.args.pretrain_model
            elif os.path.isdir(self.args.pretrain_model):
                checkpoint_path = os.path.join(self.args.pretrain_model, 'checkpoint.pth')
            else:
                raise FileNotFoundError(f"预训练模型路径不存在: {self.args.pretrain_model}")
        else:
            # 如果提供了checkpoint_path参数，检查是文件还是目录
            if os.path.isfile(checkpoint_path):
                # 直接是文件路径
                pass
            elif os.path.isdir(checkpoint_path):
                # 是目录，查找checkpoint.pth
                checkpoint_path = os.path.join(checkpoint_path, 'checkpoint.pth')
            else:
                raise FileNotFoundError(f"Checkpoint路径不存在: {checkpoint_path}")
        
        # 检查文件是否存在
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_path}")
        
        if self.args.local_rank == 0:
            print(f"正在加载预训练权重: {checkpoint_path}") 
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"加载checkpoint失败: {e}")

        # 处理checkpoint可能是字典格式（包含'model'键）的情况
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            checkpoint = checkpoint['model']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        state_dict, changed_compile = self._normalize_compiled_state_dict_keys(checkpoint)
        if changed_compile and self.args.local_rank == 0:
            print("[LoadPretrained] 检测到 compiled checkpoint，已自动移除 _orig_mod 前缀。")

        model_has_module = hasattr(self.model, 'module')
        checkpoint_has_module = any(k.startswith('module.') for k in state_dict.keys())

        if model_has_module and not checkpoint_has_module:
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        elif not model_has_module and checkpoint_has_module:
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

        # 获取当前模型的state_dict，用于检查形状
        model_state_dict = self.model.state_dict()
        
        # 过滤参数：跳过decoder相关和形状不匹配的位置编码参数
        # 下游任务（探针任务）不需要decoder，所以可以跳过所有decoder相关参数
        filtered_state_dict = {}
        skipped_keys = []
        
        # 需要跳过的参数模式（下游任务不需要这些）
        skip_patterns = [
            'decoder_pos_embed',           # decoder位置编码
            'decoder.',                    # decoder层（所有decoder相关的层）
            'pixel_decoder',               # 像素级decoder（用于重建）
        ]
        
        # 位置编码参数（支持动态处理，形状不匹配时跳过）
        position_encoding_keys = ['position_encoding.pos_table']
        
        for k, v in state_dict.items():
            # 检查是否是decoder相关的参数（下游任务不需要）
            is_decoder_related = any(pattern in k for pattern in skip_patterns)
            
            if is_decoder_related:
                # 跳过所有decoder相关的参数
                skipped_keys.append(k)
                # 不打印每个跳过的decoder参数，只在最后统计
            elif k in model_state_dict:
                # 检查是否是位置编码相关的参数
                is_pos_encoding = any(pos_key in k for pos_key in position_encoding_keys)
                
                if is_pos_encoding and v.shape != model_state_dict[k].shape:
                    # 跳过形状不匹配的位置编码参数（forward时已支持动态处理）
                    skipped_keys.append(k)
                    if self.args.local_rank == 0:
                        print(f"跳过位置编码参数（形状不匹配，forward时支持动态处理）: {k} "
                              f"checkpoint shape: {v.shape}, model shape: {model_state_dict[k].shape}")
                elif v.shape == model_state_dict[k].shape:
                    # 形状匹配，可以加载
                    filtered_state_dict[k] = v
                else:
                    # 其他形状不匹配的参数也跳过
                    skipped_keys.append(k)
                    if self.args.local_rank == 0:
                        print(f"跳过参数（形状不匹配）: {k} "
                              f"checkpoint shape: {v.shape}, model shape: {model_state_dict[k].shape}")
            else:
                # 模型中没有这个key，跳过
                skipped_keys.append(k)

        load_result = self.model.load_state_dict(filtered_state_dict, strict=False)

        if self.args.local_rank == 0:
            matched_keys_count = len(filtered_state_dict) - len(load_result.missing_keys)
            total_keys = len(state_dict)
            decoder_skipped = sum(1 for k in skipped_keys if any(p in k for p in ['decoder', 'pixel_decoder']))
            pos_encoding_skipped = sum(1 for k in skipped_keys if 'position_encoding' in k or 'decoder_pos_embed' in k)
            other_skipped = len(skipped_keys) - decoder_skipped - pos_encoding_skipped
            
            print(f"参数加载统计: {matched_keys_count}/{total_keys} 键匹配成功")
            if decoder_skipped > 0:
                print(f"跳过的decoder相关参数: {decoder_skipped} (下游任务不需要decoder)")
            if pos_encoding_skipped > 0:
                print(f"跳过的位置编码参数: {pos_encoding_skipped} (形状不匹配，forward时支持动态处理)")
            if other_skipped > 0:
                print(f"跳过的其他参数: {other_skipped} (形状不匹配或模型中没有)")
            if len(load_result.missing_keys) > 0:
                print(f"缺失键数量: {len(load_result.missing_keys)} (前3个: {load_result.missing_keys[:3]})")
            if len(load_result.unexpected_keys) > 0:
                print(f"意外键数量: {len(load_result.unexpected_keys)} (前3个: {load_result.unexpected_keys[:3]})")

    def _run_downstream_knn_tasks(self, probe, n_neighbors: int, imputator=None):
        """
        统一分发下游 KNN Probe 任务，避免训练/测试路径重复实现而出现不一致。
        支持 downstream_group: classification / segmentation / all。
        """
        task_group = str(getattr(self.args, 'downstream_group', 'classification')).strip().lower()
        valid_groups = {'classification', 'segmentation', 'all'}
        if task_group not in valid_groups:
            print(
                f">>> [Warning] unknown downstream_group='{task_group}', "
                "fallback to 'classification'."
            )
            task_group = 'classification'

        if task_group in {'classification', 'all'}:
            cls_tasks_raw = str(getattr(self.args, 'probe_classification_tasks', 'all')).strip().lower()
            if cls_tasks_raw in {'', 'all'}:
                cls_tasks = {'lcmap', 'glance', 'globaltree', 'cdl', 'cropharvest'}
            else:
                cls_tasks = {x.strip() for x in cls_tasks_raw.split(',') if x.strip()}
                valid = {'lcmap', 'glance', 'globaltree', 'cdl', 'cropharvest'}
                unknown = sorted(cls_tasks - valid)
                if unknown:
                    print(f">>> [Warning] unknown probe_classification_tasks={unknown}, ignored.")
                cls_tasks = cls_tasks & valid
                if not cls_tasks:
                    print(">>> [Warning] empty probe_classification_tasks after filtering, fallback to all.")
                    cls_tasks = valid

            if 'lcmap' in cls_tasks:
                probe.knn_probe_LCMAP_Classification(self.model, self.args, n_neighbors=n_neighbors, imputator=imputator)
            if 'glance' in cls_tasks:
                probe.knn_probe_GlanceTraining_Classification(self.model, self.args, n_neighbors=n_neighbors, imputator=imputator)
            if 'globaltree' in cls_tasks:
                probe.knn_probe_GlobalTree_Classification(self.model, self.args, n_neighbors=n_neighbors, imputator=imputator)
            if 'cdl' in cls_tasks:
                probe.knn_probe_CDL_Classification(self.model, self.args, n_neighbors=n_neighbors, imputator=imputator)
            if 'cropharvest' in cls_tasks:
                probe.knn_probe_CropHarvest_Classification(self.model, self.args, n_neighbors=n_neighbors, imputator=imputator)

        if task_group in {'segmentation', 'all'}:
            seg_tasks_raw = str(getattr(self.args, 'probe_segmentation_tasks', 'all')).strip().lower()
            if seg_tasks_raw in {'', 'all'}:
                seg_tasks = {'hansen', 'wildfire', 'lcmapchange'}
            else:
                seg_tasks = {x.strip() for x in seg_tasks_raw.split(',') if x.strip()}
                valid = {'hansen', 'wildfire', 'lcmapchange'}
                unknown = sorted(seg_tasks - valid)
                if unknown:
                    print(f">>> [Warning] unknown probe_segmentation_tasks={unknown}, ignored.")
                seg_tasks = seg_tasks & valid
                if not seg_tasks:
                    print(">>> [Warning] empty probe_segmentation_tasks after filtering, fallback to all.")
                    seg_tasks = valid

            if 'hansen' in seg_tasks:
                probe.knn_probe_Hansen_Segmentation(self.model, self.args, n_neighbors=n_neighbors, imputator=imputator)
            if 'wildfire' in seg_tasks:
                probe.knn_probe_Wildfire_Segmentation(self.model, self.args, n_neighbors=n_neighbors, imputator=imputator)
            if 'lcmapchange' in seg_tasks:
                probe.knn_probe_LCMAPChange_Segmentation(self.model, self.args, n_neighbors=n_neighbors, imputator=imputator)

    def run_probe(self, imputator=None, epoch=None):
        """
        仅运行 KNN Probe 评估（DINO 系列无需验证集 loss）。
        epoch: 当前 epoch（0-based），用于按 probe_interval 控制运行频率；若为 None 则每次调用都运行。
        """
        if self.args.use_multi_gpu:
            dist.barrier()

        probe_interval = getattr(self.args, 'probe_interval', 1)
        run_probe_this_epoch = self.args.probe and self.args.local_rank == 0 and (
            epoch is None or (epoch + 1) % max(1, probe_interval) == 0
        )
        if run_probe_this_epoch:
            print("\n>>> [Test] Starting KNN Probe Evaluation...")
            try:
                probe = ExpProbe(self.device)
                knn_k = int(getattr(self.args, 'probe_knn_k', 5))
                self._run_downstream_knn_tasks(probe=probe, n_neighbors=knn_k, imputator=imputator)
                print(f">>> [Test] KNN Probe Evaluation completed")
            except Exception as e:
                print(f">>> [Warning] KNN Probe Evaluation failed: {e}\n")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self.args.use_multi_gpu:
            dist.barrier()

    def _get_rng_state(self):
        """保存当前进程的 RNG 状态，用于断点续训时尽量保持随机序列一致。"""
        rngState = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rngState['torch_cuda_all'] = torch.cuda.get_rng_state_all()
        return rngState

    def _set_rng_state(self, rngState):
        """恢复当前进程的 RNG 状态。"""
        if not isinstance(rngState, dict):
            return
        try:
            if 'python' in rngState:
                random.setstate(rngState['python'])
            if 'numpy' in rngState:
                np.random.set_state(rngState['numpy'])
            if 'torch_cpu' in rngState:
                torch.set_rng_state(rngState['torch_cpu'])
            if torch.cuda.is_available() and 'torch_cuda_all' in rngState:
                torch.cuda.set_rng_state_all(rngState['torch_cuda_all'])
        except Exception as e:
            if getattr(self.args, 'local_rank', 0) == 0:
                print(f"[Resume] RNG state restore warning: {e}")

    def _resolve_resume_path(self, resumePath):
        """解析 resume 路径。支持传目录或文件；DDP 下优先 rank 专属文件。"""
        if resumePath is None:
            return None
        if isinstance(resumePath, str) and resumePath.strip().lower() in ('', 'none', 'null'):
            return None

        rankId = int(getattr(self.args, 'local_rank', 0))
        isDdp = bool(getattr(self.args, 'use_multi_gpu', False))

        if os.path.isdir(resumePath):
            if isDdp:
                rankPath = os.path.join(resumePath, f"train_state_rank{rankId}.pth")
                if os.path.exists(rankPath):
                    return rankPath
            singlePath = os.path.join(resumePath, "train_state.pth")
            if os.path.exists(singlePath):
                return singlePath
            return None

        if os.path.isfile(resumePath):
            if isDdp:
                baseDir = os.path.dirname(resumePath)
                rankPath = os.path.join(baseDir, f"train_state_rank{rankId}.pth")
                if os.path.exists(rankPath):
                    return rankPath
            return resumePath

        return None

    def train(self, setting):
        # --- 1. 数据加载与环境初始化 ---
        train_data, train_loader = self._get_data(flag='train')

        if self.args.local_rank == 0:
            total_params = get_total_params(self.model)
            print(f"模型参数: {total_params:,} ({total_params / 1e6:.2f}M)")
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and self.args.local_rank == 0:
            os.makedirs(path)
            full_name = getattr(self.args, "checkpoint_setting_full", None)
            if full_name:
                try:
                    meta_path = os.path.join(path, "setting_full.txt")
                    with open(meta_path, "w", encoding="utf-8") as f:
                        f.write(full_name + "\n")
                except OSError as e:
                    print(f"[Checkpoints] Could not write setting_full.txt: {e}")

        # rank0: 每 step 全量指标异步写入 JSONL
        step_log_queue = None
        step_log_thread = None
        step_log_stop_token = object()
        step_log_drop_counter = 0
        if self.args.local_rank == 0 and bool(int(getattr(self.args, "train_step_log_enable", 1))):
            step_log_file_cfg = str(getattr(self.args, "train_step_log_file", "auto"))
            if step_log_file_cfg == "auto":
                logs_dir = os.path.join(".", "logs")
                os.makedirs(logs_dir, exist_ok=True)
                safe_model_id = str(getattr(self.args, "model_id", "train")).replace("/", "_")
                step_log_file = os.path.join(logs_dir, f"{safe_model_id}.jsonl")
            elif os.path.isabs(step_log_file_cfg):
                step_log_file = step_log_file_cfg
            else:
                step_log_file = os.path.join(".", step_log_file_cfg)
                parent = os.path.dirname(step_log_file)
                if parent:
                    os.makedirs(parent, exist_ok=True)
            step_log_queue = queue.Queue(maxsize=8192)

            def _step_log_worker():
                with open(step_log_file, "a", encoding="utf-8") as fout:
                    while True:
                        item = step_log_queue.get()
                        if item is step_log_stop_token:
                            break
                        fout.write(json.dumps(item, ensure_ascii=False) + "\n")

            step_log_thread = threading.Thread(
                target=_step_log_worker,
                name="train-step-jsonl-writer",
                daemon=True,
            )
            step_log_thread.start()

        time_now = time.time()
        train_steps = len(train_loader)
        total_iterations = train_steps * self.args.train_epochs  # 计算总迭代次数，用于渐进学习
        
        # Imputator 使用早停（基于 train loss），TED/DINO 类模型不使用
        is_imputator = (self.args.model in ['Transformer'])
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True) if is_imputator else None
        scaler = GradScaler() 
        model_optim = self._select_optimizer()

        if self.args.local_rank == 0 and self.args.use_amp and torch.cuda.is_available():
            _dt = resolve_amp_dtype(self.args)
            print(
                f"[AMP] amp_dtype={getattr(self.args, 'amp_dtype', 'auto')} -> autocast {_dt}"
                if _dt is not None
                else "[AMP] disabled"
            )

        # --- 2. DINOv3 风格调度器初始化 ---
        # 构建所有调度器（LR, WD, Momentum, Teacher Temp）
        lr_schedule, wd_schedule, momentum_schedule, teacher_temp_schedule, last_layer_lr_schedule = build_schedulers(
            self.args, train_steps
        )
        
        if self.args.local_rank == 0:
            print(f"Total iterations: {total_iterations}")
            print(f"Steps per epoch: {train_steps}")
            print(f"scheduler_version: {getattr(self.args, 'scheduler_version', 'cosine')}")
            print(f"Warmup epochs: {self.args.warmup_epochs}")
            print(f"LR: {self.args.learning_rate} -> {getattr(self.args, 'min_lr', self.args.learning_rate * 1e-6)}")
            print(f"WD: {self.args.weight_decay} -> {getattr(self.args, 'weight_decay_end', self.args.weight_decay * 10)}")
            print(f"Momentum: {getattr(self.args, 'momentum_teacher', 0.992)} -> {getattr(self.args, 'final_momentum_teacher', 1.0)}")
            print(f"Teacher Temp: {getattr(self.args, 'warmup_teacher_temp', 0.04)} -> {getattr(self.args, 'teacher_temp', 0.07)}")
            print(
                "Loss λ: fft_align={} cls={} patch={} koleo={} temporal={} cls_cons={}".format(
                    getattr(self.args, "lambda_fft_align", 0),
                    getattr(self.args, "lambda_cls_proto", 0),
                    getattr(self.args, "lambda_patch_proto", 0),
                    getattr(self.args, "lambda_koleo", 0),
                    getattr(self.args, "lambda_temporal", 0),
                    getattr(self.args, "lambda_cls_cons", 0),
                )
            )
            # 打印渐进学习策略
            curriculum_strategy = getattr(self.model.module if hasattr(self.model, 'module') else self.model, 'curriculum_strategy', 'fast')
            print(f"Curriculum Strategy: {curriculum_strategy}")

        if self.args.local_rank == 0:
            print(f"use_pretrained_imputator: {self.args.use_pretrained_imputator}")
            
        if self.args.use_pretrained_imputator:
            # imputator 不参与训练，仅用于辅助 model；DDP 时由 load_pretrained_imputator 按 local_rank 放到各进程对应 GPU
            self.imputator = load_pretrained_imputator(
                self.args,
                self.args.pretrained_imputator_path,
                target_device=getattr(self.args, 'device', None),  # 单卡时用 args.device，DDP 时会被忽略
                use_ddp=bool(self.args.use_multi_gpu),
                local_rank=getattr(self.args, 'local_rank', None) if self.args.use_multi_gpu else None,
            )
            if self.args.local_rank == 0:
                imp_dev = next(self.imputator.parameters()).device
                print(f"load pretrained Imputator success (on {imp_dev}, DDP 时各进程在各自 GPU)")
        else:
            self.imputator = None

        if self.args.pretrain_model is not None:
            self.load_pretrained_model()

        # --- 3. 断点恢复（训练态） ---
        start_epoch = 0
        best_train_loss = float('inf')
        resume_path = self._resolve_resume_path(getattr(self.args, 'resume_checkpoint', None))
        if resume_path is not None:
            resume_state = torch.load(resume_path, map_location='cpu', weights_only=False)
            model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
            resume_model_state = resume_state.get('model', {})
            resume_model_state, changed_compile = self._normalize_compiled_state_dict_keys(resume_model_state)
            # train_state 是按 self.model.module 保存的（不含 module.），但这里兜底处理一下
            if any(k.startswith('module.') for k in resume_model_state.keys()):
                resume_model_state = {
                    (k.replace('module.', '', 1) if k.startswith('module.') else k): v
                    for k, v in resume_model_state.items()
                }
            try:
                model_to_load.load_state_dict(resume_model_state)
            except RuntimeError as e:
                raise RuntimeError(
                    f"[Resume] 模型权重恢复失败: {e}\\n"
                    f"resume_path={resume_path}"
                )
            if changed_compile and self.args.local_rank == 0:
                print("[Resume] 检测到 compiled train_state，已自动移除 _orig_mod 前缀后恢复成功。")
            if 'optimizer' in resume_state:
                model_optim.load_state_dict(resume_state['optimizer'])
                # 避免 optimizer state 留在 CPU，确保恢复后继续在当前 device 上训练
                for st in model_optim.state.values():
                    for k, v in st.items():
                        if torch.is_tensor(v):
                            st[k] = v.to(self.device)
            if 'scaler' in resume_state and isinstance(resume_state['scaler'], dict):
                scaler.load_state_dict(resume_state['scaler'])
            best_train_loss = float(resume_state.get('best_train_loss', float('inf')))
            start_epoch = int(resume_state.get('next_epoch', 0))
            self._set_rng_state(resume_state.get('rng_state', None))
            if self.args.local_rank == 0:
                print(
                    f"[Resume] Loaded training state from {resume_path} | "
                    f"start_epoch={start_epoch}, best_train_loss={best_train_loss:.7f}"
                )
        elif self.args.local_rank == 0 and getattr(self.args, 'resume_checkpoint', None):
            print(f"[Resume] resume_checkpoint not found: {self.args.resume_checkpoint}, start from scratch.")

        if self.args.use_multi_gpu:
            dist.barrier()

        # --- 4. 训练循环 ---
        def _save_train_state(nextEpoch, globalIter):
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            state_payload = {
                'model': model_to_save.state_dict(),
                'optimizer': model_optim.state_dict(),
                'scaler': scaler.state_dict(),
                'best_train_loss': best_train_loss,
                'next_epoch': int(nextEpoch),
                'global_iter': int(globalIter),
                'rng_state': self._get_rng_state(),
                'setting': setting,
            }
            if self.args.use_multi_gpu:
                save_path = os.path.join(path, f"train_state_rank{self.args.local_rank}.pth")
            else:
                save_path = os.path.join(path, "train_state.pth")
            torch.save(state_payload, save_path)

        max_train_steps = int(getattr(self.args, 'max_train_steps', 0) or 0)
        smoke_steps_done = 0
        for epoch in range(start_epoch, self.args.train_epochs):
            iter_count = 0
            train_loss_epoch = []
            self.model.train()
            epoch_time = time.time()
            # Optional epoch-gated FFT align (1-based epoch indexing).
            model_ref = self.model.module if hasattr(self.model, 'module') else self.model
            if hasattr(model_ref, "lambda_fft_align"):
                ep1 = int(epoch + 1)
                target_fft, fft_info = resolve_fft_align_lambda_for_epoch(ep1, self.args)
                model_ref.lambda_fft_align = float(target_fft)
                if self.args.local_rank == 0:
                    if fft_info.get("mode") == "warmup":
                        print(
                            f"[FFT Align Warmup] epoch={ep1}/{self.args.train_epochs} "
                            f"lambda_fft_align={float(target_fft):.6f} "
                            f"(warmup_epochs={fft_info['warmup_epochs']}, "
                            f"start={fft_info['lambda_start']:.6f}, peak={fft_info['lambda_peak']:.6f}, "
                            f"end={fft_info['lambda_end']:.6f})",
                            flush=True,
                        )
                    elif fft_info.get("mode") == "gate":
                        print(
                            f"[FFT Align Gate] epoch={ep1} lambda_fft_align={float(target_fft):.6f} "
                            f"(start={fft_info['start_ep']}, end={fft_info['end_ep']}, "
                            f"active={fft_info['active']:.6f}, inactive={fft_info['inactive']:.6f})",
                            flush=True,
                        )
                    else:
                        print(
                            f"[FFT Align] epoch={ep1} lambda_fft_align={float(target_fft):.6f} "
                            f"(constant; set fft_align_warmup_epochs or fft_align_epoch_start for scheduling)",
                            flush=True,
                        )

            # DDP: IterableDataset 不需要 set_epoch，因为我们在 __iter__ 里处理了 shuffle
            # if self.args.use_multi_gpu: train_loader.sampler.set_epoch(epoch)
            
            # 关键修复：记录每个进程实际处理的 batch 数量，用于调试
            actual_batches = 0
            stop_smoke = False

            global_iter = epoch * train_steps - 1
            for i, batch_tuple in enumerate(train_loader):
                batch_x, batch_x_mark, next_batch_x = batch_tuple[0], batch_tuple[1], batch_tuple[2]
                batch_lon_lat = batch_tuple[3] if len(batch_tuple) > 3 else None
                actual_batches += 1
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                if next_batch_x is not None:
                    next_batch_x = next_batch_x.float().to(self.device, non_blocking=True)
                if batch_lon_lat is not None:
                    batch_lon_lat = batch_lon_lat.float().to(self.device, non_blocking=True)
                
                iter_count += 1
                
                # 计算当前全局迭代次数
                global_iter = epoch * train_steps + i
                
                # 获取当前迭代的参数值
                current_lr = lr_schedule[global_iter]
                current_wd = wd_schedule[global_iter]
                current_momentum = momentum_schedule[global_iter]
                current_teacher_temp = teacher_temp_schedule[global_iter]
                current_last_layer_lr = last_layer_lr_schedule[global_iter]
                
                # 应用调度器到优化器
                apply_optim_scheduler(model_optim, current_lr, current_wd, current_last_layer_lr)
                
                model_optim.zero_grad(set_to_none=True)

                # 【梯度累积策略】：检查是否使用mixed_batch策略
                curriculum_strategy = getattr(self.model.module if hasattr(self.model, 'module') else self.model, 'curriculum_strategy', 'fast')
                use_gradient_accumulation = (curriculum_strategy == 'mixed_batch')
                
                gradAccumDivisor = None
                mixed_batch_effective_total = None
                if use_gradient_accumulation:
                    # 梯度累积模式：将 batch 沿样本维拆成多组，每组独立 forward+backward
                    # 注意：时间维裁剪由 TED_modular._forward 在 imputator 之后完成（保证位置编码对齐），
                    #       这里只做 batch 维拆分，不做时间维裁剪。
                    B, T, C = batch_x.shape
                    device = batch_x.device
                    weight_by_nv = bool(int(getattr(self.args, 'mixed_batch_weight_by_valid_samples', 1)))
                    
                    min_group_size = 32
                    max_groups = max(1, int(getattr(self.args, 'mixed_batch_groups', 2)))
                    num_groups = min(max_groups, max(1, B // min_group_size))
                    
                    if i == 0 and epoch == 0 and self.args.local_rank == 0:
                        print(
                            f"[Gradient Accumulation] Batch size: {B}, Groups: {num_groups}, "
                            f"Group sizes: {[B // num_groups + (1 if j < B % num_groups else 0) for j in range(num_groups)]}"
                            f"{'; grad weighted by ssl_num_valid_samples' if weight_by_nv else ''}"
                        )
                    
                    group_size = B // num_groups
                    remainder = B % num_groups
                    group_sizes = [group_size + (1 if j < remainder else 0) for j in range(num_groups)]
                    
                    total_loss = None
                    total_nv_sum = 0
                    valid_groups = 0
                    accumulated_log_vars = {}
                    
                    start_idx = 0
                    for group_idx in range(num_groups):
                        end_idx = start_idx + group_sizes[group_idx]
                        
                        batch_x_sub = batch_x[start_idx:end_idx]
                        batch_x_mark_sub = batch_x_mark[start_idx:end_idx] if batch_x_mark is not None else None
                        batch_lon_lat_sub = batch_lon_lat[start_idx:end_idx] if batch_lon_lat is not None else None
                        
                        with amp_autocast_ctx(self.args):
                            if self.args.model == 'Patch_NTP_TED' and next_batch_x is None:
                                start_idx = end_idx
                                continue
                            outputs_sub = self.model(
                                batch_x_sub, 
                                time_mark=batch_x_mark_sub, 
                                next_x_enc=next_batch_x[start_idx:end_idx] if next_batch_x is not None else None,
                                mode='train',
                                mask_rate_v1=self.args.mask_rate_v1,
                                mask_rate_v2=self.args.mask_rate_v2,
                                imputator=self.imputator,
                                teacher_temp=current_teacher_temp,
                                iteration=global_iter,
                                total_iterations=total_iterations,
                                current_epoch=epoch,
                                lon_lat=batch_lon_lat_sub,
                            )

                            nv_raw = outputs_sub.get('ssl_num_valid_samples', None)
                            if nv_raw is None:
                                nv = int(batch_x_sub.shape[0])
                            else:
                                nv = int(nv_raw)
                            if nv <= 0:
                                if self.args.local_rank == 0:
                                    print(
                                        f"[Warn] ssl_num_valid_samples=0 at step={global_iter}, "
                                        f"group={group_idx}, skip this group."
                                    )
                                del outputs_sub
                                start_idx = end_idx
                                continue
                            
                            loss_sub, log_vars_sub = self._compute_loss(outputs_sub)
                            
                            if not torch.isfinite(loss_sub):
                                if self.args.local_rank == 0:
                                    print(
                                        f"[Warn] Non-finite loss_sub detected at step={global_iter}, "
                                        f"group={group_idx}, skip this group."
                                    )
                                del outputs_sub
                                start_idx = end_idx
                                continue

                            valid_groups += 1
                            if weight_by_nv:
                                total_nv_sum += nv
                                lt = loss_sub.detach() * nv
                                if total_loss is None:
                                    total_loss = lt
                                else:
                                    total_loss = total_loss + lt
                                for k, v in log_vars_sub.items():
                                    try:
                                        fv = float(v.detach().item()) if isinstance(v, torch.Tensor) else float(v)
                                        accumulated_log_vars[k] = accumulated_log_vars.get(k, 0.0) + fv * nv
                                    except Exception:
                                        accumulated_log_vars[k] = v
                            else:
                                if total_loss is None:
                                    total_loss = loss_sub
                                else:
                                    total_loss = total_loss + loss_sub
                                for k, v in log_vars_sub.items():
                                    if k not in accumulated_log_vars:
                                        accumulated_log_vars[k] = v
                                    else:
                                        accumulated_log_vars[k] = accumulated_log_vars[k] + v
                        
                        if weight_by_nv:
                            scaler.scale(loss_sub * nv).backward()
                        else:
                            scaler.scale(loss_sub).backward()
                        del outputs_sub
                        start_idx = end_idx
                    
                    if valid_groups == 0 or total_loss is None:
                        if self.args.local_rank == 0:
                            print(f"[Warn] All sub-groups invalid at step={global_iter}, skip optimizer step.")
                        model_optim.zero_grad(set_to_none=True)
                        continue

                    if weight_by_nv:
                        mixed_batch_effective_total = float(total_nv_sum)
                        loss = total_loss / total_nv_sum
                        log_vars = {}
                        for k, v in accumulated_log_vars.items():
                            if isinstance(v, float):
                                log_vars[k] = v / total_nv_sum
                            else:
                                log_vars[k] = v
                    else:
                        # 仅按有效子组归一，避免被跳过子组导致梯度被额外缩小
                        gradAccumDivisor = float(valid_groups)
                        loss = total_loss / gradAccumDivisor
                        log_vars = {}
                        for k, v in accumulated_log_vars.items():
                            try:
                                if isinstance(v, torch.Tensor):
                                    log_vars[k] = v / gradAccumDivisor
                                elif isinstance(v, (int, float)):
                                    log_vars[k] = float(v) / gradAccumDivisor
                                else:
                                    log_vars[k] = v
                            except Exception:
                                log_vars[k] = v
                    
                else:
                    # 标准模式：不使用梯度累积
                    with amp_autocast_ctx(self.args):
                        # 1. Forward (只获取 Logits)
                        # 传递动态 teacher_temp、迭代信息以及 patch 掩码率到 model（用于渐进学习 + 掩码控制）
                        if self.args.model == 'Patch_NTP_TED' and next_batch_x is None:
                            # 当前窗口无法取到 delay 后的序列，直接跳过
                            continue
                        outputs = self.model(
                            batch_x, 
                            time_mark=batch_x_mark, 
                            next_x_enc=next_batch_x,
                            mode='train',
                            mask_rate_v1=self.args.mask_rate_v1,
                            mask_rate_v2=self.args.mask_rate_v2,
                            imputator=self.imputator,
                            teacher_temp=current_teacher_temp,
                            iteration=global_iter,
                            total_iterations=total_iterations,
                            current_epoch=epoch,
                            lon_lat=batch_lon_lat,
                        )
                        
                        # 2. Loss Calculation (使用共享的 criterion 或 ssl_loss)
                        loss, log_vars = self._compute_loss(outputs)
                        if not torch.isfinite(loss):
                            if self.args.local_rank == 0:
                                print(f"[Warn] Non-finite loss detected at step={global_iter}, skip this step.")
                            model_optim.zero_grad(set_to_none=True)
                            continue
                        
                        # 反向传播（标准模式下在这里调用）
                        scaler.scale(loss).backward()
                    
                    # 标准模式：尽早释放 outputs 引用，便于 GC 回收大张量（减轻显存爬升）
                    try:
                        del outputs
                    except NameError:
                        pass
                loss_scalar = float(loss.detach().item())
                train_loss_epoch.append(loss_scalar)

                # Step 指标每步全记录到文件；控制台按间隔打印
                last_loss = train_loss_epoch[-1]
                if self.args.local_rank == 0:
                    step_id = global_iter + 1
                    tensor_item_interval = max(1, int(getattr(self.args, "train_step_tensor_item_interval", 1)))
                    collect_detail_this_step = (step_id % tensor_item_interval == 0)
                    scalar_log_vars = {}
                    if collect_detail_this_step:
                        for k, v in log_vars.items():
                            try:
                                if isinstance(v, torch.Tensor):
                                    scalar_log_vars[k] = float(v.detach().item())
                                elif isinstance(v, (int, float)):
                                    scalar_log_vars[k] = float(v)
                                else:
                                    scalar_log_vars[k] = str(v)
                            except Exception:
                                scalar_log_vars[k] = str(v)
                        keep_log_keys = {
                            "cls",
                            "patch",
                            "fft_align",
                            "koleo",
                            "temporal",
                            "cls_cons",
                            "cls_global_ce",
                            "cls_short_cond_ce",
                            "cls_short_crop_cond_ce",
                            "cls_short_random_cond_ce",
                            "cls_short_anchor_cond_ce",
                        }
                        scalar_log_vars = {
                            k: v
                            for k, v in scalar_log_vars.items()
                            if k in keep_log_keys or k.startswith("condition_")
                        }

                    if step_log_queue is not None:
                        step_record = {
                            "step": int(step_id),
                            "epoch": int(epoch + 1),
                            "iter_in_epoch": int(i + 1),
                            "total": float(last_loss),
                        }
                        step_record.update(scalar_log_vars)
                        try:
                            step_log_queue.put_nowait(step_record)
                        except queue.Full:
                            step_log_drop_counter += 1

                    if bool(int(getattr(self.args, "train_step_console_log_enable", 0))):
                        parts = [f"step={step_id}", f"total={last_loss:.6f}"]
                        for k, v in scalar_log_vars.items():
                            if isinstance(v, float):
                                parts.append(f"{k}={v:.6f}")
                            else:
                                parts.append(f"{k}={v}")
                        print("\t" + " | ".join(parts))

                # 进度行低频打印（默认每 200 step），仅用于观察 ETA 与速度
                progress_log_interval = max(1, int(getattr(self.args, 'progress_log_interval', 200)))
                if (i + 1) % progress_log_interval == 0:
                    if self.args.local_rank == 0:
                        progress = global_iter / max(total_iterations, 1) * 100
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print(f"\t[progress {progress:.2f}%] LR: {current_lr:.6f} | T_Temp: {current_teacher_temp:.4f} | speed: {speed:.4f}s/iter | left: {left_time:.0f}s")
                    iter_count = 0
                    time_now = time.time()

                # 注意：backward已经在上面调用过了（标准模式在with autocast内，梯度累积模式在循环内）
                scaler.unscale_(model_optim)
                if mixed_batch_effective_total is not None and mixed_batch_effective_total > 0:
                    inv_eff = 1.0 / float(mixed_batch_effective_total)
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad.mul_(inv_eff)
                elif gradAccumDivisor is not None and gradAccumDivisor > 0:
                    gradScale = 1.0 / gradAccumDivisor
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad.mul_(gradScale)
                
                max_grad_norm = float(getattr(self.args, 'max_grad_norm', 1.0))
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                
                scaler.step(model_optim)
                scaler.update()
                
                # 关键修复：更新Teacher模型（EMA更新）
                # 必须在optimizer.step()之后调用，确保student参数已更新
                # 使用动态 momentum
                if hasattr(self.model, 'module') and hasattr(self.model.module, '_update_teacher'):
                    # DDP包裹的模型
                    self.model.module._update_teacher(m=current_momentum)
                elif hasattr(self.model, '_update_teacher'):
                    # 非DDP模型
                    self.model._update_teacher(m=current_momentum)

                if max_train_steps > 0:
                    smoke_steps_done += 1
                    if smoke_steps_done >= max_train_steps:
                        if self.args.local_rank == 0:
                            print(
                                f"[Smoke] max_train_steps={max_train_steps} reached, stopping.",
                                flush=True,
                            )
                        stop_smoke = True
                        break

                # 周期性释放 CUDA 缓存，缓解动态序列长度导致的显存逐渐增长（PyTorch 会缓存已释放块，不同长度导致碎片化）
                empty_cache_interval = getattr(self.args, 'empty_cache_interval', 0)
                if empty_cache_interval > 0 and (global_iter + 1) % empty_cache_interval == 0:
                    torch.cuda.empty_cache()

            if stop_smoke:
                if self.args.use_multi_gpu:
                    dist.barrier()
                break

            # 中断恢复 train_state（全 rank）；降默认频率减磁盘；首 epoch / 最后一 epoch 必写
            _tsi = max(1, int(getattr(self.args, "train_state_save_interval", 5)))
            _epn = epoch + 1
            if _epn == 1 or (_epn % _tsi == 0) or (_epn >= self.args.train_epochs):
                _save_train_state(nextEpoch=epoch + 1, globalIter=global_iter)
            # 因为 IterableDataset 可能导致不同进程处理不同数量的 batch
            # 如果不同步，会导致 collective 操作数量不一致，造成 NCCL 超时
            if self.args.use_multi_gpu:
                # 同步所有进程，确保都完成了训练循环
                dist.barrier()
                # 可选：打印每个进程处理的 batch 数量（用于调试）
                if epoch == 0:  # 只在第一个 epoch 打印，避免日志过多
                    print(f"[Rank {self.args.local_rank}] Processed {actual_batches} batches in epoch {epoch + 1}")
            
            if self.args.local_rank == 0:
                print("Epoch: {} 耗时: {}".format(epoch + 1, time.time() - epoch_time))
            
            train_loss_avg = np.average(train_loss_epoch)

            # --- 4. Probe 评估（仅 KNN Probe，无验证集 loss）---
            if self.args.use_multi_gpu:
                dist.barrier()
            self.run_probe(imputator=self.imputator, epoch=epoch)

            if self.args.local_rank == 0:
                print("Epoch: {0}, Steps: {1} | Train Loss (avg): {2:.7f}".format(
                    epoch + 1, train_steps, train_loss_avg))

                # 若当前 train loss 优于历史最优，则保存为 checkpoint.pth（供下游 test 使用）
                if train_loss_avg < best_train_loss:
                    best_train_loss = train_loss_avg
                    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                    best_ckpt_path = os.path.join(path, 'checkpoint.pth')
                    torch.save(model_to_save.state_dict(), best_ckpt_path)
                    print(f"[Train] New best train loss {best_train_loss:.7f}, saved to {best_ckpt_path}")

                # --- 5. 周期性检查点保存（按 epoch 编号）；best checkpoint.pth 仍按 train loss ---
                save_model_periodically(
                    self.model,
                    path,
                    epoch + 1,
                    save_interval=max(1, int(getattr(self.args, "checkpoint_epoch_save_interval", 1))),
                    verbose=True,
                )
                current_lr = model_optim.param_groups[0]['lr']
                print('Current learning rate: {:.8f}'.format(current_lr))

            # 早停：仅 Imputator 生效，TED/DINO 跳过（DDP 需所有 rank 参与 broadcast）
            if early_stopping is not None:
                early_stop_signal = torch.tensor(0.0, device=self.device)
                if self.args.local_rank == 0:
                    # 传解包后的模型给 EarlyStopping，避免 state_dict 带 module. 前缀
                    _model_unwrap = self.model.module if hasattr(self.model, 'module') else self.model
                    early_stopping(train_loss_avg, _model_unwrap, path)
                    if early_stopping.early_stop:
                        early_stop_signal += 1.0
                if self.args.use_multi_gpu:
                    dist.broadcast(early_stop_signal, src=0)
                if early_stop_signal.item() > 0.5:
                    if self.args.local_rank == 0:
                        print(f"[EarlyStopping] Triggered at epoch {epoch + 1}")
                    break

        # 早停触发后 / 训练结束后回载最佳 checkpoint（仅 Imputator）
        if early_stopping is not None and self.args.local_rank == 0:
            best_model_path = os.path.join(path, 'checkpoint.pth')
            if os.path.exists(best_model_path):
                state = torch.load(best_model_path, map_location='cpu')
                state, _ = self._normalize_compiled_state_dict_keys(state)
                # 兼容 DDP checkpoint：去掉可能残留的 module. 前缀
                clean_state = {
                    (k.replace('module.', '', 1) if k.startswith('module.') else k): v
                    for k, v in state.items()
                }
                model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
                model_to_load.load_state_dict(clean_state)
                print(f"[EarlyStopping] Loaded best checkpoint from {best_model_path}")
        
        # 退出前最后的同步，防止 Rank 0 加载完了退出了，其他 Rank 还在等
        if self.args.local_rank == 0 and step_log_queue is not None:
            if step_log_drop_counter > 0:
                print(f"[StepLog] dropped {step_log_drop_counter} step records due to full queue.")
            step_log_queue.put(step_log_stop_token)
            if step_log_thread is not None:
                step_log_thread.join(timeout=5)

        if self.args.use_multi_gpu:
            dist.barrier()

        return self.model

    def fine_tuning(self, setting):
        
        pass

    def test(self, setting, test=0):
        """
        测试函数：加载模型并运行KNN Probe评估
        
        Args:
            setting: 实验设置名称（当pretrain_model为None时使用）
            test: 未使用的参数（保持兼容性）
        
        模型加载优先级：
        1. 如果 args.pretrain_model 不为 None，使用 load_pretrained_model() 加载
        2. 否则，从 './checkpoints/{setting}/checkpoint.pth' 加载
        """
        try:
            if self.args.pretrain_model is not None:
                # 使用预训练模型路径
                self.load_pretrained_model()
            else:
                # 从checkpoints目录加载
                checkpoint_path = os.path.join('./checkpoints', setting, 'checkpoint.pth')
                if not os.path.exists(checkpoint_path):
                    raise FileNotFoundError(
                        f"Checkpoint文件不存在: {checkpoint_path}\n"
                        f"请设置 --pretrain_model 参数指定预训练模型路径，或确保checkpoint文件存在"
                    )
                
                if not self.args.use_multi_gpu or self.args.local_rank == 0:
                    print(f'正在从checkpoints目录加载模型: {checkpoint_path}')
                self.load_pretrained_model(checkpoint_path=checkpoint_path)
        except Exception as e:
            if not self.args.use_multi_gpu or self.args.local_rank == 0:
                print(f">>> [Error] 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return

        # === 运行 KNN Probe 评估特征质量（仅 rank 0 执行，避免多卡时结果重复打印 3 次）===
        run_probe = (not self.args.use_multi_gpu) or (self.args.local_rank == 0)
        if run_probe:
            print("\n>>> [Test] 开始运行 KNN Probe 评估...")
            knn_k = int(getattr(self.args, 'probe_knn_k', 1))
            self._run_downstream_knn_tasks(probe=self.probe, n_neighbors=knn_k)
        if self.args.use_multi_gpu and dist.is_initialized():
            dist.barrier()

        return



    def expValiImputator(self, numSamples: int = 100, saveDir: str = './imputator_val_plots', seed: int = 2026):
        """
        使用预训练 imputator 对少量样本进行可视化验证。

        - 随机抽取 numSamples 条序列（原始带缺失，不再额外加 mask）
        - 调用 imputator 进行填补
        - 对每条序列分别绘制：
          - 原始带缺失的曲线（NaN 会在图中断开）
          - 填补后的完整曲线
        - 每条序列保存为一张 PNG 图片，便于人工检查 imputator 质量

        Args:
            numSamples: 需要可视化的序列条数（默认 100）
            saveDir: 保存图片的目录
            seed: 固定随机种子；与内部 num_workers=0 配合，保证多次运行选中同一批序列

        示例::

            exp = Exp_SSL(args)
            exp.expValiImputator(numSamples=100, saveDir='./impu_debug', seed=2026)
        """
        import os
        import random
        import matplotlib.pyplot as plt
        from models.Transformer import load_pretrained_imputator

        # 仅在 rank 0 上做可视化，避免多卡重复绘图
        if self.args.use_multi_gpu and getattr(self.args, 'local_rank', 0) != 0:
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.barrier()
            except Exception:
                pass
            return

        os.makedirs(saveDir, exist_ok=True)

        checkpoint_path = getattr(self.args, 'pretrained_imputator_path', None)
        if checkpoint_path is None or str(checkpoint_path).lower() == 'none':
            raise ValueError("expValiImputator: 未设置有效的 --pretrained_imputator_path，无法加载 imputator。")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # IterableDataset 多 worker 时 batch 顺序不可复现；此处强制单 worker 以便同一 seed 下选中相同样本
        _old_nw = int(getattr(self.args, 'num_workers', 0))
        self.args.num_workers = 0
        collected = 0
        try:
            # 加载预训练 imputator（与训练时逻辑保持一致，但这里不需要 DDP 包裹）
            imputator = load_pretrained_imputator(
                configs=self.args,
                checkpoint_path=checkpoint_path,
                target_device=self.device,
                use_ddp=False,
                local_rank=None,
            )
            imputator.eval()

            # 获取数据（使用 train 集即可，只取原始序列）
            train_data, train_loader = self._get_data(flag='train')

            sample_index = 0

            with torch.no_grad():
                for batch_tuple in train_loader:
                    batch_x, batch_x_mark = batch_tuple[0], batch_tuple[1]
                    batch_lon_lat = batch_tuple[3] if len(batch_tuple) > 3 else None
                    batch_x = batch_x.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    if batch_lon_lat is not None:
                        batch_lon_lat = batch_lon_lat.float().to(self.device)

                    B, T, C = batch_x.shape
                    indices = list(range(B))
                    random.shuffle(indices)

                    need_in_batch = min(numSamples - collected, B)
                    if need_in_batch <= 0:
                        break

                    select_indices = indices[:need_in_batch]
                    x_sel = batch_x[select_indices]          # [B_sel, T, C]
                    tm_sel = batch_x_mark[select_indices]    # [B_sel, T, 2]
                    ll_sel = batch_lon_lat[select_indices] if batch_lon_lat is not None else None

                    with amp_autocast_ctx(self.args):
                        # 与训练中使用 imputator 的接口保持一致（含 lon_lat 时与训练对齐）
                        imputed_out = imputator(
                            x_sel, time_mark=tm_sel, mode='pred', lon_lat=ll_sel
                        )  # [B_sel, T, C]

                    x_sel_np = x_sel.detach().cpu().numpy()
                    imputed_np = imputed_out.detach().cpu().numpy()

                    for i in range(need_in_batch):
                        if collected >= numSamples:
                            break

                        seq_raw = x_sel_np[i]        # [T, C]
                        seq_impu = imputed_np[i]     # [T, C]

                        time_axis = list(range(seq_raw.shape[0]))
                        num_channels = seq_raw.shape[1]

                        fig, axes = plt.subplots(num_channels, 1, figsize=(12, 2 * num_channels), sharex=True)
                        if num_channels == 1:
                            axes = [axes]

                        for ch in range(num_channels):
                            ax = axes[ch]
                            # 只在有效观测点处画 raw 数据（蓝色点），NaN 不画
                            raw_ch = seq_raw[:, ch]
                            valid_mask = ~np.isnan(raw_ch)
                            ax.plot(
                                np.array(time_axis)[valid_mask],
                                raw_ch[valid_mask],
                                'b.',
                                alpha=0.8,
                                label='raw (obs)'
                            )
                            ax.plot(time_axis, seq_impu[:, ch], 'r-', alpha=0.9, label='imputed')
                            ax.set_ylabel(f'band {ch}')
                            ax.grid(True, alpha=0.3)
                            if ch == 0:
                                ax.legend(loc='upper right', fontsize=8)

                        axes[-1].set_xlabel('time step')
                        plt.tight_layout()

                        save_path = os.path.join(saveDir, f'seq_{sample_index:04d}.png')
                        plt.savefig(save_path, dpi=150)
                        plt.close(fig)

                        collected += 1
                        sample_index += 1

                        if collected >= numSamples:
                            break

            print(f"expValiImputator: 可视化完成，共保存 {collected} 条序列到目录: {saveDir}")
        finally:
            self.args.num_workers = _old_nw

    def pred(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        print('loading model')
        if self.args.pretrain_model is None:
            raise ValueError("请设置 --pretrain_model 为包含 checkpoint_epoch_x.pth 的目录")

        # 收集 epoch_1 到 epoch_14 的 checkpoint 路径
        checkpoint_paths = []
        for ep in range(1, 22):
            ckpt = os.path.join(self.args.pretrain_model, f'checkpoint_epoch_{ep}.pth')
            if os.path.exists(ckpt):
                checkpoint_paths.append(ckpt)

        if not checkpoint_paths:
            raise FileNotFoundError("未找到 checkpoint_epoch_1.pth ~ checkpoint_epoch_51.pth")


        # 遍历指定的 checkpoints
        for ckpt_path in checkpoint_paths:
            print(f"加载并推理: {ckpt_path}")
            # 处理并行训练带来的module前缀问题
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            checkpoint, _ = self._normalize_compiled_state_dict_keys(checkpoint)
            new_state_dict = {
                (key.replace('module.', '', 1) if key.startswith('module.') else key): value
                for key, value in checkpoint.items()
            }

            model_dict = self.model.state_dict()
            pretrained_dict = {}
            for k, v in new_state_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    pretrained_dict[k] = v
                else:
                    module_key = 'module.' + k
                    if module_key in model_dict and v.shape == model_dict[module_key].shape:
                        pretrained_dict[module_key] = v
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)

            # 每个 checkpoint 独立收集与保存
            preds = []
            trues = []
            masks = []
            last_tokens = []  # 存储最后一个token
            last_token_cos_list = []  # 存储cos相似度 [B, N]
            all_attns_list = []  # 存储注意力权重（按层保存）
            saved_attn_samples = 0  # 已保存的样本数量（最多100）

            self.model.eval()
            with torch.no_grad():
                for i, batch_tuple in enumerate(test_loader):
                    batch_x, batch_x_mark, next_batch_x = batch_tuple[0], batch_tuple[1], batch_tuple[2]
                    batch_x = batch_x.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    valid_mask = (1 - torch.isnan(batch_x).int()).to(self.device)

                    # Use autocast for mixed precision
                    with amp_autocast_ctx(self.args):
                        outputs, all_attns, last_token, last_token_cos = self.model(batch_x, batch_x_mark, mode=self.args.mode)
                    
                    batch_x = torch.nan_to_num(batch_x, nan=0.0)
                    outputs = outputs.detach().cpu().numpy()
                    pred = outputs
                    true = batch_x.detach().cpu().numpy()
                    valid_mask = valid_mask.detach().cpu().numpy()
                    
                    # 处理最后一个token
                    last_token_np = last_token.detach().cpu().numpy()
                    
                    # 处理所有层的注意力权重（只保留前100个样本）
                    all_attns_np = []
                    if saved_attn_samples < 100:
                        remaining = 100 - saved_attn_samples
                        truncated_attns = []
                        for attn in all_attns:
                            attn_np = attn.detach().cpu().numpy()
                            truncated_attns.append(attn_np[:remaining])
                        all_attns_np = truncated_attns

                    preds.append(pred)
                    trues.append(true)
                    last_tokens.append(last_token_np)
                    last_token_cos_list.append(last_token_cos.detach().cpu().numpy())
                    # 只保存前100个样本的注意力权重，避免文件过大
                    if saved_attn_samples < 100 and all_attns_np:
                        all_attns_list.append(all_attns_np)  # 保存每层的注意力权重
                        saved_attn_samples += all_attns_np[0].shape[0]
                    masks.append(valid_mask)

                # anomaly_mask=1-(valid_mask-anomaly_mask)

                # # 可视化并保存结果
                # visual_results(
                #     true,
                #     pred,
                #     valid_mask,
                #     None,
                #     epoch=1,
                #     batch_idx=i,
                #     plot_anomaly=False,
                #     plot_atten=None,
                #     anomalies_prob=None,
                # )


            preds = np.concatenate(preds, 0)
            last_tokens = np.concatenate(last_tokens, 0)
            last_token_cos_list = np.concatenate(last_token_cos_list, 0)
            trues = np.concatenate(trues, 0)
            masks = np.concatenate(masks, 0)
            
            print('preds.shape', preds.shape)
            print('last_tokens.shape', last_tokens.shape)
            print('last_token_cos.shape', last_token_cos_list.shape)
            print('trues.shape', trues.shape)
            print('masks.shape', masks.shape)
            
            # 处理所有层的注意力权重
            # all_attns_list 是一个列表，每个元素是一个batch的所有层的注意力权重列表
            # 需要重新组织数据结构以便保存
            num_layers = len(all_attns_list[0]) if len(all_attns_list) > 0 else 0
            print(f'注意力层数: {num_layers}')
            if num_layers > 0:
                # 将每层的注意力权重分别保存
                for layer_idx in range(num_layers):
                    layer_attns = [batch_attns[layer_idx] for batch_attns in all_attns_list]
                    layer_attns = np.concatenate(layer_attns, 0)
                    print(f'Layer {layer_idx} attn.shape: {layer_attns.shape}')

            # 创建保存目录（按 checkpoint 子目录保存）
            base_save_dir = getattr(self.args, 'save_dir', './output_images/')
            ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
            save_dir = os.path.join(base_save_dir, ckpt_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # 保存数据
            np.save(os.path.join(save_dir, 'preds.npy'), preds)
            np.save(os.path.join(save_dir, 'last_tokens.npy'), last_tokens)
            np.save(os.path.join(save_dir, 'last_token_cos.npy'), last_token_cos_list)
            np.save(os.path.join(save_dir, 'trues.npy'), trues)
            np.save(os.path.join(save_dir, 'masks.npy'), masks)
            
            # 保存每层的注意力权重
            if num_layers > 0:
                for layer_idx in range(num_layers):
                    layer_attns = [batch_attns[layer_idx] for batch_attns in all_attns_list]
                    layer_attns = np.concatenate(layer_attns, 0)
                    np.save(os.path.join(save_dir, f'all_attns_layer_{layer_idx}.npy'), layer_attns)
            
            print(f'数据已保存到: {save_dir}')


        # # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        # mae, mse, rmse, mape, mspe = metric(preds[masks == 1], trues[masks == 1])
        # print('mse:{}, mae:{}'.format(mse, mae))

        return

    def visualize(self, setting, checkpoint_path=None, num_samples=10):
        """
        可视化函数：加载训练好的模型，获取 attention 权重和 cos 相似度信息，并进行可视化
        
        Args:
            setting: 实验设置名称
            checkpoint_path: checkpoint 路径（可选，如果为 None 则使用 setting 对应的最佳模型）
            num_samples: 要可视化的样本数量（最多处理第一个 batch 的前 50 个样本）
        
        Note:
            为了加快可视化速度，只使用验证集第一个 batch 的前 50 个样本
        """
        import matplotlib.pyplot as plt
        
        # 加载模型
        if checkpoint_path is None:
            checkpoint_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"加载模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 处理并行训练带来的module前缀问题
        checkpoint, _ = self._normalize_compiled_state_dict_keys(checkpoint)
        new_state_dict = {
            (key.replace('module.', '', 1) if key.startswith('module.') else key): value
            for key, value in checkpoint.items()
        }

        model_dict = self.model.state_dict()
        pretrained_dict = {}
        for k, v in new_state_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                pretrained_dict[k] = v
            else:
                module_key = 'module.' + k
                if module_key in model_dict and v.shape == model_dict[module_key].shape:
                    pretrained_dict[module_key] = v
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        
        # 使用训练集数据做可视化（不再使用验证集）
        _, vis_loader = self._get_data(flag='train')

        save_dir = os.path.join('./visualizations/', setting)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        max_samples_per_batch = 50
        actual_num_samples = min(num_samples, max_samples_per_batch)
        print(f"可视化设置：只处理训练集第一个 batch 的前 {actual_num_samples} 个样本")

        self.model.eval()
        all_attentions = []
        all_cls_cos_sim_all = []
        all_cls_cos_sim_patch = []

        sample_count = 0

        with torch.no_grad():
            for i, batch_tuple in enumerate(vis_loader):
                batch_x, batch_x_mark, next_batch_x = batch_tuple[0], batch_tuple[1], batch_tuple[2]
                # 只处理第一个 batch
                if i > 0:
                    break
                    
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                
                # 获取模型的可视化输出
                # 处理 DDP 包裹的模型
                model_to_use = self.model.module if hasattr(self.model, 'module') else self.model
                vis_outputs = model_to_use.visualize(batch_x, timeMark=batch_x_mark, imputator=None)
                
                # 收集数据
                if vis_outputs['all_attns'] is not None and len(vis_outputs['all_attns']) > 0:
                    # all_attns 是一个列表，每个元素是一层的 attention
                    # 对于每个样本，我们需要保存每层的 attention
                    B = batch_x.shape[0]
                    # 限制每个 batch 最多处理 max_samples_per_batch 个样本
                    max_samples_this_batch = min(B, actual_num_samples)
                    
                    for b_idx in range(max_samples_this_batch):
                        if sample_count >= num_samples:
                            break
                        
                        # 收集该样本的 attention（每层）
                        sample_attns = []
                        for layer_idx, layer_attn in enumerate(vis_outputs['all_attns']):
                            # nn.MultiheadAttention 默认返回的 attention 权重形状是 3D: [B, N, N]
                            # 如果 average_attn_weights=False，则返回 4D: [B, num_heads, N, N]
                            if layer_attn is not None:
                                try:
                                    if layer_attn.dim() == 3:
                                        # [B, N, N] - 默认格式，已经对所有 head 平均过的 attention
                                        sample_attn = layer_attn[b_idx].cpu().numpy()  # [N, N]
                                    elif layer_attn.dim() == 4:
                                        # [B, num_heads, N, N] - 当 average_attn_weights=False 时
                                        # 取平均所有 head
                                        sample_attn = layer_attn[b_idx].mean(dim=0).cpu().numpy()  # [N, N]
                                    elif layer_attn.dim() == 2:
                                        # [N, N] - 单个样本的 attention（不应该出现，但处理一下）
                                        sample_attn = layer_attn.cpu().numpy() if b_idx == 0 else None
                                    else:
                                        print(f"Warning: Unexpected attention shape at layer {layer_idx}: {layer_attn.shape}")
                                        continue
                                    
                                    if sample_attn is not None:
                                        sample_attns.append(sample_attn)
                                except Exception as e:
                                    print(f"Error processing attention at layer {layer_idx}: {e}")
                                    continue
                            else:
                                # 如果该层没有返回 attention，添加 None 占位
                                sample_attns.append(None)
                        
                        # 检查是否有有效的 attention（至少有一层不是 None）
                        has_valid_attn = any(attn is not None for attn in sample_attns)
                        if has_valid_attn:
                            all_attentions.append(sample_attns)
                            
                            # 收集 cos 相似度
                            if vis_outputs['cls_cos_sim_all'] is not None:
                                all_cls_cos_sim_all.append(vis_outputs['cls_cos_sim_all'][b_idx].cpu().numpy())
                            if vis_outputs['cls_cos_sim_patch'] is not None:
                                all_cls_cos_sim_patch.append(vis_outputs['cls_cos_sim_patch'][b_idx].cpu().numpy())
                            
                            sample_count += 1
                        elif len(sample_attns) > 0:
                            # 即使没有有效的 attention，也保存（用于调试）
                            all_attentions.append(sample_attns)
                            sample_count += 1
                
                # 只处理第一个 batch，处理完就退出
                break
        
        print(f"收集了 {sample_count} 个样本的可视化数据（来自第一个 batch 的前 {actual_num_samples} 个样本）")
        
        # 可视化 attention 权重（每层）
        if len(all_attentions) > 0:
            num_layers = len(all_attentions[0])
            print(f"模型有 {num_layers} 层")
            
            # 对每个样本，可视化每层的 attention
            for sample_idx in range(min(len(all_attentions), num_samples)):
                sample_attns = all_attentions[sample_idx]
                
                # 创建子图：每层一个
                fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5))
                if num_layers == 1:
                    axes = [axes]
                
                for layer_idx, attn in enumerate(sample_attns):
                    ax = axes[layer_idx]
                    if attn is not None:
                        im = ax.imshow(attn, cmap='viridis', aspect='auto')
                        ax.set_title(f'Layer {layer_idx + 1} Attention')
                        ax.set_xlabel('Key Position')
                        ax.set_ylabel('Query Position')
                        plt.colorbar(im, ax=ax)
                    else:
                        ax.text(0.5, 0.5, 'No Attention', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'Layer {layer_idx + 1} (No Attention)')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'attention_sample_{sample_idx}.png'), dpi=150, bbox_inches='tight')
                plt.close()
            
            # 可视化平均 attention（所有样本的平均）
            if len(all_attentions) > 1:
                # 计算所有样本的平均 attention（每层）
                avg_attns_per_layer = []
                for layer_idx in range(num_layers):
                    # 过滤掉 None 值
                    layer_attns = [sample_attns[layer_idx] for sample_attns in all_attentions if sample_attns[layer_idx] is not None]
                    if len(layer_attns) > 0:
                        avg_attn = np.mean(layer_attns, axis=0)
                        avg_attns_per_layer.append(avg_attn)
                    else:
                        avg_attns_per_layer.append(None)
                
                # 可视化平均 attention
                fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5))
                if num_layers == 1:
                    axes = [axes]
                
                for layer_idx, avg_attn in enumerate(avg_attns_per_layer):
                    ax = axes[layer_idx]
                    if avg_attn is not None:
                        im = ax.imshow(avg_attn, cmap='viridis', aspect='auto')
                        ax.set_title(f'Layer {layer_idx + 1} Avg Attention')
                        ax.set_xlabel('Key Position')
                        ax.set_ylabel('Query Position')
                        plt.colorbar(im, ax=ax)
                    else:
                        ax.text(0.5, 0.5, 'No Attention', ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'Layer {layer_idx + 1} (No Attention)')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'attention_avg_all_samples.png'), dpi=150, bbox_inches='tight')
                plt.close()
        
        # 可视化 CLS token 与所有 token 的 cos 相似度
        if len(all_cls_cos_sim_all) > 0:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # 第一个子图：所有样本的 cos 相似度（热力图）
            cos_sim_matrix = np.array(all_cls_cos_sim_all)  # [num_samples, num_tokens]
            im1 = axes[0].imshow(cos_sim_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            axes[0].set_title('CLS Token vs All Tokens Cosine Similarity (All Samples)')
            axes[0].set_xlabel('Token Position')
            axes[0].set_ylabel('Sample Index')
            plt.colorbar(im1, ax=axes[0])
            
            # 第二个子图：平均 cos 相似度
            avg_cos_sim = np.mean(cos_sim_matrix, axis=0)
            axes[1].plot(avg_cos_sim)
            axes[1].set_title('Average CLS Token vs All Tokens Cosine Similarity')
            axes[1].set_xlabel('Token Position')
            axes[1].set_ylabel('Cosine Similarity')
            axes[1].grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'cls_cos_sim_all_tokens.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # 可视化 CLS token 与 patch tokens 的 cos 相似度
        if len(all_cls_cos_sim_patch) > 0:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # 第一个子图：所有样本的 cos 相似度（热力图）
            cos_sim_patch_matrix = np.array(all_cls_cos_sim_patch)  # [num_samples, num_patches]
            im1 = axes[0].imshow(cos_sim_patch_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            axes[0].set_title('CLS Token vs Patch Tokens Cosine Similarity (All Samples)')
            axes[0].set_xlabel('Patch Position')
            axes[0].set_ylabel('Sample Index')
            plt.colorbar(im1, ax=axes[0])
            
            # 第二个子图：平均 cos 相似度
            avg_cos_sim_patch = np.mean(cos_sim_patch_matrix, axis=0)
            axes[1].plot(avg_cos_sim_patch)
            axes[1].set_title('Average CLS Token vs Patch Tokens Cosine Similarity')
            axes[1].set_xlabel('Patch Position')
            axes[1].set_ylabel('Cosine Similarity')
            axes[1].grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'cls_cos_sim_patch_tokens.png'), dpi=150, bbox_inches='tight')
            plt.close()
        
        # 保存原始数据（numpy 格式）
        if len(all_attentions) > 0:
            np.save(os.path.join(save_dir, 'all_attentions.npy'), all_attentions)
        if len(all_cls_cos_sim_all) > 0:
            np.save(os.path.join(save_dir, 'cls_cos_sim_all.npy'), np.array(all_cls_cos_sim_all))
        if len(all_cls_cos_sim_patch) > 0:
            np.save(os.path.join(save_dir, 'cls_cos_sim_patch.npy'), np.array(all_cls_cos_sim_patch))
        
        print(f"可视化结果已保存到: {save_dir}")
        return