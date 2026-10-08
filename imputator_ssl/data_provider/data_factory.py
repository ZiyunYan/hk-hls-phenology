import os

from data_provider.data_loader import (
    Dataset_HLS,
    Dataset_LCMAP_Classification,
    Dataset_LCMAP_Segmentation,
    Dataset_GlanceTraining_Classification,
    Dataset_GlobalTree_Classification,
    Dataset_GlobalTree_Segmentation,
    Dataset_GlanceTraining_Segmentation,
    Dataset_CDL_Classification,
    Dataset_CropHarvest_Classification,
)
from torch.utils.data import DataLoader
import torch

DOWNSTREAM_SUBPATHS = {
    # New layout under /DownStreamTasks
    'LCMAP_Classification': (
        'downstream_classification_task/hls_composite_nc/',
        'LCMAP&Alphaearth/outdir/',
    ),
    'GlanceTraining_Classification': (
        'downstream_classification_task/hls_composite_nc/',
        'GlanceTraining/outdir/',
    ),
    'GlobalTree_Classification': (
        'downstream_classification_task/hls_composite_nc/',
        'GlobalTree/outdir/',
    ),
    'CDL_Classification': (
        'downstream_classification_task/hls_composite_nc/',
        'CDL/outdir/',
    ),
    'CropHarvest_Classification': (
        'downstream_classification_task/hls_composite_nc/',
    ),
    # Keep legacy segmentation defaults for backward compatibility.
    'LCMAP_Segmentation': ('LCMAP_output_dir/',),
    'GlobalTree_Segmentation': ('GlobalTree/outdir/',),
    'GlanceTraining_Segmentation': ('GlanceTraining_segmentation/outdir/',),
}


def _resolve_downstream_root(args, task_name):
    base = getattr(args, 'downstream_data_root', '/intelnvme01/ziyun/DownStreamTasks')
    rel_candidates = DOWNSTREAM_SUBPATHS[task_name]
    if isinstance(rel_candidates, str):
        rel_candidates = (rel_candidates,)
    for rel in rel_candidates:
        candidate = os.path.join(base, rel)
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(base, rel_candidates[0])

data_dict = {
    'HLS': Dataset_HLS,
    'LCMAP_Classification': Dataset_LCMAP_Classification,
    'LCMAP_Segmentation': Dataset_LCMAP_Segmentation,
    'GlanceTraining_Classification': Dataset_GlanceTraining_Classification,
    'GlanceTraining_Segmentation': Dataset_GlanceTraining_Segmentation,
    'GlobalTree_Classification': Dataset_GlobalTree_Classification,
    'GlobalTree_Segmentation': Dataset_GlobalTree_Segmentation,
    'CDL_Classification': Dataset_CDL_Classification,
    'CropHarvest_Classification': Dataset_CropHarvest_Classification,
}

def _get_common_dataloader(args, data_set, flag):
    """
    统一的 DataLoader 构建函数 (针对 IterableDataset + DDP)
    """
    # 1. 针对 IterableDataset，sampler 必须为 None
    sampler = None 
    
    # 2. IterableDataset 不支持 DataLoader 级别的 shuffle
    # (Shuffle 必须在 Dataset 内部的 buffer 中进行)
    shuffle = False 
    
    # 3. 计算 prefetch_factor（增大以提升预取，减少 GPU 等待）
    prefetch_factor = max(4, min(16, args.num_workers * 2)) if args.num_workers > 0 else None

    # 4. 构建 DataLoader
    data_loader = DataLoader(
        data_set,

        batch_size=None,  
        shuffle=shuffle,
        pin_memory=True,
        sampler=sampler, # 必须为 None
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=prefetch_factor,
        num_workers=args.num_workers,
        drop_last=False
    )
    return data_loader

def data_provider(args, flag, dataset_dir=None):
    Data = data_dict['RS_pred'] if flag == 'pred' else data_dict.get(args.data, data_dict.get(flag, Dataset_HLS))
    timeenc = 1 if args.embed == 'timeF' else 0
    
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
        sampling_stride=args.sampling_stride,
        delay=args.delay,
        seasonal_patterns=args.seasonal_patterns,
        batch_size=args.batch_size,
        train_data_ratio=getattr(args, 'train_data_ratio', 1.0),
    )
    
    # 只在主进程输出（避免DDP重复输出）
    try:
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(flag, len(data_set))
    except:
        try:
            print(flag, len(data_set))
        except:
            pass

    return data_set, _get_common_dataloader(args, data_set, flag)

def data_provider_LCMAP_Classification(args, flag, dataset_dir=None, disable_ddp_split=False):
    Data = data_dict['LCMAP_Classification']  
    timeenc = 0 if args.embed != 'timeF' else 1
    ds_root = dataset_dir or _resolve_downstream_root(args, 'LCMAP_Classification')
    data_set = Data(
        root_path=ds_root,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        data_path=args.data_path,
        scale=True,  
        timeenc=timeenc,
        freq=args.freq,
        sampling_stride=getattr(args, 'sampling_stride', 366),
        batch_size=args.batch_size,
        disable_ddp_split=disable_ddp_split  # 用于KNN Probe等场景
    )
    # 只在主进程输出（避免DDP重复输出）
    try:
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"{flag}: {len(data_set)} batches")
    except:
        try:
            print(f"{flag}: {len(data_set)} batches")
        except:
            pass

    return data_set, _get_common_dataloader(args, data_set, flag)

def data_provider_GlanceTraining_Classification(args, flag, dataset_dir=None, disable_ddp_split=False):
    Data = data_dict['GlanceTraining_Classification']  
    timeenc = 0 if args.embed != 'timeF' else 1
    ds_root = dataset_dir or _resolve_downstream_root(args, 'GlanceTraining_Classification')
    data_set = Data(
        root_path=ds_root,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        data_path=args.data_path,
        scale=True,  
        timeenc=timeenc,
        freq=args.freq,
        sampling_stride=getattr(args, 'sampling_stride', 366),
        batch_size=args.batch_size,
        disable_ddp_split=disable_ddp_split  # 用于KNN Probe等场景
    )
    # 只在主进程输出（避免DDP重复输出）
    try:
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"{flag}: {len(data_set)} batches")
    except:
        try:
            print(f"{flag}: {len(data_set)} batches")
        except:
            pass

    return data_set, _get_common_dataloader(args, data_set, flag)

def data_provider_LCMAP_Segmentation(args, flag, dataset_dir=None, disable_ddp_split=False):
    Data = data_dict['LCMAP_Segmentation']  
    timeenc = 0 if args.embed != 'timeF' else 1
    ds_root = dataset_dir or _resolve_downstream_root(args, 'LCMAP_Segmentation')
    data_set = Data(
        root_path=ds_root,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        data_path=args.data_path,
        scale=True,  
        timeenc=timeenc,
        freq=args.freq,
        sampling_stride=getattr(args, 'sampling_stride', 366),
        batch_size=args.batch_size,
        disable_ddp_split=disable_ddp_split  # 用于KNN Probe等场景
    )
    # 只在主进程输出（避免DDP重复输出）
    try:
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"{flag}: {len(data_set)} batches")
    except:
        try:
            print(f"{flag}: {len(data_set)} batches")
        except:
            pass

    return data_set, _get_common_dataloader(args, data_set, flag)


def data_provider_GlobalTree_Classification(args, flag, dataset_dir=None, disable_ddp_split=False):
    Data = data_dict['GlobalTree_Classification']
    timeenc = 0 if args.embed != 'timeF' else 1
    ds_root = dataset_dir or _resolve_downstream_root(args, 'GlobalTree_Classification')
    data_set = Data(
        root_path=ds_root,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        data_path=args.data_path,
        scale=True,
        timeenc=timeenc,
        freq=args.freq,
        sampling_stride=getattr(args, 'sampling_stride', 244),
        batch_size=args.batch_size,
        disable_ddp_split=disable_ddp_split,
    )
    try:
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"{flag}: {len(data_set)} batches (GlobalTree_Classification)")
    except Exception:
        try:
            print(f"{flag}: {len(data_set)} batches (GlobalTree_Classification)")
        except Exception:
            pass

    return data_set, _get_common_dataloader(args, data_set, flag)


def data_provider_GlobalTree_Segmentation(args, flag, dataset_dir=None, disable_ddp_split=False):
    Data = data_dict['GlobalTree_Segmentation']
    timeenc = 0 if args.embed != 'timeF' else 1
    ds_root = dataset_dir or _resolve_downstream_root(args, 'GlobalTree_Segmentation')
    data_set = Data(
        root_path=ds_root,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        data_path=args.data_path,
        scale=True,
        timeenc=timeenc,
        freq=args.freq,
        sampling_stride=getattr(args, 'sampling_stride', 122),
        batch_size=args.batch_size,
        disable_ddp_split=disable_ddp_split,
    )
    try:
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"{flag}: {len(data_set)} batches (GlobalTree_Segmentation)")
    except Exception:
        try:
            print(f"{flag}: {len(data_set)} batches (GlobalTree_Segmentation)")
        except Exception:
            pass

    return data_set, _get_common_dataloader(args, data_set, flag)


def data_provider_GlanceTraining_Segmentation(args, flag, dataset_dir=None, disable_ddp_split=False):
    Data = data_dict['GlanceTraining_Segmentation']
    timeenc = 0 if args.embed != 'timeF' else 1
    ds_root = dataset_dir or _resolve_downstream_root(args, 'GlanceTraining_Segmentation')
    data_set = Data(
        root_path=ds_root,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        data_path=args.data_path,
        scale=True,
        timeenc=timeenc,
        freq=args.freq,
        sampling_stride=getattr(args, 'sampling_stride', args.seq_len),
        batch_size=args.batch_size,
        disable_ddp_split=disable_ddp_split,
    )
    try:
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"{flag}: {len(data_set)} batches (GlanceTraining_Segmentation)")
    except Exception:
        try:
            print(f"{flag}: {len(data_set)} batches (GlanceTraining_Segmentation)")
        except Exception:
            pass

    return data_set, _get_common_dataloader(args, data_set, flag)

def data_provider_CDL_Classification(args, flag, dataset_dir=None, disable_ddp_split=False):
    Data = data_dict['CDL_Classification']
    timeenc = 0 if args.embed != 'timeF' else 1
    ds_root = dataset_dir or _resolve_downstream_root(args, 'CDL_Classification')
    data_set = Data(
        root_path=ds_root,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        data_path=args.data_path,
        scale=True,
        timeenc=timeenc,
        freq=args.freq,
        sampling_stride=getattr(args, 'sampling_stride', 122),
        batch_size=args.batch_size,
        disable_ddp_split=disable_ddp_split,
    )
    try:
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"{flag}: {len(data_set)} batches (CDL_Classification)")
    except Exception:
        try:
            print(f"{flag}: {len(data_set)} batches (CDL_Classification)")
        except Exception:
            pass

    return data_set, _get_common_dataloader(args, data_set, flag)


def data_provider_CropHarvest_Classification(args, flag, dataset_dir=None, disable_ddp_split=False):
    Data = data_dict['CropHarvest_Classification']
    timeenc = 0 if args.embed != 'timeF' else 1
    ds_root = dataset_dir or _resolve_downstream_root(args, 'CropHarvest_Classification')
    data_set = Data(
        root_path=ds_root,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        data_path=args.data_path,
        scale=True,
        timeenc=timeenc,
        freq=args.freq,
        sampling_stride=getattr(args, 'sampling_stride', 732),
        batch_size=args.batch_size,
        disable_ddp_split=disable_ddp_split,
    )
    try:
        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"{flag}: {len(data_set)} batches (CropHarvest_Classification)")
    except Exception:
        try:
            print(f"{flag}: {len(data_set)} batches (CropHarvest_Classification)")
        except Exception:
            pass

    return data_set, _get_common_dataloader(args, data_set, flag)