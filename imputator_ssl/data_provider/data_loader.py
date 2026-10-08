import os
import numpy as np
from numpy import random
import pandas as pd
import glob
import re
import torch
from sktime.datasets import load_from_tsfile_to_dataframe
from torch.utils.data import Dataset, IterableDataset
from netCDF4 import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import Normalizer, interpolate_missing
from data_provider.HLS import composite_series, process_in_blocks, add_band_diff_ratio, reshape_data, load_file
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed
import warnings
import gc
import h5py
import sys
import torch.distributed as dist
warnings.filterwarnings('ignore')


class Dataset_HLS(IterableDataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='', target='OT', scale=True,
                 timeenc=1, freq='d', sampling_stride=None, delay=None, seasonal_patterns=None, batch_size=1000,
                 train_data_ratio: float = 1.0):
        super().__init__()
        # size [seq_len, label_len, pred_len]
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val', 'pred']
        type_map = {'train': 0, 'val': 1, 'test': 2, 'pred': 3}
        self.set_type = type_map[flag]
        self.flag = flag

        # self.dynamic_sequence = dynamic_sequence
        self.features = features
        self.target = target
        self.scale = scale
        self.pre_scaler = {
            'mean': np.array([4.0530856e+02, 6.7968939e+02, 7.3541718e+02, 2.5394734e+03,
                              2.0182101e+03, 1.2844141e+03, 5.2847379e-01], dtype=np.float32),
            'std': np.array([2.7406531e+02, 3.4935846e+02, 5.2149530e+02, 9.8295978e+02,
                             9.3158044e+02, 8.0511346e+02, 2.8968227e-01], dtype=np.float32),
        }
        self.timeenc = timeenc
        self.freq = freq
        self.stride = sampling_stride if sampling_stride is not None else 1
        self.seasonal_patterns = seasonal_patterns
        # self.root_path = r'/intelnvme01/ziyun/USA_OUTPUT/'  # Adjust as needed
        self.root_path = root_path
        self.data_path = data_path
        self.batch_size = batch_size
        self.delay = delay
        self.train_data_ratio = float(train_data_ratio) if train_data_ratio is not None else 1.0
        self.__read_data__()

    def __read_data__(self):
        # Determine file path and type
        h5_path = os.path.join(self.root_path, f"{self.flag}.h5")
        nc_path = os.path.join(self.root_path, f"{self.flag}.nc")
        nc_opt_path = os.path.join(self.root_path, f"{self.flag}_optimized.nc")
        
        if os.path.exists(h5_path):
            self.file_type = 'h5'
            self.data_x = h5_path
        elif os.path.exists(nc_opt_path):
            self.file_type = 'nc'
            self.data_x = nc_opt_path
        elif os.path.exists(nc_path):
            self.file_type = 'nc'
            self.data_x = nc_path
        else:
            raise FileNotFoundError(f"Neither HDF5 nor netCDF4 file found: {h5_path} or {nc_path}")

        self.has_lonlat = False
        if self.file_type == 'h5':
            with h5py.File(self.data_x, 'r', swmr=True) as f:
                shape = f['metadata/shape'][:]
                self.num_pixels, self.time_steps, self.bands = shape
                try:
                    import torch.distributed as dist
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        print(f"HDF5 file: {self.data_x}, num_pixels: {self.num_pixels}, time_steps: {self.time_steps}, bands: {self.bands}")
                except:
                    print(f"HDF5 file: {self.data_x}, num_pixels: {self.num_pixels}, time_steps: {self.time_steps}, bands: {self.bands}")
                df_stamp = f['time'][:].astype(str)
                if 'lon' in f and 'lat' in f:
                    self.has_lonlat = True
        else:  # netCDF4
            with Dataset(self.data_x, 'r') as f:
                data_var = f.variables['data']
                shape = data_var.shape
                dim_names = data_var.dimensions
                if dim_names[0] == 'pixels' or shape[0] > shape[1] and shape[0] > shape[2]:
                    self.nc_layout = 'optimized'  # (pixels, time, bands)
                    self.num_pixels, self.time_steps, self.bands = shape
                else:
                    self.nc_layout = 'original'  # (time, bands, pixels)
                    self.time_steps, self.bands, self.num_pixels = shape
                try:
                    import torch.distributed as dist
                    if not dist.is_initialized() or dist.get_rank() == 0:
                        print(f"netCDF4 file: {self.data_x}, layout={self.nc_layout}, num_pixels: {self.num_pixels}, time_steps: {self.time_steps}, bands: {self.bands}")
                except:
                    print(f"netCDF4 file: {self.data_x}, layout={self.nc_layout}, num_pixels: {self.num_pixels}, time_steps: {self.time_steps}, bands: {self.bands}")
                df_stamp = f.variables['time'][:].astype(str)
                if 'lon' in f.variables and 'lat' in f.variables:
                    self.has_lonlat = True
                if 'band_mean' in f.variables and 'band_std' in f.variables:
                    self.pre_scaler = {
                        'mean': np.array(f.variables['band_mean'][:], dtype=np.float32).reshape(-1),
                        'std': np.array(f.variables['band_std'][:], dtype=np.float32).reshape(-1),
                    }
        if self.has_lonlat:
            try:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"  lon/lat 可用，将加入 batch 输出")
            except:
                print(f"  lon/lat 可用，将加入 batch 输出")

        if self.timeenc == 1:
            df_stamp = pd.to_datetime(df_stamp, format='%Y%j')
            data_stamp = time_features(df_stamp, freq=self.freq).transpose(1, 0)
        else:
            data_stamp = np.zeros((self.time_steps, 4))  # Adjust for time encoding
        self.data_stamp = data_stamp
        self.data_sets = {self.flag: self.data_x}
        self.windows_per_sample = (self.time_steps - self.label_len) // self.stride + 1
        self.window_indices = list(range(self.windows_per_sample))
        self.num_batches = (self.num_pixels + self.batch_size - 1) // self.batch_size
        self.batch_indices = list(range(self.num_batches))

        # Optional: subsample training data by taking the first N batch indices.
        # This is applied BEFORE DDP split so that "ratio" refers to the global training set.
        # Note: we deliberately keep the order deterministic (sequential) to make quick testing easier.
        if self.flag == 'train' and self.train_data_ratio < 1.0:
            if not (0.0 < self.train_data_ratio <= 1.0):
                raise ValueError(f"train_data_ratio must be in (0, 1], got {self.train_data_ratio}")
            target_batches = int(np.floor(self.num_batches * self.train_data_ratio))
            target_batches = max(1, min(target_batches, self.num_batches))
            self.batch_indices = self.batch_indices[:target_batches]
            self.num_batches = len(self.batch_indices)
       
    def __iter__(self):
        total_length = self.time_steps
        seq_len = self.seq_len
        pred_len = self.pred_len

        # 1. 获取所有索引
        indices = self.batch_indices[:]

        # 2. DDP 进程级切分 (Process Split)
        # 先把数据分给不同的显卡
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            
            # 关键修复：确保每个进程处理相同数量的 batch
            # 如果总数不能被 world_size 整除，截断到相同长度
            batches_per_rank = len(indices) // world_size
            if batches_per_rank > 0:
                # 截断到能被 world_size 整除的长度
                indices = indices[:batches_per_rank * world_size]
                # 然后按 rank 切分：显卡0拿 [0, 3, 6], 显卡1拿 [1, 4, 7]...
                indices = indices[rank::world_size]
            else:
                # 如果 batch 数量太少，只让 rank 0 处理
                indices = indices if rank == 0 else []
        
        # 3. Worker 线程级切分 (Worker Split)
        # 再把当前显卡分到的数据，分给不同的 DataLoader worker
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            # Worker0 拿 [0, 6], Worker1 拿 [3, 9]...
            indices = indices[worker_id::num_workers]

        # 4. Shuffle (只打乱分到自己手里的这部分)
        window_indices = self.window_indices[:]
        if self.flag == 'train':
            random.shuffle(window_indices) 
            # When using a subset (train_data_ratio < 1), keep indices sequential for easier debugging/testing.
            if not (self.train_data_ratio < 1.0):
                random.shuffle(indices)

        # 5. 打开文件
        if self.file_type == 'h5':
            f = h5py.File(self.data_x, 'r', swmr=True)
        else:
            f = Dataset(self.data_x, 'r')

        try:
            # 6. 迭代最终的 indices
            for batch_idx in indices:
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, self.num_pixels)
                batch_len = end - start

                # 用 slice 替代 list 索引，连续区间读取更快（尤其对优化版 NC）
                if self.file_type == 'h5':
                    seq_x = f['data'][start:end, :self.time_steps, :]
                elif getattr(self, 'nc_layout', 'original') == 'optimized':
                    seq_x = f.variables['data'][start:end, :, :]
                else:
                    seq_x = f.variables['data'][:, :, start:end]
                    seq_x = np.ascontiguousarray(np.transpose(seq_x, (2, 0, 1)))

                batch_lonlat = None
                if self.has_lonlat:
                    if self.file_type == 'h5':
                        b_lon = f['lon'][start:end]
                        b_lat = f['lat'][start:end]
                    else:
                        b_lon = f.variables['lon'][start:end]
                        b_lat = f.variables['lat'][start:end]
                    batch_lonlat = np.stack([b_lon, b_lat], axis=-1).astype(np.float32)  # [batch_len, 2]

                if self.scale:
                    seq_x = (seq_x - self.pre_scaler['mean']) / self.pre_scaler['std']

                for w_idx in window_indices:
                    s_begin = w_idx * self.stride
                    s_end = s_begin + seq_len
                    batch_seq_x = seq_x[:, s_begin:s_end, :]
                    seq_x_mark = self.data_stamp[s_begin:s_end, :]

                    if self.seq_len == seq_len:
                        padded_x = batch_seq_x
                        padded_stamp = np.tile(seq_x_mark[None, :, :], (batch_len, 1, 1))
                    else:
                        padded_x = np.full((batch_len, self.seq_len, batch_seq_x.shape[2]), 0)
                        padded_stamp = np.zeros((batch_len, self.seq_len, seq_x_mark.shape[1]))
                        pad_start = (self.seq_len - seq_len) // 2
                        pad_end = pad_start + seq_len
                        padded_x[:, pad_start:pad_end, :] = batch_seq_x
                        padded_stamp[:, pad_start:pad_end, :] = np.tile(seq_x_mark[None, :, :], (batch_len, 1, 1))

                    next_s_begin = s_begin + self.delay
                    next_s_end = next_s_begin + pred_len

                    if next_s_end <= self.time_steps:
                        target_seq = seq_x[:, next_s_begin:next_s_end, :]
                        B, actual_len, C = target_seq.shape
                        next_padded_x = torch.full((B, self.pred_len, C), fill_value=torch.nan, dtype=torch.float32)
                        next_padded_x[:, :actual_len, :] = torch.from_numpy(target_seq).float()
                    else:
                        next_padded_x = None

                    if batch_lonlat is not None:
                        ll_expanded = np.tile(batch_lonlat[:, None, :], (1, padded_x.shape[1], 1))  # [B, T, 2]
                        batch_ll_tensor = torch.from_numpy(np.ascontiguousarray(ll_expanded)).float()
                    else:
                        batch_ll_tensor = None

                    yield (
                        torch.from_numpy(np.ascontiguousarray(padded_x)).float(),
                        torch.from_numpy(np.ascontiguousarray(padded_stamp)).float(),
                        next_padded_x,
                        batch_ll_tensor,
                    )
        finally:
            f.close()

    def __len__(self):
        import torch.distributed as dist
        total_len = self.num_batches * self.windows_per_sample
        
        # 如果是 DDP，每个进程看到的长度应该是 总长度 / 卡数
        if dist.is_initialized():
            return total_len // dist.get_world_size()
        return total_len

    def inverse_transform(self, data):
        if self.pre_scaler is not None:
            return data * self.pre_scaler['std'] + self.pre_scaler['mean']
        else:
            return data


class Dataset_LCMAP_Classification(IterableDataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='', scale=True,
                 timeenc=1, freq='d', sampling_stride=None, batch_size=1000, disable_ddp_split=False):
        super().__init__()
        """
        整体逻辑（KNN / 下游分类用）：
        1）长度设定
           - 外部通过 size[0] 传入 seq_len（通常等于训练时的 args.seq_len，例如 366 / 732）；
           - 如果 size 为空，则默认 seq_len = 366。
        2）时间轴与窗口
           - 从 HDF5 / NetCDF 读取整段时间序列 [B, time_steps, C]；
           - KNN 探针场景：不做滑动窗口，只取一个窗口：
               * windows_per_sample = 1
               * window_indices = [0]
               * 窗口范围固定为 [0 : seq_len]。
        3）长度对齐
           - 如果 seq_len ≤ time_steps：直接切片 [0 : seq_len]，不做 padding；
           - 如果 seq_len > time_steps：先取 [0 : time_steps]，再用最后一个时间步复制填充到长度 seq_len。
        4）时间编码
           - 使用 self.data_stamp 上同样的 [0 : seq_len] 时间步，必要时也按最后一步复制做 padding；
           - 将 [seq_len, time_feat] broadcast 成 [B, seq_len, time_feat] 与 batch 对齐。
        5）返回给模型
           - 返回 (batch_x, batch_x_mark, labels)，其中：
               * batch_x: [B, seq_len, C]（与当前实验的 args.seq_len 一致）；
               * batch_x_mark: [B, seq_len, time_feat]；
               * labels: [B]（像素级静态标签）。
        6）KNN 调用方式
           - ExpProbe.knn_probe_LCMAP_Classification 中，直接将 batch_x / batch_x_mark 送入 model.backbone.encode(...)，
             然后从 outputs['cls_token'] 抽取特征做 KNN。
        """
        # size [seq_len, label_len, pred_len] -> 分类任务只关心 seq_len (窗口大小)
        # 使用size参数，如果没有提供则使用默认值366
        if size is not None and len(size) > 0:
            self.seq_len = size[0]
        else:
            self.seq_len = 366
        
        # 初始化设置
        self.flag = flag
        self.features = features
        self.scale = scale
        self.disable_ddp_split = disable_ddp_split  # 用于KNN Probe等场景，禁用DDP切分
        # 保持原有的 hard-coded scaler
        self.pre_scaler = {
            'mean': np.array([4.0530856e+02, 6.7968939e+02, 7.3541718e+02, 2.5394734e+03,
                              2.0182101e+03, 1.2844141e+03, 5.2847379e-01], dtype=np.float32),
            'std': np.array([2.7406531e+02, 3.4935846e+02, 5.2149530e+02, 9.8295978e+02,
                             9.3158044e+02, 8.0511346e+02, 2.8968227e-01], dtype=np.float32)
        }
        self.timeenc = timeenc
        self.freq = freq
        # 使用sampling_stride参数，如果没有提供则使用seq_len（即不重叠）
        self.stride = sampling_stride if sampling_stride is not None else self.seq_len
        print(f">>> [Dataset_LCMAP_Classification] seq_len: {self.seq_len}, stride: {self.stride}, sampling_stride param: {sampling_stride}")
        self.root_path = root_path
        # self.data_path = data_path
        self.batch_size = batch_size
        
        self.__read_data__()

    def __read_data__(self):
        # 1. 确定文件路径和类型（兼容新旧命名）
        h5_candidates = [
            os.path.join(self.root_path, "lcmap_classification_dataset.h5"),
        ]
        nc_candidates = [
            os.path.join(self.root_path, "lcmap_hls_classification_processed.nc"),
            os.path.join(self.root_path, "lcmap_classification_dataset.nc"),
        ]

        self.file_type = None
        self.data_x = None
        for p in h5_candidates:
            if os.path.exists(p):
                self.file_type = 'h5'
                self.data_x = p
                break
        if self.data_x is None:
            for p in nc_candidates:
                if os.path.exists(p):
                    self.file_type = 'nc'
                    self.data_x = p
                    break
        if self.data_x is None:
            raise FileNotFoundError(f"File not found in {self.root_path}")

        # 2. 读取元数据和全部 Label (因为 Label 比较小，可以一次读入内存)
        if self.file_type == 'h5':
            with h5py.File(self.data_x, 'r', swmr=True) as f:
                shape = f['metadata/shape'][:]
                self.num_pixels, self.time_steps, self.bands = shape
                df_stamp = f['time'][:].astype(str)
                self.labels = f['labels'][:] # 加载 Label [num_pixels]
        else: # netCDF4
            with Dataset(self.data_x, 'r') as f:
                # 注意：NetCDF 生成脚本中 data 维度是 (time, bands, samples)，这里需要确认 dataset 实现是否转置
                # 根据之前的 Dataset_HLS 逻辑，这里假设 nc 文件读取时需要转置或者通过维度名获取
                self.time_steps = f.dimensions['time'].size
                self.bands = f.dimensions['bands'].size
                self.num_pixels = f.dimensions['samples'].size
                
                df_stamp = f.variables['time'][:].astype(str)
                self.labels = f.variables['labels'][:] # 加载 Label [num_pixels]

        print(f"Dataset ({self.flag}): {self.num_pixels} pixels, {self.time_steps} steps, Labels loaded.")

        # 3. 时间编码处理
        if self.timeenc == 1:
            try:
                df_stamp = pd.to_datetime(df_stamp, format='%Y%j')
            except:
                df_stamp = pd.to_datetime(df_stamp) # 尝试自动推断
            data_stamp = time_features(df_stamp, freq=self.freq).transpose(1, 0)
        else:
            data_stamp = np.zeros((self.time_steps, 4))
        
        self.data_stamp = data_stamp
        
        # 4. 计算迭代参数
        # 修改：探针任务只需要读取固定长度，不使用滑动窗口
        # 每个样本只返回一个窗口（从0开始，长度为seq_len）
        self.windows_per_sample = 1  # 固定为1，不使用滑动窗口
        self.window_indices = [0]  # 只使用第一个窗口（从0开始）
        self.num_batches = (self.num_pixels + self.batch_size - 1) // self.batch_size
        self.batch_indices = list(range(self.num_batches))
        
        print(f">>> [Dataset_LCMAP_Classification] 固定长度读取模式:")
        print(f"    time_steps={self.time_steps}, seq_len={self.seq_len}")
        print(f"    windows_per_sample=1 (不使用滑动窗口)")
        print(f"    num_pixels={self.num_pixels}, batch_size={self.batch_size}")
        print(f"    num_batches={self.num_batches}")
        print(f"    Total samples = {self.num_batches} batches × 1 window = {self.num_batches}")

    def __iter__(self):

        # 1. 获取所有 batch 索引
        indices = self.batch_indices[:]

        # -----------------------------------------------------------
        # 【关键修改 1】DDP 进程级切分 (Process Split)
        # 注意：如果 disable_ddp_split=True，则跳过DDP切分（用于KNN Probe等场景）
        # -----------------------------------------------------------
        if dist.is_initialized() and not self.disable_ddp_split:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            
            # 关键修复：确保每个进程处理相同数量的 batch
            # 如果总数不能被 world_size 整除，截断到相同长度
            batches_per_rank = len(indices) // world_size
            if batches_per_rank > 0:
                # 截断到能被 world_size 整除的长度
                indices = indices[:batches_per_rank * world_size]
                # 步长切分: 确保不同卡拿到的数据互斥
                # 例如: 卡0处理 [0, 3, 6...], 卡1处理 [1, 4, 7...]
                indices = indices[rank::world_size]
            else:
                # 如果 batch 数量太少，只让 rank 0 处理
                indices = indices if rank == 0 else []
        # -----------------------------------------------------------

        # -----------------------------------------------------------
        # 【关键修改 2】Worker 线程级切分 (Worker Split)
        # 注意：这里必须基于上面已经切分过的 indices 继续切分
        # -----------------------------------------------------------
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            # 在当前进程分到的数据中，再分给不同线程
            indices = indices[worker_id::num_workers]

        # -----------------------------------------------------------
        # 【可选】Shuffle 逻辑
        # 原代码注释写着"严格顺序读取"，通常用于测试/推理。
        # 如果是训练(train)，建议打乱；如果是测试(test/val)，保持顺序。
        # -----------------------------------------------------------
        # 注意：不再需要window_indices，因为不使用滑动窗口
        if self.flag == 'train':
            random.shuffle(indices)

        # 打开文件句柄
        if self.file_type == 'h5':
            # SWMR 模式支持多进程读取
            f = h5py.File(self.data_x, 'r', swmr=True)
        else:
            f = Dataset(self.data_x, 'r')

        try:
            # 迭代最终切分好的 indices
            for batch_idx in indices:
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, self.num_pixels)
                batch_pixels = list(range(start, end))
                
                # 获取当前 Batch 对应的 Labels
                # 注意：self.labels 应该是在 __init__ 里加载到内存的 numpy 数组
                batch_labels_raw = self.labels[start:end] # [Batch_Size]

                # 读取数据 (IO操作)
                if self.file_type == 'h5':
                    seq_x = f['data'][batch_pixels, :self.time_steps, :] # [B, T, C]
                else:
                    # NetCDF: (T, C, B) -> 转置为 (B, T, C)
                    seq_x = f.variables['data'][:, :, batch_pixels]
                    seq_x = np.transpose(seq_x, (2, 0, 1))

                # 归一化
                if self.scale:
                    seq_x = (seq_x - self.pre_scaler['mean']) / self.pre_scaler['std']

                # 固定长度读取：只读取前seq_len个时间步，不使用滑动窗口
                s_begin = 0
                s_end = self.seq_len
                
                # 边界检查：确保不会超出数据范围
                if s_end > self.time_steps:
                    # 如果seq_len超过可用数据长度，只读取可用部分并填充
                    actual_len = self.time_steps
                    batch_seq_x = seq_x[:, s_begin:actual_len, :]  # [B, actual_len, C]
                    seq_x_mark = self.data_stamp[s_begin:actual_len, :]  # [actual_len, time_feat]
                    
                    # 如果实际长度小于seq_len，进行填充
                    if actual_len < self.seq_len:
                        pad_len = self.seq_len - actual_len
                        # 使用最后一个时间步的值进行填充
                        last_timestep = batch_seq_x[:, -1:, :]  # [B, 1, C]
                        last_stamp = seq_x_mark[-1:, :]  # [1, time_feat]
                        batch_seq_x = np.concatenate([
                            batch_seq_x,
                            np.tile(last_timestep, (1, pad_len, 1))
                        ], axis=1)  # [B, seq_len, C]
                        seq_x_mark = np.concatenate([
                            seq_x_mark,
                            np.tile(last_stamp, (pad_len, 1))
                        ], axis=0)  # [seq_len, time_feat]
                else:
                    batch_seq_x = seq_x[:, s_begin:s_end, :]  # [B, seq_len, C]
                    seq_x_mark = self.data_stamp[s_begin:s_end, :]  # [seq_len, time_feat]
                
                # 扩展时间戳以匹配 Batch 维度
                padded_stamp = np.tile(seq_x_mark[None, :, :], (len(batch_pixels), 1, 1))

                # 返回 (Input, Time, Label)
                yield (
                    torch.tensor(batch_seq_x, dtype=torch.float32),
                    torch.tensor(padded_stamp, dtype=torch.float32),
                    torch.tensor(batch_labels_raw, dtype=torch.long)
                )

        finally:
            f.close()

    def __len__(self):
        # 修正长度计算：每个样本只有一个窗口（不使用滑动窗口）
        import torch.distributed as dist
        total_len = self.num_batches * self.windows_per_sample  # windows_per_sample = 1
        
        if dist.is_initialized():
            return total_len // dist.get_world_size()
        return total_len


class Dataset_LCMAP_Segmentation(IterableDataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='', scale=True,
                 timeenc=1, freq='d', sampling_stride=None, batch_size=1000, disable_ddp_split=False):
        super().__init__()
        # size [seq_len, label_len, pred_len] -> 分割任务只关心 seq_len (窗口大小)
        self.seq_len = size[0]
        
        # 初始化设置
        self.flag = flag
        self.features = features
        self.scale = scale
        self.disable_ddp_split = disable_ddp_split  # 用于KNN Probe等场景，禁用DDP切分
        # 保持原有的 hard-coded scaler
        self.pre_scaler = {
            'mean': np.array([4.0530856e+02, 6.7968939e+02, 7.3541718e+02, 2.5394734e+03,
                              2.0182101e+03, 1.2844141e+03, 5.2847379e-01], dtype=np.float32),
            'std': np.array([2.7406531e+02, 3.4935846e+02, 5.2149530e+02, 9.8295978e+02,
                             9.3158044e+02, 8.0511346e+02, 2.8968227e-01], dtype=np.float32)
        }
        self.timeenc = timeenc
        self.freq = freq
        self.stride = sampling_stride if sampling_stride is not None else 1
        
        self.root_path = root_path
        # self.data_path = data_path
        self.batch_size = batch_size
        
        self.__read_data__()

    def __read_data__(self):
        # 1. 确定文件路径和类型
        h5_path = os.path.join(self.root_path, f"segmentation_dataset_processed.h5")
        nc_path = os.path.join(self.root_path, f"segmentation_dataset_processed.nc") # 假设分类用的是处理后的nc文件名
        
        if os.path.exists(h5_path):
            self.file_type = 'h5'
            self.data_x = h5_path
        elif os.path.exists(nc_path):
            self.file_type = 'nc'
            self.data_x = nc_path
        else:
            raise FileNotFoundError(f"File not found in {self.root_path}")

        # 2. 读取元数据和全部 Label (因为 Label 比较小，可以一次读入内存)
        if self.file_type == 'h5':
            with h5py.File(self.data_x, 'r', swmr=True) as f:
                shape = f['metadata/shape'][:]
                self.num_pixels, self.time_steps, self.bands = shape
                df_stamp = f['time'][:].astype(str)
                self.labels = f['yearly_status'][:] # 加载 Label [num_pixels,years]
        else: # netCDF4
            with Dataset(self.data_x, 'r') as f:
                # 注意：NetCDF 生成脚本中 data 维度是 (time, bands, samples)，这里需要确认 dataset 实现是否转置
                # 根据之前的 Dataset_HLS 逻辑，这里假设 nc 文件读取时需要转置或者通过维度名获取
                self.time_steps = f.dimensions['time'].size
                self.bands = f.dimensions['bands'].size
                self.num_pixels = f.dimensions['samples'].size
                
                df_stamp = f.variables['time'][:].astype(str)
                self.labels = f.variables['yearly_status'][:] # 加载 Label [num_pixels,years]

        # 获取年份数量
        self.num_years = self.labels.shape[1]  # labels shape: [num_pixels, num_years]
        # 计算每年平均时间步数（用于等比例缩放）
        self.steps_per_year = self.time_steps / self.num_years
        print(f"Dataset ({self.flag}): {self.num_pixels} pixels, {self.time_steps} steps, {self.num_years} years, {self.steps_per_year:.1f} steps/year, Labels loaded.")

        # 3. 时间编码处理
        if self.timeenc == 1:
            try:
                df_stamp = pd.to_datetime(df_stamp, format='%Y%j')
            except:
                df_stamp = pd.to_datetime(df_stamp) # 尝试自动推断
            data_stamp = time_features(df_stamp, freq=self.freq).transpose(1, 0)
        else:
            data_stamp = np.zeros((self.time_steps, 4))
        
        self.data_stamp = data_stamp
        
        # 4. 计算迭代参数
        self.windows_per_sample = (self.time_steps - self.seq_len) // self.stride + 1
        self.window_indices = list(range(self.windows_per_sample))
        self.num_batches = (self.num_pixels + self.batch_size - 1) // self.batch_size
        self.batch_indices = list(range(self.num_batches))

    def __iter__(self):
        # 不进行 Shuffle，严格顺序读取
        window_indices = self.window_indices[:]
        worker_info = torch.utils.data.get_worker_info()
        
        all_batch_indices = self.batch_indices[:]
        
        # DDP 进程级切分（如果未禁用）
        if dist.is_initialized() and not self.disable_ddp_split:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            
            # 关键修复：确保每个进程处理相同数量的 batch
            # 如果总数不能被 world_size 整除，截断到相同长度
            batches_per_rank = len(all_batch_indices) // world_size
            if batches_per_rank > 0:
                # 截断到能被 world_size 整除的长度
                all_batch_indices = all_batch_indices[:batches_per_rank * world_size]
                # 然后按 rank 切分
                all_batch_indices = all_batch_indices[rank::world_size]
            else:
                # 如果 batch 数量太少，只让 rank 0 处理
                all_batch_indices = all_batch_indices if rank == 0 else []
        
        if worker_info is None:  
            batch_indices = all_batch_indices
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            batch_indices = all_batch_indices[worker_id::num_workers]
        # ----------------------------------
        # 打开文件句柄
        if self.file_type == 'h5':
            f = h5py.File(self.data_x, 'r', swmr=True)
        else:
            f = Dataset(self.data_x, 'r')

        try:
            for batch_idx in batch_indices:
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, self.num_pixels)
                batch_pixels = list(range(start, end))

                # 读取数据 (IO操作)
                if self.file_type == 'h5':
                    seq_x = f['data'][batch_pixels, :self.time_steps, :] # [B, T, C]
                else:
                    # NetCDF: (T, C, B) -> 转置为 (B, T, C)
                    seq_x = f.variables['data'][:, :, batch_pixels]
                    seq_x = np.transpose(seq_x, (2, 0, 1))

                # 归一化
                if self.scale:
                    seq_x = (seq_x - self.pre_scaler['mean']) / self.pre_scaler['std']

                # 滑动窗口切片 (内存操作)
                for w_idx in window_indices:
                    s_begin = w_idx * self.stride
                    s_end = s_begin + self.seq_len
                    
                    batch_seq_x = seq_x[:, s_begin:s_end, :] # [B, seq_len, C]
                    seq_x_mark = self.data_stamp[s_begin:s_end, :] # [seq_len, time_feat]
                    
                    # 扩展时间戳以匹配 Batch 维度
                    padded_stamp = np.tile(seq_x_mark[None, :, :], (len(batch_pixels), 1, 1))

                    # 根据窗口在总序列中的位置，等比例计算覆盖的年份范围
                    # 计算窗口开始和结束位置对应的年份索引
                    year_start_idx = int(s_begin / self.steps_per_year)
                    year_end_idx = int((s_end - 1) / self.steps_per_year)
                    # 确保年份索引在有效范围内
                    year_start_idx = max(0, min(year_start_idx, self.num_years - 1))
                    year_end_idx = max(0, min(year_end_idx, self.num_years - 1))
                    # 确保 year_end_idx >= year_start_idx
                    if year_end_idx < year_start_idx:
                        year_end_idx = year_start_idx
                    # 提取窗口覆盖的所有年份的标签 [batch_size, num_years_in_window]
                    year_indices = list(range(year_start_idx, year_end_idx + 1))
                    batch_labels = self.labels[start:end, year_indices]  # [batch_size, num_years_in_window]

                    # 返回 (Input, Time, Label)
                    # Label 是一个序列，包含窗口覆盖的所有年份的标签
                    yield (
                        torch.tensor(batch_seq_x, dtype=torch.float32),
                        torch.tensor(padded_stamp, dtype=torch.float32),
                        torch.tensor(batch_labels, dtype=torch.long) # [batch_size, num_years_in_window]
                    )

        finally:
            f.close()

    def __len__(self):
        return self.num_batches * self.windows_per_sample


class Dataset_GlobalTree_Classification(IterableDataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='', scale=True,
                 timeenc=1, freq='d', sampling_stride=None, batch_size=1000, disable_ddp_split=False):
        super().__init__()
        """
        GlobalTree 分类数据集（与 LCMAP_Classification 结构对齐）：
        - 原始 netCDF: global_tree_classification_dataset.nc
          形状类似 [time, bands, samples]，这里统一转换为 [samples, time, bands]
        - 只做单窗口读取（不滑窗），窗口长度由 seq_len 控制，默认 244（两年，日尺度）。
        """
        if size is not None and len(size) > 0:
            self.seq_len = size[0]
        else:
            # 默认使用两年长度（244）
            self.seq_len = 244

        self.flag = flag
        self.features = features
        self.scale = scale
        self.disable_ddp_split = disable_ddp_split
        self.pre_scaler = {
            'mean': np.array([4.0530856e+02, 6.7968939e+02, 7.3541718e+02, 2.5394734e+03,
                              2.0182101e+03, 1.2844141e+03, 5.2847379e-01], dtype=np.float32),
            'std': np.array([2.7406531e+02, 3.4935846e+02, 5.2149530e+02, 9.8295978e+02,
                             9.3158044e+02, 8.0511346e+02, 2.8968227e-01], dtype=np.float32)
        }
        self.timeenc = timeenc
        self.freq = freq
        self.stride = sampling_stride if sampling_stride is not None else self.seq_len
        print(f">>> [Dataset_GlobalTree_Classification] seq_len: {self.seq_len}, stride: {self.stride}, sampling_stride param: {sampling_stride}")

        # 覆盖 root_path，使用固定下游数据目录
        self.root_path = root_path
        self.batch_size = batch_size

        self.__read_data__()

    def __read_data__(self):
        h5_candidates = [
            os.path.join(self.root_path, "global_tree_classification_dataset.h5"),
        ]
        nc_candidates = [
            os.path.join(self.root_path, "globaltree_hls_classification_processed.nc"),
            os.path.join(self.root_path, "global_tree_classification_dataset.nc"),
        ]

        self.file_type = None
        self.data_x = None
        for p in h5_candidates:
            if os.path.exists(p):
                self.file_type = 'h5'
                self.data_x = p
                break
        if self.data_x is None:
            for p in nc_candidates:
                if os.path.exists(p):
                    self.file_type = 'nc'
                    self.data_x = p
                    break
        if self.data_x is None:
            raise FileNotFoundError(f"GlobalTree classification file not found in {self.root_path}")

        if self.file_type == 'h5':
            with h5py.File(self.data_x, 'r', swmr=True) as f:
                shape = f['metadata/shape'][:]
                self.num_pixels, self.time_steps, self.bands = shape
                df_stamp = f['time'][:].astype(str)

                # 尝试多种标签字段名，提高兼容性
                if 'labels' in f:
                    self.labels = f['labels'][:]
                elif 'class_ids' in f:
                    self.labels = f['class_ids'][:]
                else:
                    raise KeyError("No labels or class_ids found in GlobalTree classification HDF5 file.")
        else:
            with Dataset(self.data_x, 'r') as f:
                data_var = f.variables['data'] if 'data' in f.variables else f.variables['images']
                shape = data_var.shape
                # 统一视为 (time, bands, samples)
                self.time_steps, self.bands, self.num_pixels = shape
                df_stamp = f.variables['time'][:].astype(str)

                if 'labels' in f.variables:
                    self.labels = f.variables['labels'][:]
                elif 'class_ids' in f.variables:
                    self.labels = f.variables['class_ids'][:]
                else:
                    raise KeyError("No labels or class_ids found in GlobalTree classification netCDF file.")

        # Optional per-sample year windows (updated GlobalTree: 3y cube + year_window).
        # Prefer h5py even for *.nc (these processed files are HDF5-backed).
        self.year_window = None
        self.time_years = None
        self.year_to_indices = {}
        try:
            with h5py.File(self.data_x, 'r') as f:
                if 'year_window' in f:
                    self.year_window = np.asarray(f['year_window'], dtype=np.int32)
        except Exception:
            try:
                with Dataset(self.data_x, 'r') as f:
                    if 'year_window' in f.variables:
                        self.year_window = np.asarray(f.variables['year_window'][:], dtype=np.int32)
            except Exception as e:
                print(f"[GlobalTree_Classification] year_window load skipped: {e}")

        self.time_years = np.array([int(str(t)[:4]) for t in df_stamp], dtype=np.int32)
        for y in np.unique(self.time_years):
            self.year_to_indices[int(y)] = np.where(self.time_years == int(y))[0]

        print(f"[GlobalTree_Classification-{self.flag}] pixels={self.num_pixels}, time_steps={self.time_steps}")
        if self.year_window is not None:
            n_pair = int(np.sum((self.year_window[:, 0] > 0) & (self.year_window[:, 1] > 0)))
            n_one = int(np.sum((self.year_window[:, 0] > 0) & (self.year_window[:, 1] < 0)))
            print(
                f"[GlobalTree_Classification-{self.flag}] year_window enabled: "
                f"2y={n_pair}, 1y={n_one}, years={sorted(self.year_to_indices)}"
            )

        if self.timeenc == 1:
            try:
                df_stamp = pd.to_datetime(df_stamp, format='%Y%j')
            except Exception:
                df_stamp = pd.to_datetime(df_stamp)
            data_stamp = time_features(df_stamp, freq=self.freq).transpose(1, 0)
        else:
            data_stamp = np.zeros((self.time_steps, 4))

        self.data_stamp = data_stamp

        self.windows_per_sample = 1
        self.window_indices = [0]
        self.num_batches = (self.num_pixels + self.batch_size - 1) // self.batch_size
        self.batch_indices = list(range(self.num_batches))

    def _gather_year_window_sample(self, seq_full: np.ndarray, sample_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Build [T,C] + stamp [T,F] for one sample using year_window (122 steps/year)."""
        yw = self.year_window[sample_idx]
        years = [int(y) for y in yw if int(y) > 0]
        if not years:
            # Fallback: first seq_len steps
            t_use = min(self.seq_len, seq_full.shape[0])
            return seq_full[:t_use], self.data_stamp[:t_use]

        blocks_x = []
        blocks_t = []
        for y in years:
            idx = self.year_to_indices.get(y)
            if idx is None or len(idx) == 0:
                raise KeyError(f"GlobalTree year {y} not found in time axis for sample {sample_idx}")
            blocks_x.append(seq_full[idx])
            blocks_t.append(self.data_stamp[idx])
        x = np.concatenate(blocks_x, axis=0)
        tm = np.concatenate(blocks_t, axis=0)
        if x.shape[0] > self.seq_len:
            x = x[: self.seq_len]
            tm = tm[: self.seq_len]
        elif x.shape[0] < self.seq_len:
            pad = self.seq_len - x.shape[0]
            x = np.concatenate([x, np.repeat(x[-1:, :], pad, axis=0)], axis=0)
            tm = np.concatenate([tm, np.repeat(tm[-1:, :], pad, axis=0)], axis=0)
        return x, tm

    def __iter__(self):
        indices = self.batch_indices[:]

        if dist.is_initialized() and not self.disable_ddp_split:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            batches_per_rank = len(indices) // world_size
            if batches_per_rank > 0:
                indices = indices[:batches_per_rank * world_size]
                indices = indices[rank::world_size]
            else:
                indices = indices if rank == 0 else []

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            indices = indices[worker_id::num_workers]

        if self.flag == 'train':
            random.shuffle(indices)

        if self.file_type == 'h5':
            f = h5py.File(self.data_x, 'r', swmr=True)
        else:
            f = Dataset(self.data_x, 'r')

        try:
            for batch_idx in indices:
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, self.num_pixels)
                batch_pixels = list(range(start, end))

                batch_labels_raw = self.labels[start:end]

                if self.file_type == 'h5':
                    seq_x = f['data'][batch_pixels, :self.time_steps, :]
                else:
                    data_var = f.variables['data'] if 'data' in f.variables else f.variables['images']
                    seq_x = data_var[:, :, batch_pixels]
                    seq_x = np.transpose(seq_x, (2, 0, 1))

                if self.scale:
                    seq_x = (seq_x - self.pre_scaler['mean']) / self.pre_scaler['std']

                if self.year_window is not None:
                    xs, tms = [], []
                    for local_i, pix in enumerate(batch_pixels):
                        x_i, tm_i = self._gather_year_window_sample(seq_x[local_i], pix)
                        xs.append(x_i)
                        tms.append(tm_i)
                    batch_seq_x = np.stack(xs, axis=0)
                    padded_stamp = np.stack(tms, axis=0)
                else:
                    # Legacy: take leading seq_len timesteps
                    s_begin = 0
                    s_end = self.seq_len
                    if s_end > self.time_steps:
                        actual_len = self.time_steps
                        batch_seq_x = seq_x[:, s_begin:actual_len, :]
                        seq_x_mark = self.data_stamp[s_begin:actual_len, :]
                        if actual_len < self.seq_len:
                            pad_len = self.seq_len - actual_len
                            last_timestep = batch_seq_x[:, -1:, :]
                            last_stamp = seq_x_mark[-1:, :]
                            batch_seq_x = np.concatenate(
                                [batch_seq_x, np.tile(last_timestep, (1, pad_len, 1))],
                                axis=1,
                            )
                            seq_x_mark = np.concatenate(
                                [seq_x_mark, np.tile(last_stamp, (pad_len, 1))],
                                axis=0,
                            )
                    else:
                        batch_seq_x = seq_x[:, s_begin:s_end, :]
                        seq_x_mark = self.data_stamp[s_begin:s_end, :]
                    padded_stamp = np.tile(seq_x_mark[None, :, :], (len(batch_pixels), 1, 1))

                yield (
                    torch.tensor(batch_seq_x, dtype=torch.float32),
                    torch.tensor(padded_stamp, dtype=torch.float32),
                    torch.tensor(batch_labels_raw, dtype=torch.long),
                )
        finally:
            f.close()

    def __len__(self):
        import torch.distributed as dist

        total_len = self.num_batches * self.windows_per_sample
        if dist.is_initialized():
            return total_len // dist.get_world_size()
        return total_len


class Dataset_GlobalTree_Segmentation(IterableDataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='', scale=True,
                 timeenc=1, freq='d', sampling_stride=None, batch_size=1000, disable_ddp_split=False):
        super().__init__()
        """
        GlobalTree 分割数据集（与 LCMAP_Segmentation 结构对齐）：
        - 原始 netCDF: global_tree_segmentation_dataset.nc
        - 总长度约 3 年（366 时间步）。
        """
        if size is not None and len(size) > 0:
            self.seq_len = size[0]
        else:
            self.seq_len = 366

        self.flag = flag
        self.features = features
        self.scale = scale
        self.disable_ddp_split = disable_ddp_split
        self.pre_scaler = {
            'mean': np.array([4.0530856e+02, 6.7968939e+02, 7.3541718e+02, 2.5394734e+03,
                              2.0182101e+03, 1.2844141e+03, 5.2847379e-01], dtype=np.float32),
            'std': np.array([2.7406531e+02, 3.4935846e+02, 5.2149530e+02, 9.8295978e+02,
                             9.3158044e+02, 8.0511346e+02, 2.8968227e-01], dtype=np.float32)
        }
        self.timeenc = timeenc
        self.freq = freq
        self.stride = sampling_stride if sampling_stride is not None else 1

        self.root_path = root_path
        self.batch_size = batch_size

        self.__read_data__()

    def __read_data__(self):
        nc_path = os.path.join(self.root_path, "global_tree_segmentation_dataset.nc")
        h5_path = os.path.join(self.root_path, "global_tree_segmentation_dataset.h5")

        if os.path.exists(h5_path):
            self.file_type = 'h5'
            self.data_x = h5_path
        elif os.path.exists(nc_path):
            self.file_type = 'nc'
            self.data_x = nc_path
        else:
            raise FileNotFoundError(f"GlobalTree segmentation file not found in {self.root_path}")

        if self.file_type == 'h5':
            with h5py.File(self.data_x, 'r', swmr=True) as f:
                shape = f['metadata/shape'][:]
                self.num_pixels, self.time_steps, self.bands = shape
                df_stamp = f['time'][:].astype(str)
                if 'yearly_status' in f:
                    self.labels = f['yearly_status'][:]
                else:
                    raise KeyError("No yearly_status found in GlobalTree segmentation HDF5 file.")
        else:
            with Dataset(self.data_x, 'r') as f:
                data_var = f.variables['data'] if 'data' in f.variables else f.variables['images']
                self.time_steps, self.bands, self.num_pixels = data_var.shape
                df_stamp = f.variables['time'][:].astype(str)
                if 'yearly_status' in f.variables:
                    self.labels = f.variables['yearly_status'][:]
                else:
                    raise KeyError("No yearly_status found in GlobalTree segmentation netCDF file.")

        self.num_years = self.labels.shape[1]
        self.steps_per_year = self.time_steps / self.num_years
        print(f"[GlobalTree_Segmentation-{self.flag}] pixels={self.num_pixels}, time_steps={self.time_steps}, years={self.num_years}")

        # 如果 seq_len 大于时间序列长度，则自动截断到 time_steps，避免没有任何窗口可用
        if self.seq_len > self.time_steps:
            print(f"[GlobalTree_Segmentation-{self.flag}] seq_len {self.seq_len} > time_steps {self.time_steps}, 自动截断为 {self.time_steps}")
            self.seq_len = self.time_steps

        if self.timeenc == 1:
            try:
                df_stamp = pd.to_datetime(df_stamp, format='%Y%j')
            except Exception:
                df_stamp = pd.to_datetime(df_stamp)
            data_stamp = time_features(df_stamp, freq=self.freq).transpose(1, 0)
        else:
            data_stamp = np.zeros((self.time_steps, 4))

        self.data_stamp = data_stamp

        self.windows_per_sample = (self.time_steps - self.seq_len) // self.stride + 1
        self.window_indices = list(range(self.windows_per_sample))
        self.num_batches = (self.num_pixels + self.batch_size - 1) // self.batch_size
        self.batch_indices = list(range(self.num_batches))

    def __iter__(self):
        window_indices = self.window_indices[:]
        worker_info = torch.utils.data.get_worker_info()
        all_batch_indices = self.batch_indices[:]

        if dist.is_initialized() and not self.disable_ddp_split:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            batches_per_rank = len(all_batch_indices) // world_size
            if batches_per_rank > 0:
                all_batch_indices = all_batch_indices[:batches_per_rank * world_size]
                all_batch_indices = all_batch_indices[rank::world_size]
            else:
                all_batch_indices = all_batch_indices if rank == 0 else []

        if worker_info is None:
            batch_indices = all_batch_indices
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            batch_indices = all_batch_indices[worker_id::num_workers]

        if self.file_type == 'h5':
            f = h5py.File(self.data_x, 'r', swmr=True)
        else:
            f = Dataset(self.data_x, 'r')

        try:
            for batch_idx in batch_indices:
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, self.num_pixels)
                batch_pixels = list(range(start, end))

                if self.file_type == 'h5':
                    seq_x = f['data'][batch_pixels, :self.time_steps, :]
                else:
                    data_var = f.variables['data'] if 'data' in f.variables else f.variables['images']
                    seq_x = data_var[:, :, batch_pixels]
                    seq_x = np.transpose(seq_x, (2, 0, 1))

                if self.scale:
                    seq_x = (seq_x - self.pre_scaler['mean']) / self.pre_scaler['std']

                for w_idx in window_indices:
                    s_begin = w_idx * self.stride
                    s_end = s_begin + self.seq_len

                    batch_seq_x = seq_x[:, s_begin:s_end, :]
                    seq_x_mark = self.data_stamp[s_begin:s_end, :]

                    padded_stamp = np.tile(seq_x_mark[None, :, :], (len(batch_pixels), 1, 1))

                    year_start_idx = int(s_begin / self.steps_per_year)
                    year_end_idx = int((s_end - 1) / self.steps_per_year)
                    year_start_idx = max(0, min(year_start_idx, self.num_years - 1))
                    year_end_idx = max(0, min(year_end_idx, self.num_years - 1))
                    if year_end_idx < year_start_idx:
                        year_end_idx = year_start_idx
                    year_indices = list(range(year_start_idx, year_end_idx + 1))
                    batch_labels = self.labels[start:end, year_indices]

                    yield (
                        torch.tensor(batch_seq_x, dtype=torch.float32),
                        torch.tensor(padded_stamp, dtype=torch.float32),
                        torch.tensor(batch_labels, dtype=torch.long),
                    )
        finally:
            f.close()

    def __len__(self):
        return self.num_batches * self.windows_per_sample


class Dataset_GlanceTraining_Segmentation(IterableDataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='', scale=True,
                 timeenc=1, freq='d', sampling_stride=None, batch_size=1000, disable_ddp_split=False):
        super().__init__()
        """
        GlanceTraining 二元变化检测数据集（序列级分割 / Change vs Stable）.

        假设 NetCDF/HDF5 结构与分类任务类似:
        - data:   [time, bands, samples] 或 [samples, time, bands]（通过维度名或形状自动判断）
        - labels: [samples]，0 = Stable, 1 = Transitional.

        与 Dataset_GlanceTraining_Classification 的差异:
        - 标签为二元变化标签（是否属于变化片段），而非多类 land cover.
        - 仍然采用「每个样本单窗口」模式：不做滑动窗口，seq_len 控制时间长度，
          若 seq_len 大于实际长度则使用最后一个时间步进行复制填充。
        """
        if size is not None and len(size) > 0:
            self.seq_len = size[0]
        else:
            self.seq_len = 366

        self.flag = flag
        self.features = features
        self.scale = scale
        self.disable_ddp_split = disable_ddp_split
        self.pre_scaler = {
            'mean': np.array([4.0530856e+02, 6.7968939e+02, 7.3541718e+02, 2.5394734e+03,
                              2.0182101e+03, 1.2844141e+03, 5.2847379e-01], dtype=np.float32),
            'std': np.array([2.7406531e+02, 3.4935846e+02, 5.2149530e+02, 9.8295978e+02,
                             9.3158044e+02, 8.0511346e+02, 2.8968227e-01], dtype=np.float32)
        }
        self.timeenc = timeenc
        self.freq = freq
        # 默认不滑动窗口：每个样本一个窗口
        self.stride = sampling_stride if sampling_stride is not None else self.seq_len
        print(f">>> [Dataset_GlanceTraining_Segmentation] seq_len: {self.seq_len}, stride: {self.stride}, sampling_stride param: {sampling_stride}")

        # 固定下游 GlanceTraining segmentation 路径
        self.root_path = root_path
        self.batch_size = batch_size

        self.__read_data__()

    def __read_data__(self):
        """
        读取 GlanceTraining 二元变化检测 NetCDF/HDF5 文件元数据与标签.
        """
        nc_path = os.path.join(self.root_path, "glancetraining_segmentation_dataset.nc")
        h5_path = os.path.join(self.root_path, "glancetraining_segmentation_dataset.h5")

        if os.path.exists(h5_path):
            self.file_type = 'h5'
            self.data_x = h5_path
        elif os.path.exists(nc_path):
            self.file_type = 'nc'
            self.data_x = nc_path
        else:
            raise FileNotFoundError(f"GlanceTraining segmentation file not found in {self.root_path}")

        if self.file_type == 'h5':
            with h5py.File(self.data_x, 'r', swmr=True) as f:
                # 与其他下游数据集保持一致，优先使用 metadata/shape
                if 'metadata/shape' in f:
                    shape = f['metadata/shape'][:]
                    self.num_pixels, self.time_steps, self.bands = shape
                    data_layout = 'pixels_time_bands'
                else:
                    # 回退: 直接从 data 维度推断
                    data_var = f['data']
                    shape = data_var.shape
                    if shape[0] > shape[1] and shape[0] > shape[2]:
                        # (pixels, time, bands)
                        self.num_pixels, self.time_steps, self.bands = shape
                        data_layout = 'pixels_time_bands'
                    else:
                        # (time, bands, pixels)
                        self.time_steps, self.bands, self.num_pixels = shape
                        data_layout = 'time_bands_pixels'
                self.data_layout = data_layout
                df_stamp = f['time'][:].astype(str)

                # 标签字段名容错: 优先 labels, 回退 change_labels / binary_labels
                if 'labels' in f:
                    self.labels = f['labels'][:]
                elif 'change_labels' in f:
                    self.labels = f['change_labels'][:]
                elif 'binary_labels' in f:
                    self.labels = f['binary_labels'][:]
                else:
                    raise KeyError("No labels / change_labels / binary_labels found in GlanceTraining segmentation HDF5 file.")
        else:
            with Dataset(self.data_x, 'r') as f:
                data_var = f.variables['data'] if 'data' in f.variables else f.variables['images']
                shape = data_var.shape
                # 优先根据维度名判断，否则按形状启发式判断
                if hasattr(data_var, 'dimensions'):
                    dims = data_var.dimensions
                    if dims[0] == 'samples' or dims[0] == 'pixels':
                        self.num_pixels, self.time_steps, self.bands = shape
                        data_layout = 'pixels_time_bands'
                    else:
                        self.time_steps, self.bands, self.num_pixels = shape
                        data_layout = 'time_bands_pixels'
                else:
                    if shape[0] > shape[1] and shape[0] > shape[2]:
                        self.num_pixels, self.time_steps, self.bands = shape
                        data_layout = 'pixels_time_bands'
                    else:
                        self.time_steps, self.bands, self.num_pixels = shape
                        data_layout = 'time_bands_pixels'
                self.data_layout = data_layout
                df_stamp = f.variables['time'][:].astype(str)

                # 标签字段名容错
                if 'labels' in f.variables:
                    self.labels = f.variables['labels'][:]
                elif 'change_labels' in f.variables:
                    self.labels = f.variables['change_labels'][:]
                elif 'binary_labels' in f.variables:
                    self.labels = f.variables['binary_labels'][:]
                else:
                    raise KeyError("No labels / change_labels / binary_labels found in GlanceTraining segmentation netCDF file.")

        print(f"[GlanceTraining_Segmentation-{self.flag}] pixels={self.num_pixels}, time_steps={self.time_steps}")

        # 时间编码
        if self.timeenc == 1:
            try:
                df_stamp = pd.to_datetime(df_stamp, format='%Y%j')
            except Exception:
                df_stamp = pd.to_datetime(df_stamp)
            data_stamp = time_features(df_stamp, freq=self.freq).transpose(1, 0)
        else:
            data_stamp = np.zeros((self.time_steps, 4))

        self.data_stamp = data_stamp

        # 每个样本只返回一个窗口（从 0 开始，长度为 seq_len）
        self.windows_per_sample = 1
        self.window_indices = [0]
        self.num_batches = (self.num_pixels + self.batch_size - 1) // self.batch_size
        self.batch_indices = list(range(self.num_batches))

    def __iter__(self):
        """
        迭代样本，返回 (batch_x, batch_x_mark, labels) 三元组:
        - batch_x:       [B, seq_len, C]
        - batch_x_mark:  [B, seq_len, time_feat]
        - labels:        [B]，二元变化标签.
        """
        window_indices = self.window_indices[:]
        all_batch_indices = self.batch_indices[:]

        # DDP 进程级切分
        if dist.is_initialized() and not self.disable_ddp_split:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            batches_per_rank = len(all_batch_indices) // world_size
            if batches_per_rank > 0:
                all_batch_indices = all_batch_indices[:batches_per_rank * world_size]
                all_batch_indices = all_batch_indices[rank::world_size]
            else:
                all_batch_indices = all_batch_indices if rank == 0 else []

        # Worker 级切分
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            batch_indices = all_batch_indices
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            batch_indices = all_batch_indices[worker_id::num_workers]

        # 打开文件句柄
        if self.file_type == 'h5':
            f = h5py.File(self.data_x, 'r', swmr=True)
        else:
            f = Dataset(self.data_x, 'r')

        try:
            for batch_idx in batch_indices:
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, self.num_pixels)
                batch_pixels = list(range(start, end))

                batch_labels_raw = self.labels[start:end]

                # 读取数据并根据布局转换为 [B, T, C]
                if self.file_type == 'h5':
                    if self.data_layout == 'pixels_time_bands':
                        seq_x = f['data'][batch_pixels, :self.time_steps, :]
                    else:
                        seq_x = f['data'][:self.time_steps, :, batch_pixels]
                        seq_x = np.transpose(seq_x, (2, 0, 1))
                else:
                    data_var = f.variables['data'] if 'data' in f.variables else f.variables['images']
                    if self.data_layout == 'pixels_time_bands':
                        seq_x = data_var[batch_pixels, :self.time_steps, :]
                    else:
                        seq_x = data_var[:, :, batch_pixels]
                        seq_x = np.transpose(seq_x, (2, 0, 1))

                if self.scale:
                    seq_x = (seq_x - self.pre_scaler['mean']) / self.pre_scaler['std']

                for w_idx in window_indices:
                    s_begin = w_idx * self.stride
                    s_end = s_begin + self.seq_len

                    if s_end > self.time_steps:
                        actual_len = self.time_steps
                        batch_seq_x = seq_x[:, s_begin:actual_len, :]
                        seq_x_mark = self.data_stamp[s_begin:actual_len, :]
                        if actual_len < self.seq_len:
                            pad_len = self.seq_len - actual_len
                            last_timestep = batch_seq_x[:, -1:, :]
                            last_stamp = seq_x_mark[-1:, :]
                            batch_seq_x = np.concatenate(
                                [batch_seq_x, np.tile(last_timestep, (1, pad_len, 1))],
                                axis=1,
                            )
                            seq_x_mark = np.concatenate(
                                [seq_x_mark, np.tile(last_stamp, (pad_len, 1))],
                                axis=0,
                            )
                    else:
                        batch_seq_x = seq_x[:, s_begin:s_end, :]
                        seq_x_mark = self.data_stamp[s_begin:s_end, :]

                    padded_stamp = np.tile(seq_x_mark[None, :, :], (len(batch_pixels), 1, 1))

                    yield (
                        torch.tensor(batch_seq_x, dtype=torch.float32),
                        torch.tensor(padded_stamp, dtype=torch.float32),
                        torch.tensor(batch_labels_raw, dtype=torch.long),
                    )
        finally:
            f.close()

    def __len__(self):
        return self.num_batches * self.windows_per_sample


class Dataset_CDL_Classification(IterableDataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='', scale=True,
                 timeenc=1, freq='d', sampling_stride=None, batch_size=1000, disable_ddp_split=False):
        super().__init__()
        """
        CDL 单年分类数据集：
        - 文件路径示例：/intelnvme01/ziyun/DownStreamTasks/CDL/outdir/classification_dataset_processed.nc
        - 时间长度约 122（1 年）。
        - 标签字段兼容 'labels' / 'class_ids'。
        """
        if size is not None and len(size) > 0:
            self.seq_len = size[0]
        else:
            self.seq_len = 122

        self.flag = flag
        self.features = features
        self.scale = scale
        self.disable_ddp_split = disable_ddp_split
        self.pre_scaler = {
            'mean': np.array([4.0530856e+02, 6.7968939e+02, 7.3541718e+02, 2.5394734e+03,
                              2.0182101e+03, 1.2844141e+03, 5.2847379e-01], dtype=np.float32),
            'std': np.array([2.7406531e+02, 3.4935846e+02, 5.2149530e+02, 9.8295978e+02,
                             9.3158044e+02, 8.0511346e+02, 2.8968227e-01], dtype=np.float32)
        }
        self.timeenc = timeenc
        self.freq = freq
        self.stride = sampling_stride if sampling_stride is not None else self.seq_len
        print(f">>> [Dataset_CDL_Classification] seq_len: {self.seq_len}, stride: {self.stride}, sampling_stride param: {sampling_stride}")

        self.root_path = root_path
        self.batch_size = batch_size

        self.__read_data__()

    def __read_data__(self):
        h5_candidates = [
            os.path.join(self.root_path, "cdl_classification_dataset.h5"),
        ]
        nc_candidates = [
            os.path.join(self.root_path, "cdl_hls_classification_processed.nc"),
            os.path.join(self.root_path, "cdl_classification_dataset.nc"),
        ]

        self.file_type = None
        self.data_x = None
        for p in h5_candidates:
            if os.path.exists(p):
                self.file_type = 'h5'
                self.data_x = p
                break
        if self.data_x is None:
            for p in nc_candidates:
                if os.path.exists(p):
                    self.file_type = 'nc'
                    self.data_x = p
                    break
        if self.data_x is None:
            raise FileNotFoundError(f"CDL classification file not found in {self.root_path}")

        if self.file_type == 'h5':
            with h5py.File(self.data_x, 'r', swmr=True) as f:
                shape = f['metadata/shape'][:]
                self.num_pixels, self.time_steps, self.bands = shape
                df_stamp = f['time'][:].astype(str)

                if 'labels' in f:
                    self.labels = f['labels'][:]
                elif 'class_ids' in f:
                    self.labels = f['class_ids'][:]
                else:
                    raise KeyError("No labels or class_ids found in CDL classification HDF5 file.")
        else:
            with Dataset(self.data_x, 'r') as f:
                data_var = f.variables['data'] if 'data' in f.variables else f.variables['images']
                self.time_steps, self.bands, self.num_pixels = data_var.shape
                df_stamp = f.variables['time'][:].astype(str)

                if 'labels' in f.variables:
                    self.labels = f.variables['labels'][:]
                elif 'class_ids' in f.variables:
                    self.labels = f.variables['class_ids'][:]
                else:
                    raise KeyError("No labels or class_ids found in CDL classification netCDF file.")

        print(f"[CDL_Classification-{self.flag}] pixels={self.num_pixels}, time_steps={self.time_steps}")

        if self.timeenc == 1:
            try:
                df_stamp = pd.to_datetime(df_stamp, format='%Y%j')
            except Exception:
                df_stamp = pd.to_datetime(df_stamp)
            data_stamp = time_features(df_stamp, freq=self.freq).transpose(1, 0)
        else:
            data_stamp = np.zeros((self.time_steps, 4))

        self.data_stamp = data_stamp

        self.windows_per_sample = 1
        self.window_indices = [0]
        self.num_batches = (self.num_pixels + self.batch_size - 1) // self.batch_size
        self.batch_indices = list(range(self.num_batches))

    def __iter__(self):
        indices = self.batch_indices[:]

        if dist.is_initialized() and not self.disable_ddp_split:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            batches_per_rank = len(indices) // world_size
            if batches_per_rank > 0:
                indices = indices[:batches_per_rank * world_size]
                indices = indices[rank::world_size]
            else:
                indices = indices if rank == 0 else []

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            indices = indices[worker_id::num_workers]

        if self.flag == 'train':
            random.shuffle(indices)

        if self.file_type == 'h5':
            f = h5py.File(self.data_x, 'r', swmr=True)
        else:
            f = Dataset(self.data_x, 'r')

        try:
            for batch_idx in indices:
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, self.num_pixels)
                batch_pixels = list(range(start, end))

                batch_labels_raw = self.labels[start:end]

                if self.file_type == 'h5':
                    seq_x = f['data'][batch_pixels, :self.time_steps, :]
                else:
                    data_var = f.variables['data'] if 'data' in f.variables else f.variables['images']
                    seq_x = data_var[:, :, batch_pixels]
                    seq_x = np.transpose(seq_x, (2, 0, 1))

                if self.scale:
                    seq_x = (seq_x - self.pre_scaler['mean']) / self.pre_scaler['std']

                s_begin = 0
                s_end = self.seq_len

                if s_end > self.time_steps:
                    actual_len = self.time_steps
                    batch_seq_x = seq_x[:, s_begin:actual_len, :]
                    seq_x_mark = self.data_stamp[s_begin:actual_len, :]
                    if actual_len < self.seq_len:
                        pad_len = self.seq_len - actual_len
                        last_timestep = batch_seq_x[:, -1:, :]
                        last_stamp = seq_x_mark[-1:, :]
                        batch_seq_x = np.concatenate(
                            [batch_seq_x, np.tile(last_timestep, (1, pad_len, 1))],
                            axis=1,
                        )
                        seq_x_mark = np.concatenate(
                            [seq_x_mark, np.tile(last_stamp, (pad_len, 1))],
                            axis=0,
                        )
                else:
                    batch_seq_x = seq_x[:, s_begin:s_end, :]
                    seq_x_mark = self.data_stamp[s_begin:s_end, :]

                padded_stamp = np.tile(seq_x_mark[None, :, :], (len(batch_pixels), 1, 1))

                yield (
                    torch.tensor(batch_seq_x, dtype=torch.float32),
                    torch.tensor(padded_stamp, dtype=torch.float32),
                    torch.tensor(batch_labels_raw, dtype=torch.long),
                )
        finally:
            f.close()

    def __len__(self):
        import torch.distributed as dist

        total_len = self.num_batches * self.windows_per_sample
        if dist.is_initialized():
            return total_len // dist.get_world_size()
        return total_len


class Dataset_GlanceTraining_Classification(IterableDataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='', scale=True,
                 timeenc=1, freq='d', sampling_stride=None, batch_size=1000, disable_ddp_split=False):
        super().__init__()
        """
        整体逻辑（GlanceTraining 分类 + KNN 探针）：
        1）长度设定
           - 外部通过 size[0] 传入 seq_len（与主模型的 args.seq_len 对齐，例如 366 / 732）；
           - 如果 size 为空，则默认 seq_len = 366。
        2）时间轴与窗口
           - 从 HDF5 / NetCDF 读取整段时间序列 [B, time_steps, C]；
           - 与 Dataset_LCMAP_Classification 保持一致，这里也不再做滑动窗口：
               * windows_per_sample = 1
               * window_indices = [0]
               * 窗口范围固定为 [0 : seq_len]。
           - 之前版本这里使用 stride 和滑窗，现在为了 KNN / 下游评估与主模型保持一致，改为固定长度单窗口。
        3）长度对齐
           - 假设 time_steps ≥ seq_len（HLS/Glance 数据通常是完整年度栅格），则直接切片 [0 : seq_len]；
           - 如果未来有 time_steps < seq_len 的情况，可参考 Dataset_LCMAP_Classification 的逻辑做额外 padding。
        4）时间编码
           - 使用 self.data_stamp 上对应 [0 : seq_len] 的时间特征；
           - 将 [seq_len, time_feat] broadcast 成 [B, seq_len, time_feat] 与 batch 对齐。
        5）返回给模型
           - 返回 (batch_x, batch_x_mark, labels)，其中：
               * batch_x: [B, seq_len, C]；
               * batch_x_mark: [B, seq_len, time_feat]；
               * labels: [B]，GlanceTraining 中是重映射过的 class_ids。
        6）KNN 调用方式
           - ExpProbe.knn_probe_GlanceTraining_Classification 中，batch_x / batch_x_mark 送入 model.backbone.encode(...)，
             使用 outputs['cls_token'] 作为序列级表示做 KNN 分类，并同时计算 raw mean baseline。
        """
        # size [seq_len, label_len, pred_len] -> 分类任务只关心 seq_len (窗口大小)
        # 使用size参数，如果没有提供则使用默认值366
        if size is not None and len(size) > 0:
            self.seq_len = size[0]
        else:
            self.seq_len = 366
        
        # 初始化设置
        self.flag = flag
        self.features = features
        self.scale = scale
        self.disable_ddp_split = disable_ddp_split  # 用于KNN Probe等场景，禁用DDP切分
        # 保持原有的 hard-coded scaler
        self.pre_scaler = {
            'mean': np.array([4.0530856e+02, 6.7968939e+02, 7.3541718e+02, 2.5394734e+03,
                              2.0182101e+03, 1.2844141e+03, 5.2847379e-01], dtype=np.float32),
            'std': np.array([2.7406531e+02, 3.4935846e+02, 5.2149530e+02, 9.8295978e+02,
                             9.3158044e+02, 8.0511346e+02, 2.8968227e-01], dtype=np.float32)
        }
        self.timeenc = timeenc
        self.freq = freq
        # 使用sampling_stride参数，如果没有提供则使用seq_len（即不重叠）
        self.stride = sampling_stride if sampling_stride is not None else self.seq_len
        print(f">>> [Dataset_GlanceTraining_Classification] seq_len: {self.seq_len}, stride: {self.stride}, sampling_stride param: {sampling_stride}")
        self.root_path = root_path
        # self.data_path = data_path
        self.batch_size = batch_size
        
        self.__read_data__()

    def __read_data__(self):
        # 1. 确定文件路径和类型（兼容新旧命名）
        h5_candidates = [
            os.path.join(self.root_path, "glancetraining_classification_dataset.h5"),
        ]
        nc_candidates = [
            os.path.join(self.root_path, "glancetraining_hls_classification_processed.nc"),
            os.path.join(self.root_path, "glancetraining_classification_dataset.nc"),
        ]

        self.file_type = None
        self.data_x = None
        for p in h5_candidates:
            if os.path.exists(p):
                self.file_type = 'h5'
                self.data_x = p
                break
        if self.data_x is None:
            for p in nc_candidates:
                if os.path.exists(p):
                    self.file_type = 'nc'
                    self.data_x = p
                    break
        if self.data_x is None:
            raise FileNotFoundError(f"File not found in {self.root_path}")

        # 2. 读取元数据和全部 Label (因为 Label 比较小，可以一次读入内存)
        if self.file_type == 'h5':
            with h5py.File(self.data_x, 'r', swmr=True) as f:
                shape = f['metadata/shape'][:]
                self.num_pixels, self.time_steps, self.bands = shape
                df_stamp = f['time'][:].astype(str)
                self.labels = f['class_ids'][:] # 加载 Label [num_pixels]
                unique_ids = np.unique(self.labels)

                # 例如：1->0, 5->1, 24->2
                self.labels = np.searchsorted(unique_ids, self.labels)

                # 4. 自动更新类别数 (防止你手动写错 Classes: 6)
                self.num_classes = len(unique_ids)
        else: # netCDF4
            with Dataset(self.data_x, 'r') as f:
                # 注意：NetCDF 生成脚本中 data 维度是 (time, bands, samples)，这里需要确认 dataset 实现是否转置
                # 根据之前的 Dataset_HLS 逻辑，这里假设 nc 文件读取时需要转置或者通过维度名获取
                self.time_steps = f.dimensions['time'].size
                self.bands = f.dimensions['bands'].size
                self.num_pixels = f.dimensions['samples'].size

                # for var_name, var_obj in f.variables.items():
                #     # var_obj.dimensions 返回维度名称的元组，如 ('time', 'bands', 'samples')
                #     # var_obj.shape 返回具体数值，如 (365, 10, 1000)
                #     print(f"  Name: {var_name:<15} | Shape: {str(var_obj.shape):<20} | Dims: {var_obj.dimensions}")

                df_stamp = f.variables['time'][:].astype(str)
                self.labels = f.variables['class_ids'][:] # 加载 Label [num_pixels]
                unique_ids = np.unique(self.labels)

                # 例如：1->0, 5->1, 24->2
                self.labels = np.searchsorted(unique_ids, self.labels)

                # 4. 自动更新类别数 (防止你手动写错 Classes: 6)
                self.num_classes = len(unique_ids)

                # # --- 调试信息 (必看) ---
                # print(f"\n[Label Remap Success]")
                # print(f"Original IDs found: {unique_ids}")
                # print(f"Mapped to:        {np.arange(self.num_classes)}")
                # print(f"Total Classes:    {self.num_classes}")
                # print(f"Label Shape:      {self.labels.shape}")
                # print("-" * 30)
                # print(f"Dataset ({self.flag}): {self.num_pixels} pixels, {self.time_steps} steps, Labels loaded.")

        # 3. 时间编码处理
        if self.timeenc == 1:
            try:
                df_stamp = pd.to_datetime(df_stamp, format='%Y%j')
            except:
                df_stamp = pd.to_datetime(df_stamp) # 尝试自动推断
            data_stamp = time_features(df_stamp, freq=self.freq).transpose(1, 0)
        else:
            data_stamp = np.zeros((self.time_steps, 4))
        
        self.data_stamp = data_stamp
        
        # 4. 计算迭代参数
        # 探针任务：与 Dataset_LCMAP_Classification 保持一致，使用固定长度读取，不做滑动窗口
        # 每个像素只返回一个窗口（从 0 开始，长度为 seq_len）
        self.windows_per_sample = 1
        self.window_indices = [0]
        self.num_batches = (self.num_pixels + self.batch_size - 1) // self.batch_size
        self.batch_indices = list(range(self.num_batches))
        
        print(f">>> [Dataset_GlanceTraining_Classification] 固定长度读取模式:")
        print(f"    time_steps={self.time_steps}, seq_len={self.seq_len}")
        print(f"    windows_per_sample=1 (不使用滑动窗口)")
        print(f"    num_pixels={self.num_pixels}, batch_size={self.batch_size}")
        print(f"    num_batches={self.num_batches}")
        print(f"    Total samples = {self.num_batches} batches × 1 window = {self.num_batches}")

    def __iter__(self):
        # 不进行 Shuffle，严格顺序读取
        window_indices = self.window_indices[:]

        worker_info = torch.utils.data.get_worker_info()
        
        all_batch_indices = self.batch_indices[:]
        
        # DDP 进程级切分（如果未禁用）
        if dist.is_initialized() and not self.disable_ddp_split:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            
            # 关键修复：确保每个进程处理相同数量的 batch
            # 如果总数不能被 world_size 整除，截断到相同长度
            batches_per_rank = len(all_batch_indices) // world_size
            if batches_per_rank > 0:
                # 截断到能被 world_size 整除的长度
                all_batch_indices = all_batch_indices[:batches_per_rank * world_size]
                # 然后按 rank 切分
                all_batch_indices = all_batch_indices[rank::world_size]
            else:
                # 如果 batch 数量太少，只让 rank 0 处理
                all_batch_indices = all_batch_indices if rank == 0 else []
        
        if worker_info is None:  
            batch_indices = all_batch_indices
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            batch_indices = all_batch_indices[worker_id::num_workers]
        # ----------------------------------

        # 打开文件句柄
        if self.file_type == 'h5':
            f = h5py.File(self.data_x, 'r', swmr=True)
        else:
            f = Dataset(self.data_x, 'r')

        try:
            for batch_idx in batch_indices:
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, self.num_pixels)
                batch_pixels = list(range(start, end))
                
                # 获取当前 Batch 对应的 Labels
                batch_labels_raw = self.labels[start:end] # [Batch_Size]

                # 读取数据 (IO操作)
                if self.file_type == 'h5':
                    seq_x = f['data'][batch_pixels, :self.time_steps, :] # [B, T, C]
                else:
                    # NetCDF: (T, C, B) -> 转置为 (B, T, C)
                    seq_x = f.variables['data'][:, :, batch_pixels]
                    seq_x = np.transpose(seq_x, (2, 0, 1))

                # 归一化
                if self.scale:
                    seq_x = (seq_x - self.pre_scaler['mean']) / self.pre_scaler['std']

                # 滑动窗口切片 (内存操作)
                for w_idx in window_indices:
                    s_begin = w_idx * self.stride
                    s_end = s_begin + self.seq_len
                    
                    batch_seq_x = seq_x[:, s_begin:s_end, :] # [B, seq_len, C]
                    seq_x_mark = self.data_stamp[s_begin:s_end, :] # [seq_len, time_feat]
                    
                    # 扩展时间戳以匹配 Batch 维度
                    padded_stamp = np.tile(seq_x_mark[None, :, :], (len(batch_pixels), 1, 1))

                    # 返回 (Input, Time, Label)
                    # Label 不需要时间维度，它是该像素的静态属性
                    yield (
                        torch.tensor(batch_seq_x, dtype=torch.float32),
                        torch.tensor(padded_stamp, dtype=torch.float32),
                        torch.tensor(batch_labels_raw, dtype=torch.long) # 分类标签通常用 long
                    )

        finally:
            f.close()

    def __len__(self):
        return self.num_batches * self.windows_per_sample


class Dataset_CropHarvest_Classification(IterableDataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='', scale=True,
                 timeenc=1, freq='d', sampling_stride=None, batch_size=1000, disable_ddp_split=False):
        super().__init__()
        if size is not None and len(size) > 0:
            self.seq_len = size[0]
        else:
            self.seq_len = 366

        self.flag = flag
        self.features = features
        self.scale = scale
        self.disable_ddp_split = disable_ddp_split
        self.pre_scaler = {
            'mean': np.array([4.0530856e+02, 6.7968939e+02, 7.3541718e+02, 2.5394734e+03,
                              2.0182101e+03, 1.2844141e+03, 5.2847379e-01], dtype=np.float32),
            'std': np.array([2.7406531e+02, 3.4935846e+02, 5.2149530e+02, 9.8295978e+02,
                             9.3158044e+02, 8.0511346e+02, 2.8968227e-01], dtype=np.float32)
        }
        self.timeenc = timeenc
        self.freq = freq
        self.stride = sampling_stride if sampling_stride is not None else self.seq_len
        print(f">>> [Dataset_CropHarvest_Classification] seq_len: {self.seq_len}, stride: {self.stride}, sampling_stride param: {sampling_stride}")
        self.root_path = root_path
        self.batch_size = batch_size

        self.__read_data__()

    def __read_data__(self):
        h5_candidates = [
            os.path.join(self.root_path, "cropharvest_classification_dataset.h5"),
        ]
        nc_candidates = [
            os.path.join(self.root_path, "cropharvest_hls_classification_processed.nc"),
            os.path.join(self.root_path, "cropharvest_classification_dataset.nc"),
        ]

        self.file_type = None
        self.data_x = None
        for p in h5_candidates:
            if os.path.exists(p):
                self.file_type = 'h5'
                self.data_x = p
                break
        if self.data_x is None:
            for p in nc_candidates:
                if os.path.exists(p):
                    self.file_type = 'nc'
                    self.data_x = p
                    break
        if self.data_x is None:
            raise FileNotFoundError(f"File not found in {self.root_path}")

        if self.file_type == 'h5':
            with h5py.File(self.data_x, 'r', swmr=True) as f:
                shape = f['metadata/shape'][:]
                self.num_pixels, self.time_steps, self.bands = shape
                df_stamp = f['time'][:].astype(str)
                if 'labels' in f:
                    self.labels = f['labels'][:]
                elif 'class_ids' in f:
                    self.labels = f['class_ids'][:]
                else:
                    raise KeyError("No labels or class_ids found in CropHarvest classification HDF5 file.")
        else:
            with Dataset(self.data_x, 'r') as f:
                self.time_steps = f.dimensions['time'].size
                self.bands = f.dimensions['bands'].size
                self.num_pixels = f.dimensions['samples'].size
                df_stamp = f.variables['time'][:].astype(str)
                if 'labels' in f.variables:
                    self.labels = f.variables['labels'][:]
                elif 'class_ids' in f.variables:
                    self.labels = f.variables['class_ids'][:]
                else:
                    raise KeyError("No labels or class_ids found in CropHarvest classification netCDF file.")

        if self.timeenc == 1:
            try:
                df_stamp = pd.to_datetime(df_stamp, format='%Y%j')
            except Exception:
                df_stamp = pd.to_datetime(df_stamp)
            data_stamp = time_features(df_stamp, freq=self.freq).transpose(1, 0)
        else:
            data_stamp = np.zeros((self.time_steps, 4))

        self.data_stamp = data_stamp
        self.windows_per_sample = 1
        self.window_indices = [0]
        self.num_batches = (self.num_pixels + self.batch_size - 1) // self.batch_size
        self.batch_indices = list(range(self.num_batches))

    def __iter__(self):
        indices = self.batch_indices[:]
        if dist.is_initialized() and not self.disable_ddp_split:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            batches_per_rank = len(indices) // world_size
            if batches_per_rank > 0:
                indices = indices[:batches_per_rank * world_size]
                indices = indices[rank::world_size]
            else:
                indices = indices if rank == 0 else []

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            indices = indices[worker_info.id::worker_info.num_workers]

        if self.file_type == 'h5':
            f = h5py.File(self.data_x, 'r', swmr=True)
        else:
            f = Dataset(self.data_x, 'r')
        try:
            for batch_idx in indices:
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, self.num_pixels)
                batch_pixels = list(range(start, end))
                batch_labels_raw = self.labels[start:end]

                if self.file_type == 'h5':
                    seq_x = f['data'][batch_pixels, :self.time_steps, :]
                else:
                    seq_x = f.variables['data'][:, :, batch_pixels]
                    seq_x = np.transpose(seq_x, (2, 0, 1))

                if self.scale:
                    seq_x = (seq_x - self.pre_scaler['mean']) / self.pre_scaler['std']

                s_begin = 0
                s_end = min(self.seq_len, self.time_steps)
                batch_seq_x = seq_x[:, s_begin:s_end, :]
                seq_x_mark = self.data_stamp[s_begin:s_end, :]
                if s_end < self.seq_len:
                    pad_len = self.seq_len - s_end
                    last_timestep = batch_seq_x[:, -1:, :]
                    last_stamp = seq_x_mark[-1:, :]
                    batch_seq_x = np.concatenate(
                        [batch_seq_x, np.tile(last_timestep, (1, pad_len, 1))],
                        axis=1,
                    )
                    seq_x_mark = np.concatenate(
                        [seq_x_mark, np.tile(last_stamp, (pad_len, 1))],
                        axis=0,
                    )
                padded_stamp = np.tile(seq_x_mark[None, :, :], (len(batch_pixels), 1, 1))

                yield (
                    torch.tensor(batch_seq_x, dtype=torch.float32),
                    torch.tensor(padded_stamp, dtype=torch.float32),
                    torch.tensor(batch_labels_raw, dtype=torch.long),
                )
        finally:
            f.close()

    def __len__(self):
        return self.num_batches * self.windows_per_sample
