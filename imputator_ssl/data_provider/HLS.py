import os
from collections import defaultdict
import numpy as np
from datetime import datetime
from typing import Tuple, Union, List

from joblib import Parallel, delayed
import warnings

try:
    from joblib import Parallel, delayed
    parallel_available = True
except ImportError:
    parallel_available = False

class FmaskProcessor:
    def __init__(self, fmask, dimension=3):
        """
        初始化 FmaskProcessor 类
        :param fmask: numpy 数组, 如果 dimension=1，则形状为 [num_images]；如果 dimension=3，则形状为 [num_images, width, height]
        :param dimension: Fmask 的维度，1 表示 [num_images]，3 表示 [num_images, width, height]，默认值为 3
        """
        if dimension not in [1, 3]:
            raise ValueError("dimension 参数必须为 1 或 3！")

        self.fmask = fmask
        self.dimension = dimension

    def create_mask(self, cirrus=False, cloud=True, adj_could=False, shadow=False, snow_ice=True, water=False):
        """
        将 Fmask 数据转为掩膜
        :param cirrus: 是否包括卷云 (Bit 0)
        :param cloud: 是否包括云 (Bit 1)
        :param adj_could: 是否临近云 (Bit 2)
        :param shadow: 是否包括云阴影 (Bit 3)
        :param snow_ice: 是否包括雪/冰 (Bit 4)
        :param water: 是否包括水 (Bit 5)
        :return: 掩膜数组，与 Fmask 形状相同，有效像素为 1，无效像素为 0
        """
        if self.dimension == 1:
            mask = np.ones_like(self.fmask, dtype=np.uint8)  # 初始化为有效像素 (1)
        elif self.dimension == 3:
            mask = np.ones_like(self.fmask, dtype=np.uint8)

        # 按需排除指定条件的像素
        if cirrus:
            mask &= (self.fmask & (1 << 0)) == 0
        if cloud:
            mask &= (self.fmask & (1 << 1)) == 0
        if adj_could:
            mask &= (self.fmask & (1 << 2)) == 0
        if shadow:
            mask &= (self.fmask & (1 << 3)) == 0
        if snow_ice:
            mask &= (self.fmask & (1 << 4)) == 0
        if water:
            mask &= (self.fmask & (1 << 5)) == 0

        return mask

    def apply_mask(self, data, mask):
        """
        利用生成的掩膜对数据进行掩膜处理
        :param data: 输入数据, 如果 dimension=1，则形状为 [num_images, bands]；如果 dimension=3，则形状为 [num_images, bands, width, height]
        :param mask: 掩膜数组，与 Fmask 的形状相同
        :return: 掩膜后的数据, 无效像素设置为 NaN
        """
        if self.dimension == 1:
            # 检查形状是否匹配
            if data.shape[0] != mask.shape[0]:
                raise ValueError("掩膜和数据的形状不匹配！")
            # 扩展掩膜到数据的维度
            expanded_mask = mask[:, np.newaxis]  # 扩展维度到 [num_images, 1]
            masked_data = np.where(expanded_mask == 1, data, np.nan)
        elif self.dimension == 3:
            # 检查形状是否匹配
            if data.shape[0] != mask.shape[0] or data.shape[2:] != mask.shape[1:]:
                raise ValueError("掩膜和数据的形状不匹配！")
            # 扩展掩膜到数据的维度
            expanded_mask = mask[:, np.newaxis, :, :]  # 扩展维度到 [num_images, 1, width, height]
            masked_data = np.where(expanded_mask == 1, data, np.nan)

        return masked_data


def _check_dimensions(series: np.ndarray) -> Tuple[bool, tuple]:
    """检查输入数组的维度并确定处理模式"""
    if series.ndim == 4:
        return True, series.shape
    elif series.ndim == 3:
        return False, series.shape
    else:
        raise ValueError("Input series must be 3D or 4D array")

def _get_output_shape(is_mode1: bool, unique_years: np.ndarray,
                     steps_per_year: int, series: np.ndarray) -> tuple:
    """确定输出数组的形状"""
    num_years = len(unique_years)
    if is_mode1:
        return (num_years, steps_per_year, *series.shape[1:])
    else:
        # 修改为新的维度顺序
        return (num_years, steps_per_year, series.shape[1], series.shape[2])


# def _parse_time(time: Union[List[str], List[datetime], List[int]]) -> dict:

#     return {'year': np.array([t.year if isinstance(t, datetime) else int(str(t)[:4]) for t in time]),
#             'day': np.array([t.timetuple().tm_yday if isinstance(t, datetime) else int(str(t)[4:]) for t in time])}

def _parse_time(time: Union[List[str], List[datetime], List[int]]) -> dict:
    """
    Parse a list of time inputs (datetime objects, strings, or integers) into year and day-of-year arrays.
    
    Args:
        time: List of datetime objects, strings (e.g., '2023001', '2023-01-01', '20230101'), or integers.
    
    Returns:
        dict: Dictionary with 'year' and 'day' as numpy arrays containing year and day-of-year.
    
    Raises:
        ValueError: If a time input cannot be parsed.
    """
    # Define possible date string formats
    date_formats = ['%Y%m%d', '%Y-%m-%d']
    
    def parse_single_time(t):
        if isinstance(t, datetime):
            return t.year, t.timetuple().tm_yday
        try:
            if isinstance(t, (int, float)):
                t = str(int(t))  # Convert int/float to string
            # Try parsing as YYYYDDD (year + day-of-year)
            if len(str(t)) >= 7 and t.isdigit():
                return int(t[:4]), int(t[4:])
            # Try parsing with defined date formats
            for fmt in date_formats:
                try:
                    dt = datetime.strptime(t, fmt)
                    return dt.year, dt.timetuple().tm_yday
                except ValueError:
                    continue
            raise ValueError(f"Cannot parse time input: {t}")
        except Exception as e:
            raise ValueError(f"Invalid time format for input {t}: {str(e)}")
    
    # Process each time input
    years, days = zip(*[parse_single_time(t) for t in time])
    
    return {
        'year': np.array(years),
        'day': np.array(days)
    }

def _generate_time_info(unique_years: np.ndarray, step_size: int, steps_per_year: int) -> np.ndarray:
    """生成时间信息数组"""
    time_info = []
    for year in unique_years:
        for step in range(steps_per_year):
            day = step * step_size + 1
            time_info.append(day * 10000 + year)  # DDDYYYY format
    return np.array(time_info)

def scale_batch(batch, mean=None, std=None, scaler=None):
    reshaped = batch.reshape(-1, batch.shape[-1])
    if scaler:
        return scaler.transform(reshaped).reshape(batch.shape)
    return ((reshaped - mean) / std).reshape(batch.shape)

def reshape_data(df_data):
    """
    自动识别输入数据的维度并转换为 [total_pixels, time_steps, bands]
    :param df_data: 输入数据，可以是 3D [time_steps, bands, pixels] 或 4D [time_steps, bands, width, height]
    :return: 重塑后的数据，形状为 [total_pixels, time_steps, bands]
    """
    input_dims = len(df_data.shape)
    if input_dims not in [3, 4]:
        raise ValueError("输入数据必须是3维或4维数组")

    time_steps = df_data.shape[0]
    bands = df_data.shape[1]

    if input_dims == 4:
        width, height = df_data.shape[2], df_data.shape[3]
        total_pixels = width * height
        # 先重塑再移动轴
        df_data = df_data.reshape(time_steps, bands, total_pixels)
        df_data = np.moveaxis(df_data, [0, 1, 2], [1, 2, 0])
    else:
        df_data = np.moveaxis(df_data, [0, 1, 2], [1, 2, 0])

    if not df_data.flags['C_CONTIGUOUS']:
        df_data = np.ascontiguousarray(df_data)

    return df_data

def safe_nanmedian(arr):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(arr, axis=0).astype(np.float16)

def composite_series(
    series: np.ndarray,
    time: Union[List[str], List[datetime], List[int]],
    mode: str = "3day",
    verbose: bool = False,
    n_jobs: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    # 输入验证
    if not isinstance(series, np.ndarray):
        raise TypeError("series 必须是 numpy 数组")
    if len(series) != len(time):
        raise ValueError(f"series长度({len(series)})和time长度({len(time)})不匹配")
    if mode not in ["3day", "8day"]:
        raise ValueError("mode 必须是 '3day' 或 '8day'")
    if len(series.shape) != 3:
        raise ValueError("输入数据必须是3维数组 [timesteps, bands, pixels]")
    if verbose:
        print(f"输入数据形状: {series.shape}")

    # 解析时间数据
    time_data = _parse_time(time)
    years = time_data["year"]
    days = time_data["day"]
    unique_years = np.unique(years)

    # 设置时间步长参数
    step_size = 3 if mode == "3day" else 8
    steps_per_year = 122 if mode == "3day" else 46
    total_steps = len(unique_years) * steps_per_year

    # 计算位置索引（向量化操作）
    year_indices = np.searchsorted(unique_years, years)
    step_indices = (days - 1) // step_size
    positions = year_indices * steps_per_year + step_indices

    # 获取唯一位置、首次出现索引和计数
    unique_positions, unique_indices, counts = np.unique(
        positions, return_index=True, return_counts=True
    )

    # 处理单值位置
    single_mask = counts == 1
    single_data = series[unique_indices[single_mask]]

    # 处理多值位置（优化分组逻辑）
    multi_mask = counts > 1
    if np.any(multi_mask):
        # 通过排序和分组一次性获取所有多值索引
        sorted_indices = np.argsort(positions)
        split_idx = np.cumsum(counts)[:-1]
        groups = np.split(sorted_indices, split_idx)
        multi_groups = [groups[i] for i in np.where(multi_mask)[0]]

        # 并行计算中位数
        multi_data = Parallel(n_jobs=n_jobs)(
            delayed(safe_nanmedian)(series[idx]) for idx in multi_groups
        )
        multi_data = np.stack(multi_data)
    else:
        multi_data = np.empty((0, *series.shape[1:]), dtype=np.float16)

    # 创建并填充结果数组
    result = np.full((total_steps, *series.shape[1:]), np.nan, dtype=np.float16)
    result[unique_positions[single_mask]] = single_data
    result[unique_positions[multi_mask]] = multi_data

    # 生成时间信息
    years_repeated = np.repeat(unique_years, steps_per_year)
    steps = np.tile(np.arange(steps_per_year) * step_size + 1, len(unique_years))
    time_info = np.char.add(years_repeated.astype(str), np.char.zfill(steps.astype(str), 3))

    if verbose:
        print(f"处理完成，输出形状: {result.shape}")
        print(f"时间范围: {time_info[0]} - {time_info[-1]}")

    return result, time_info


# 分块处理函数
def process_in_blocks(series, time, block_size=50, use_parallel=True):
    """
    按块处理数据，支持3D和4D输入，内部统一转为3D处理。
    此函数可选择使用并行处理来加速计算。

    :param series: 原始数据，形状为 [timesteps, bands, width, height] 或 [timesteps, bands, pixels]
    :param time: 时间信息，长度为 timesteps
    :param block_size: 每块的大小（默认50）
    :param use_parallel: 是否使用并行处理（默认True）。需要安装joblib。
    :return: 合成后的数据和时间信息
    """

    # 检查输入维度并获取形状
    input_dims = len(series.shape)
    if input_dims not in [3, 4]:
        raise ValueError("输入数据必须是3维或4维数组")

    # 保存原始形状，用于最后恢复
    timesteps = series.shape[0]
    bands = series.shape[1]
    if input_dims == 4:
        _, _, width, height = series.shape
        pixels = width * height
        series = series.reshape(timesteps, bands, pixels)
    else:  # 3D
        pixels = series.shape[2]

    # 计算分块数量（仅在像素维度上分块）
    num_blocks = int(np.ceil(pixels / block_size))

    # 计算合成后时间步数并初始化结果数组
    temp_result, time_info = composite_series(series[:, :, :1], time)  # 取第一个像素点测试
    total_steps = temp_result.shape[0]
    result = np.full((total_steps, bands, pixels), np.nan, dtype=series.dtype)

    # 定义块范围
    block_ranges = [(i * block_size, min((i + 1) * block_size, pixels)) for i in range(num_blocks)]

    # 定义处理每个块的函数
    def process_block(start_idx, end_idx):
        block_series = series[:, :, start_idx:end_idx]
        block_result, _ = composite_series(block_series, time)
        return block_result

    # 处理块，使用并行处理如果可用且请求
    if num_blocks > 1 and use_parallel and parallel_available:
        n_jobs = min(40, num_blocks)
        results = Parallel(n_jobs=n_jobs, max_nbytes='500M')(delayed(process_block)(start_idx, end_idx) for start_idx, end_idx in block_ranges)
    else:
        results = [process_block(start_idx, end_idx) for start_idx, end_idx in block_ranges]

    # 将结果放回结果数组
    for (start_idx, end_idx), block_result in zip(block_ranges, results):
        result[:, :, start_idx:end_idx] = block_result

    # 如果输入是4D，恢复为4D输出
    if input_dims == 4:
        result = result.reshape(total_steps, bands, width, height)

    return result, time_info

def load_and_process_satellite_data(data_directory):
    """
    加载并处理卫星图像数据

    Returns:
        dict: {
            'data': np.ndarray,  # 所有站点的数据
            'time': list,        # 时间列表
            'sitename': list     # 站点名称列表
        }
    """
    # 列出所有数据文件
    npz_files = [f for f in os.listdir(data_directory) if f.endswith('.npz')]
    npy_files = [f for f in os.listdir(data_directory) if f.endswith('.npy')]

    # 读取时间信息
    time_info = {}
    for npy_file in npy_files:
        dataset = npy_file.split('_')[0]
        file_path = os.path.join(data_directory, npy_file)
        time_data = np.load(file_path)
        time_info[dataset] = [t.split('T')[0] for t in time_data]

    # 处理站点数据
    temp_site_data = {}
    for npz_file in npz_files:
        split_result = os.path.splitext(npz_file)[0].split('_')
        site_name, dataset = split_result[:2]
        file_path = os.path.join(data_directory, npz_file)

        try:
            with np.load(file_path) as data:
                # 数据验证
                if not all(key in data.files for key in ['image_data', 'fmask_data']):
                    print(f"Skipping {npz_file}: missing required data.")
                    continue

                # 处理掩膜
                fmask_processor = FmaskProcessor(data['fmask_data'], dimension=1)
                mask = fmask_processor.create_mask()
                processed_image = fmask_processor.apply_mask(data['image_data'], mask)

                # 初始化站点数据结构
                if site_name not in temp_site_data:
                    temp_site_data[site_name] = {}

                # 关联时间数据
                timestamps = time_info.get(dataset, [])
                if len(timestamps) != processed_image.shape[0]:
                    print(f"Warning: Time data mismatch for {npz_file}.")
                    continue

                temp_site_data[site_name][dataset] = list(zip(timestamps, processed_image))

        except Exception as e:
            print(f"Error processing {npz_file}: {str(e)}")
            continue

    # 删除空站点并重构数据
    sites_data = []
    sites_times = []
    valid_sites = []

    for site_name, site_datasets in temp_site_data.items():
        # 跳过空站点
        if not site_datasets:
            print(f"Skipping empty site: {site_name}")
            continue

        # 合并该站点的所有数据集
        site_data_pairs = []
        for dataset in site_datasets.values():
            site_data_pairs.extend(dataset)

        # 确保站点有数据
        if not site_data_pairs:
            print(f"Skipping site with no valid data: {site_name}")
            continue

        # 按时间排序
        site_data_pairs.sort(key=lambda x: x[0])
        times, data = zip(*site_data_pairs)

        sites_data.append(np.array(data))
        sites_times.append(list(times))
        valid_sites.append(site_name)

    # 确保至少有一个有效站点
    if not valid_sites:
        raise ValueError("No valid data found for any site")

    # 构建最终结果时调整维度顺序
    result = {
        'data': np.stack(sites_data, axis=1).transpose(1, 0, 2),  # 从(steps, num, bands)转换为(num, steps, bands)
        'time': sites_times[0],
        'sitename': valid_sites
    }

    return result


def add_band_diff_ratio(df_data):
    # 检查输入数据的维度
    is_4d = len(df_data.shape) == 4

    if is_4d:
        steps, bands, width, height = df_data.shape
    else:
        steps, bands, pixels = df_data.shape

    assert bands >= 3, "df_data必须至少包含3个波段"

    # 如果是4D数据，将其重塑为3D进行处理
    if is_4d:
        df_data = df_data.reshape(steps, bands, -1)

    # 提取波段 2 和波段 3
    band2 = df_data[:, 1, :]
    band3 = df_data[:, 2, :]

    # 计算 (波段 3 - 波段 2) / (波段 3 + 波段 2)
    numerator = band3 - band2
    denominator = band3 + band2

    # 创建一个掩码，用来处理NaN值和除以零的情况
    valid_mask = (denominator != 0) & ~np.isnan(numerator) & ~np.isnan(denominator)

    # 初始化新波段并赋默认NaN值
    new_band = np.full_like(numerator, np.nan)

    # 计算新的波段值
    new_band[valid_mask] = numerator[valid_mask] / denominator[valid_mask]

    # 直接检查并修正超出范围的值
    new_band[(new_band > 1) | (new_band < -1)] = np.nan

    # 将新波段添加到 df_data 中
    df_data = np.concatenate((df_data, new_band[:, np.newaxis, :]), axis=1)

    # 检查新波段是否为 NaN，如果是，则将其他波段也设置为 NaN
    new_band_nan_mask = np.isnan(new_band)  # 新波段为 NaN 的掩码
    if np.any(new_band_nan_mask):
        # 扩展掩码到所有波段
        new_band_nan_mask_expanded = np.expand_dims(new_band_nan_mask, axis=1)  # [steps, 1, pixels]
        new_band_nan_mask_expanded = np.repeat(new_band_nan_mask_expanded, bands + 1,
                                               axis=1)  # [steps, bands+1, pixels]

        # 将新波段为 NaN 的位置的所有波段设置为 NaN
        df_data[new_band_nan_mask_expanded] = np.nan

    # 如果原始输入是4D，将结果重塑回4D
    if is_4d:
        df_data = df_data.reshape(steps, bands + 1, width, height)

    return df_data

def load_file(file_path, year):
    print(f"Processing {file_path}...")
    data_npz = np.load(file_path)
    data = data_npz['data']  # [steps, bands, width, height]
    time = data_npz['time'].astype(str)  # [steps]
    steps, bands, width, height = data.shape
    data_reshaped = data.reshape(steps, bands, -1)  # [steps, bands, width*height]
    return data_reshaped, time

