from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5

class DayOfYearSin(TimeFeature):
    """Sine encoding of day of year to capture periodicity"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        dayofyear = index.dayofyear - 1  # 0-based (0 to 364 or 365)
        days_in_year = index.is_leap_year.astype(int) + 365  # 365 or 366
        angle = 2 * np.pi * dayofyear / days_in_year
        return np.sin(angle)  # 形状 (len(index),)

class DayOfYearCos(TimeFeature):
    """Cosine encoding of day of year to capture periodicity"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        dayofyear = index.dayofyear - 1
        days_in_year = index.is_leap_year.astype(int) + 365
        angle = 2 * np.pi * dayofyear / days_in_year
        return np.cos(angle)  # 形状 (len(index),)

class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


class Year(TimeFeature):
    """Year encoded as a normalized absolute count (linear trend)."""

    def __init__(self, start_year: int = 1985, end_year: int = 2025):
        self.start_year = start_year 
        self.year_span = end_year - start_year 

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        years = index.year.astype(float)
        normalized_years = (years - self.start_year) / self.year_span
        return normalized_years - 0.5 

class Phenology(TimeFeature):
    """植物生长季节编码"""
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # 示例：将一年分为生长季(4-10月)和休眠季(11-3月)
        month = index.month
        growing_season = ((month >= 4) & (month <= 10))
        return growing_season.astype(float) - 0.5

def time_features_from_frequency_str(freq_str: str, start_year: int = 2013, end_year: int = 2025) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [

            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        # 'rs': [ DayOfYearSin, DayOfYearCos, Year],
        'rs': [DayOfYearSin, DayOfYearCos],
    }

    # 首先检查是否是自定义模式
    if freq_str.upper() == 'RS':  # 遥感数据模式
        # return [DayOfYearSin(), DayOfYearCos(), Year(start_year=start_year, end_year=end_year)]
        return [DayOfYearSin(), DayOfYearCos()]


    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]



    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
        RS  - remote sensing
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h', start_year=2013, end_year=2025):
    features = time_features_from_frequency_str(freq, start_year=start_year, end_year=end_year)
    return np.vstack([feat(dates) for feat in features])
