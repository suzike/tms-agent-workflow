"""特征工程:把(全局环境 + 单座位乘员)映射为归一化加权特征向量。

设计:
- 连续量先分桶(降噪),再归一化并乘以维度权重,使"加权欧氏距离"=普通欧氏距离。
- 季节用 one-hot 且高权重,实现"季节隔离"(不同季节天然距离大,互不命中)。
- 纯结构化数值,不依赖 embedding(零成本、可解释)。
"""
from __future__ import annotations

import time

import numpy as np

from .config import SUN_WM2_MAX, WEIGHTS
from .schemas import CabinContext, OccupantState
from .tools.thermal_comfort import seat_air_temp

_WEATHER_ORDINAL = {"sunny": 0.0, "cloudy": 0.33, "rain": 0.66, "snow": 1.0}
_ACTIVITY_ORDINAL = {"sleeping": -1.0, "calm": 0.0, "excited": 1.0}
_CLOTHING_ORDINAL = {"light": -1.0, "medium": 0.0, "heavy": 1.0}
_CATEGORY_ORDINAL = {"child": -1.0, "adult": 0.0, "elderly": 1.0}
_SEASON_INDEX = {"summer": 0, "winter": 1, "transition": 2}


def _bucket(value: float, step: float) -> float:
    """按 step 分桶(四舍五入到最近的桶中心),降低连续值噪声。"""
    return round(value / step) * step


def _time_of_day_bucket(timestamp: float) -> float:
    """时段分 4 桶并归一化到 [0,1]:夜0 / 晨0.33 / 午0.66 / 晚1。"""
    hour = time.localtime(timestamp).tm_hour
    if 5 <= hour < 11:
        return 0.33  # 晨
    if 11 <= hour < 16:
        return 0.66  # 午
    if 16 <= hour < 22:
        return 1.0  # 晚
    return 0.0  # 夜


def featurize(cabin: CabinContext, occupant: OccupantState) -> np.ndarray:
    """生成该座位的加权特征向量。

    维度顺序固定:[ambient, cabin_temp_est, sun, humidity, activity, clothing,
    category, weather, time, speed, season_summer, season_winter, season_transition]
    """
    w = WEIGHTS
    season_onehot = [0.0, 0.0, 0.0]
    season_onehot[_SEASON_INDEX[cabin.season]] = w.season
    seat = occupant.seat_id
    sun = cabin.seat_sun(seat)
    cabin_temp, _ = seat_air_temp(cabin, seat)  # 内温实测优先,缺失则估算
    sun_norm = min(sun / SUN_WM2_MAX, 1.0)

    vec = [
        _bucket(cabin.ambient_temp, 3.0) / 10.0 * w.ambient_temp,
        _bucket(cabin_temp, 3.0) / 10.0 * w.local_cabin_temp,
        _bucket(sun_norm, 0.2) * w.local_sun,
        _bucket(cabin.humidity, 20.0) / 100.0 * w.humidity,
        _ACTIVITY_ORDINAL[occupant.activity] * w.activity,
        _CLOTHING_ORDINAL[occupant.clothing] * w.clothing,
        _CATEGORY_ORDINAL[occupant.category] * w.person_category,
        _WEATHER_ORDINAL[cabin.weather] * 0.3,
        _time_of_day_bucket(cabin.timestamp) * w.time_of_day,
        _bucket(cabin.speed, 20.0) / 100.0 * w.speed,
        *season_onehot,
    ]
    return np.asarray(vec, dtype=float)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """加权欧氏距离(权重已内嵌于特征值)。"""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
