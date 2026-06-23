"""智能除雾 Agent 专属确定性工具(纯函数、零幻觉)。

判定输入(仅与起雾相关,与舒适解耦):
- 玻璃表面温度 glass_temp
- 玻璃附近空气温度 air_temp
- 玻璃附近空气相对湿度 humidity
- 雨量信号 rain_level(无/小雨/中雨/大雨)+ 天气兜底

物理核心:玻璃温度低于(或接近)玻璃附近空气露点时,水汽在玻璃内表面凝结起雾。
故"玻璃温度 − 露点 = 裕度",裕度越小越危险;雨量/高湿提升基线风险(前馈)。
共用露点物理函数 ``tools.thermal_comfort.dew_point`` 保持 DRY。
"""
from __future__ import annotations

from langchain_core.tools import StructuredTool

from ..config import DEFOG
from ..tools.thermal_comfort import dew_point

# 紧急度数值编码:0=无 1=轻度 2=强
_LEVEL_NAME = {0: "none", 1: "mild", 2: "strong"}
_RAIN_PRESSURE = {"none": 0, "light": 1, "moderate": 2, "heavy": 3}


def rain_fog_pressure(rain_level: str, weather: str = "sunny") -> int:
    """雨量信号 → 前馈起雾压力(0~3)。天气作兜底:rain/snow 至少中雨级。"""
    lvl = _RAIN_PRESSURE.get(rain_level, 0)
    if weather == "rain":
        lvl = max(lvl, 2)
    elif weather == "snow":
        lvl = max(lvl, 2)
    return lvl


def dew_point_margin(glass_temp: float, air_temp: float, humidity: float) -> float:
    """除雾反馈核心:玻璃表面温度 − 玻璃附近空气露点(℃)。≤0 即已结露。"""
    return round(glass_temp - dew_point(air_temp, humidity), 1)


def _feedback_severity(glass_temp, air_temp, humidity) -> tuple[int, str]:
    """反馈严重度(玻璃温度 vs 露点裕度)。玻璃温度缺失时返回 (0, '')。"""
    if glass_temp is None:
        return 0, ""
    air = air_temp if air_temp is not None else glass_temp
    margin = dew_point_margin(glass_temp, air, humidity)
    dp = dew_point(air, humidity)
    if margin <= DEFOG.margin_strong:
        sev = 2
    elif margin <= DEFOG.margin_mild:
        sev = 1
    else:
        sev = 0
    return sev, f"玻璃{glass_temp:.0f}℃ vs 露点{dp:.0f}℃,裕度{margin:.1f}℃"


def _feedforward_severity(humidity, rain_level, weather) -> tuple[int, str]:
    """前馈严重度(雨量/高湿等基线风险,玻璃温度未接入时的主依据)。"""
    rain = rain_fog_pressure(rain_level, weather)
    if rain >= 3:
        sev = 2
    elif rain == 2 or humidity >= DEFOG.humidity_high:
        sev = 1
    elif rain == 1 and humidity >= DEFOG.humidity_mid:
        sev = 1
    else:
        sev = 0
    # 中/大雨 + 高湿叠加 → 升至强
    if rain >= 2 and humidity >= DEFOG.humidity_high:
        sev = 2
    rain_zh = {0: "无雨", 1: "小雨", 2: "中雨", 3: "大雨"}[rain]
    return sev, f"{rain_zh}/湿度{humidity:.0f}%"


def defog_urgency(
    glass_temp: float | None,
    air_temp: float | None,
    humidity: float,
    rain_level: str = "none",
    weather: str = "sunny",
) -> dict:
    """综合前馈+反馈给出除雾紧急度。返回诊断 dict(含 level/need_defog/明细)。

    取前馈与反馈的更严重者;两者均非零(雨势+裕度小)再升一级,体现叠加风险。
    """
    fb, fb_detail = _feedback_severity(glass_temp, air_temp, humidity)
    ff, ff_detail = _feedforward_severity(humidity, rain_level, weather)
    overall = max(fb, ff)
    if fb >= 1 and ff >= 1:
        overall = min(2, overall + 1)
    level = _LEVEL_NAME[overall]
    parts = [p for p in (f"反馈[{fb_detail}]" if fb_detail else "",
                         f"前馈[{ff_detail}]") if p]
    return {
        "level": level,
        "need_defog": overall >= 1,
        "feedback_severity": fb,
        "feedforward_severity": ff,
        "dew_point": dew_point(
            air_temp if air_temp is not None else (glass_temp or 0.0), humidity),
        "margin": (dew_point_margin(
            glass_temp, air_temp if air_temp is not None else glass_temp, humidity)
            if glass_temp is not None else None),
        "rain_level": rain_level,
        "detail": ";".join(parts),
    }


# ---- LangChain Tool 封装(扩展用;核心路径直接调用上面纯函数)----
DEFOG_TOOLS = [
    StructuredTool.from_function(
        func=dew_point_margin, name="dew_point_margin",
        description="计算玻璃表面温度与玻璃附近空气露点的裕度(℃),≤0 即已结露。",
    ),
    StructuredTool.from_function(
        func=rain_fog_pressure, name="rain_fog_pressure",
        description="雨量信号(无/小雨/中雨/大雨)→ 前馈起雾压力等级 0~3。",
    ),
    StructuredTool.from_function(
        func=defog_urgency, name="defog_urgency",
        description="综合玻璃温度/空气温度/湿度/雨量,给出除雾紧急度(none/mild/strong)。",
    ),
]
