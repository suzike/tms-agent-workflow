"""专业热舒适计算工具(确定性、纯函数、零幻觉)。

实现 ISO 7730 / Fanger PMV-PPD 模型,并派生:
- 车内当量温度 EQT(综合空气温度/辐射/气流的体感,汽车热舒适常用)
- 目标舒适温度 target_comfort_temp(反解使 PMV≈0,作 LLM 推荐锚点)

座舱适配:平均辐射温度 MRT 由日照强度估算,风速由风量档位映射。
这些函数在 graph 的 comfort 节点被确定性直接调用;同时以 LangChain Tool 暴露
(``THERMAL_TOOLS``),保留未来 agent tool-calling 扩展。
"""
from __future__ import annotations

import math

from langchain_core.tools import StructuredTool

from ..config import (
    COMFORT_DEFAULTS, FAN_MAX, FAN_MIN, FAN_VELOCITY_TABLE,
    TEMP_MAX, TEMP_MIN, THRESHOLDS,
)

# 座舱物理映射常数
_SUN_MRT_GAIN = 15.0   # 满日照对 MRT 的最大增量(℃)
_SUN_REF_WM2 = 1500.0  # 太阳辐照参考上限(W/m²),用于归一化
_VEL_STILL = 0.05      # 静风基线 (m/s),用于 EQT 气流冷却项


def approach_step(cursor: float, pref: float, grid: float,
                  rate: float = None, lo: float = None, hi: float = None) -> float:
    """逐步逼近一步:从 cursor 沿 pref 方向迈出 rate×差值,贴齐到网格。

    差值越大首步越大;距偏好一格以内直接贴齐。保证每步至少移动一格、单调收敛、不超调。
    例:风量 cursor=7,pref=3,grid=1 → 5;再 5→4;再 4→3。
    """
    rate = THRESHOLDS.APPROACH_RATE if rate is None else rate
    gap = pref - cursor
    if abs(gap) <= grid:
        nxt = pref
    else:
        raw = cursor + rate * gap
        nxt = math.floor(raw / grid) * grid if pref < cursor else math.ceil(raw / grid) * grid
        if abs(nxt - cursor) < grid:  # 防停滞:至少迈一格
            nxt = cursor + (grid if pref > cursor else -grid)
        if (pref < cursor and nxt < pref) or (pref > cursor and nxt > pref):
            nxt = pref  # 防超调
    if lo is not None:
        nxt = max(lo, nxt)
    if hi is not None:
        nxt = min(hi, nxt)
    return round(nxt, 1)


def met_from_activity(activity: str) -> float:
    """活动("热状态")→ 代谢率 met:睡眠最低、兴奋最高。"""
    d = COMFORT_DEFAULTS
    return {
        "sleeping": d.met_sleeping,
        "calm": d.met_calm,
        "excited": d.met_excited,
    }.get(activity, d.met)


def comfort_offset(category: str, gender: str, bmi: float) -> float:
    """人体特征 → 目标舒适温度启发式偏移(℃,正=偏暖)。"""
    d = COMFORT_DEFAULTS
    off = 0.0
    if category == "child":
        off += d.child_offset
    elif category == "elderly":
        off += d.elderly_offset
    if gender == "female":
        off += d.female_offset
    if bmi >= d.bmi_high:
        off += d.bmi_high_offset
    elif bmi < d.bmi_low:
        off += d.bmi_low_offset
    return round(off, 2)


def air_velocity_from_fan(fan_level: int) -> float:
    """风量档位(1~7 档)→ 乘员处等效风速 (m/s),查表映射(用于 PMV)。"""
    f = int(max(FAN_MIN, min(fan_level, FAN_MAX)))
    return FAN_VELOCITY_TABLE[f]


def mrt_from_sun(air_temp: float, sun_wm2: float) -> float:
    """由空气温度与太阳辐照(W/m²,0~1500)估算平均辐射温度 MRT。"""
    frac = max(0.0, min(sun_wm2 / _SUN_REF_WM2, 1.0))
    return air_temp + frac * _SUN_MRT_GAIN


def clo_for_season(season: str) -> float:
    d = COMFORT_DEFAULTS
    return {
        "summer": d.clo_summer,
        "winter": d.clo_winter,
        "transition": d.clo_transition,
    }.get(season, d.clo_transition)


def clo_from_clothing(clothing: str) -> float:
    """衣着 → 服装热阻 clo(实测量,优先于季节兜底)。"""
    d = COMFORT_DEFAULTS
    return {
        "light": d.clo_light,
        "medium": d.clo_medium,
        "heavy": d.clo_heavy,
    }.get(clothing, d.clo_medium)


def estimate_cabin_temp(
    ambient_temp: float,
    sun_wm2: float,
    window_open_pct: float = 0.0,
    speed: float = 0.0,
) -> float:
    """车厢热模型:由车外温度+太阳辐照估算车内空气温度(无车内温感时使用)。

    稳态近似:车内 = 车外 + 日照升温×(1-通风衰减)。
    日照越强升温越高;开窗或行驶带来通风,把车内拉回接近车外温度。
    这是工程估算(非实测),其不确定性会在 DecisionTrace.assumptions 中标注。
    """
    soak = (max(0.0, sun_wm2) / 1000.0) * COMFORT_DEFAULTS.soak_gain_at_1kw
    vent = min(1.0, (window_open_pct / 100.0) * 0.7 + (0.2 if speed > 0 else 0.0))
    return round(ambient_temp + soak * (1.0 - vent), 1)


def seat_air_temp(cabin, seat_id: str) -> tuple[float, bool]:
    """该座位空气温度 ta:有内温实测则用之;否则由热模型回退估算。

    返回 (温度, 是否实测)。cabin 为 CabinContext(鸭子类型,避免循环导入)。
    """
    if getattr(cabin, "cabin_temp", None) is not None:
        return float(cabin.cabin_temp), True
    est = estimate_cabin_temp(
        cabin.ambient_temp, cabin.seat_sun(seat_id),
        cabin.window_open_pct(seat_id), cabin.speed,
    )
    return est, False


def compute_pmv_ppd(
    air_temp: float,
    mean_radiant_temp: float,
    air_velocity: float,
    humidity: float,
    met: float = COMFORT_DEFAULTS.met,
    clo: float = 0.5,
) -> tuple[float, float]:
    """ISO 7730 PMV/PPD。返回 (PMV ∈ ~[-3,3], PPD %∈[5,100])。

    参数:air_temp/mrt(℃)、air_velocity(m/s)、humidity(%RH)、met、clo。
    """
    pa = humidity * 10.0 * math.exp(16.6536 - 4030.183 / (air_temp + 235.0))
    icl = 0.155 * clo
    m = met * 58.15
    mw = m  # 无外部机械功
    fcl = 1.0 + 1.29 * icl if icl <= 0.078 else 1.05 + 0.645 * icl
    hcf = 12.1 * math.sqrt(max(air_velocity, 0.0))

    taa = air_temp + 273.0
    tra = mean_radiant_temp + 273.0
    tcla = taa + (35.5 - air_temp) / (3.5 * icl + 0.1)

    p1 = icl * fcl
    p2 = p1 * 3.96
    p3 = p1 * 100.0
    p4 = p1 * taa
    p5 = 308.7 - 0.028 * mw + p2 * (tra / 100.0) ** 4

    xn = tcla / 100.0
    xf = xn
    hc = hcf
    for _ in range(150):
        xf = (xf + xn) / 2.0
        hcn = 2.38 * abs(100.0 * xf - taa) ** 0.25
        hc = max(hcf, hcn)
        xn = (p5 + p4 * hc - p2 * xf**4) / (100.0 + p3 * hc)
        if abs(xn - xf) <= 0.00015:
            break
    tcl = 100.0 * xn - 273.0

    hl1 = 3.05 * 0.001 * (5733.0 - 6.99 * mw - pa)
    hl2 = 0.42 * (mw - 58.15) if mw > 58.15 else 0.0
    hl3 = 1.7 * 0.00001 * m * (5867.0 - pa)
    hl4 = 0.0014 * m * (34.0 - air_temp)
    hl5 = 3.96 * fcl * (xn**4 - (tra / 100.0) ** 4)
    hl6 = fcl * hc * (tcl - air_temp)

    ts = 0.303 * math.exp(-0.036 * m) + 0.028
    pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
    ppd = 100.0 - 95.0 * math.exp(-0.03353 * pmv**4 - 0.2179 * pmv**2)
    return round(pmv, 2), round(ppd, 1)


def equivalent_temperature(
    air_temp: float, mean_radiant_temp: float, air_velocity: float
) -> float:
    """车内当量温度 EQT 近似:辐射与空气温度均权,叠加气流冷却。"""
    base = 0.5 * (air_temp + mean_radiant_temp)
    cooling = 4.0 * math.sqrt(max(air_velocity - _VEL_STILL, 0.0))
    return round(base - cooling, 1)


def target_comfort_temp(
    sun_wm2: float,
    humidity: float,
    season: str,
    air_velocity: float = 0.15,
    met: float = COMFORT_DEFAULTS.met,
    clo: float | None = None,
    lo: float = 15.5,
    hi: float = 31.5,
) -> float:
    """二分搜索使 PMV≈0 的空气温度(MRT 随之由太阳辐照重算),作推荐锚点。

    clo 为 None 时按季节兜底;有衣着输入时应显式传入。
    """
    if clo is None:
        clo = clo_for_season(season)

    def pmv_at(ta: float) -> float:
        mrt = mrt_from_sun(ta, sun_wm2)
        return compute_pmv_ppd(ta, mrt, air_velocity, humidity, met, clo)[0]

    f_lo, f_hi = pmv_at(lo), pmv_at(hi)
    if f_lo > 0:  # 全程偏热,最低温即最优
        return lo
    if f_hi < 0:  # 全程偏冷
        return hi
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if pmv_at(mid) < 0:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2.0, 1)


def transient_setpoint_fan(cabin_temp: float, target_temp: float) -> tuple[float, int]:
    """瞬态控制:由"车内温度 vs 目标"给出设定温度与风量。

    - 车内远高于目标(暴晒/大负荷)→ 设定更低 + 大风量,快速降温;
    - 车内逐步接近目标 → 设定温度回调向目标、风量降低;
    - 达到稳态 → 设定=最舒适目标、风量降至最低档(兼顾 NVH)。
    制热方向对称(车内远低于目标 → 设定更高 + 大风量)。
    返回 (设定温度℃[0.5 步进], 风量档[1-7])。
    """
    d = COMFORT_DEFAULTS
    delta = cabin_temp - target_temp  # >0 需制冷,<0 需制热
    overshoot = max(-d.cooldown_overshoot_max,
                    min(d.cooldown_overshoot_rate * delta, d.cooldown_overshoot_max))
    setpoint = target_temp - overshoot  # 制冷时偏低、制热时偏高,随 delta→0 回归目标
    setpoint = round(max(TEMP_MIN, min(setpoint, TEMP_MAX)) / 0.5) * 0.5
    fan = int(max(d.fan_steady_min,
                  min(round(abs(delta) / d.fan_per_degree) + d.fan_steady_min, FAN_MAX)))
    return setpoint, fan


def dew_point(air_temp: float, humidity: float) -> float:
    """露点温度(Magnus 公式,℃)。低于物体表面温度即不结露。"""
    rh = max(1.0, min(humidity, 100.0))
    a, b = 17.62, 243.12
    gamma = math.log(rh / 100.0) + a * air_temp / (b + air_temp)
    return round(b * gamma / (a - gamma), 1)


def fog_risk(cabin_temp: float, ambient_temp: float, humidity: float) -> str:
    """起雾风险等级 low/medium/high:车内空气露点接近(冷)风挡表面温度即高风险。"""
    margin = ambient_temp - dew_point(cabin_temp, humidity)  # 表面温度≈车外温
    if margin > 4:
        return "low"
    if margin > 1:
        return "medium"
    return "high"


def comfort_temp_band(
    sun_wm2: float, humidity: float, season: str,
    air_velocity: float = 0.15, met: float = COMFORT_DEFAULTS.met,
    clo: float | None = None,
) -> tuple[float, float]:
    """可接受舒适温度区间(PMV∈[-0.5,0.5] 对应 PPD<10%)。返回 (下限, 上限)℃。"""
    if clo is None:
        clo = clo_for_season(season)
    band = []
    t = 15.5
    while t <= 31.5 + 1e-9:
        pmv = compute_pmv_ppd(t, mrt_from_sun(t, sun_wm2), air_velocity,
                              humidity, met, clo)[0]
        if -0.5 <= pmv <= 0.5:
            band.append(round(t, 1))
        t += 0.5
    if band:
        return band[0], band[-1]
    tgt = target_comfort_temp(sun_wm2, humidity, season, air_velocity, met, clo)
    return tgt, tgt


def recirculation_hint(sun_wm2: float, ambient_temp: float, soc: float,
                       weather: str) -> str:
    """内外循环建议:inner(内循环)/ outer(外循环)/ auto。"""
    if weather in ("rain", "snow"):
        return "outer"   # 除雾宜引入新风/外循环
    if sun_wm2 >= 750 or soc < 30 or abs(ambient_temp - 24) > 12:
        return "inner"   # 快速调节/节能宜内循环
    return "auto"


def energy_hint(soc: float, temp_set: float, ambient_temp: float) -> str:
    """节能建议:低电量且设定与车外温差大时提示收敛。"""
    if soc < 30 and abs(temp_set - ambient_temp) > 8:
        return "低电量:建议设定温度向车外温度收敛、降低风量、多用内循环以省电增程"
    return "电量充足:可优先舒适"


# ---- LangChain Tool 封装(扩展用;核心路径仍直接调用上面纯函数)----
THERMAL_TOOLS = [
    StructuredTool.from_function(
        func=compute_pmv_ppd,
        name="compute_pmv_ppd",
        description="按 ISO 7730 计算热舒适指标 PMV 与 PPD。",
    ),
    StructuredTool.from_function(
        func=equivalent_temperature,
        name="equivalent_temperature",
        description="计算车内当量温度 EQT(体感温度)。",
    ),
    StructuredTool.from_function(
        func=target_comfort_temp,
        name="target_comfort_temp",
        description="反解使 PMV≈0 的目标舒适温度,作为空调温度推荐锚点。",
    ),
    StructuredTool.from_function(
        func=comfort_temp_band,
        name="comfort_temp_band",
        description="给出可接受舒适温度区间(PMV∈[-0.5,0.5])。",
    ),
    StructuredTool.from_function(
        func=transient_setpoint_fan,
        name="transient_setpoint_fan",
        description="瞬态控制:按车内温度与目标差给出设定温度与风量(快速降温→稳态低风量)。",
    ),
    StructuredTool.from_function(
        func=fog_risk, name="fog_risk",
        description="评估起雾风险等级(low/medium/high)。",
    ),
    StructuredTool.from_function(
        func=dew_point, name="dew_point", description="计算露点温度(℃)。",
    ),
    StructuredTool.from_function(
        func=recirculation_hint, name="recirculation_hint",
        description="建议内外循环(inner/outer/auto)。",
    ),
    StructuredTool.from_function(
        func=energy_hint, name="energy_hint", description="低电量节能建议。",
    ),
]
