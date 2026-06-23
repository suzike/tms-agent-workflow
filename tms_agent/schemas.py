"""数据模型(Pydantic v2):输入/输出/记忆/决策留痕。

设计要点:
- 输入做范围校验,缺值/NaN 由上层 sanitize 兜底(见 ``sanitize_scene``)。
- 输出对象不可变语义:节点产生新对象,不原地修改既有记录。
"""
from __future__ import annotations

import math
import time
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from . import config

SeatId = Literal["driver", "front_passenger", "rear_left", "rear_right"]
Weather = Literal["sunny", "cloudy", "rain", "snow"]
RainLevel = Literal["none", "light", "moderate", "heavy"]  # 雨量信号:无/小雨/中雨/大雨
DefogLevel = Literal["none", "mild", "strong"]             # 除雾紧急度:无/轻度/强
Season = Literal["summer", "winter", "transition"]
Gender = Literal["male", "female"]
Emotion = Literal["happy", "angry", "sad", "joy", "worried", "neutral"]  # 喜怒哀乐愁
Activity = Literal["excited", "calm", "sleeping"]  # "热状态":活动/代谢状态
Clothing = Literal["light", "medium", "heavy"]     # 衣着:薄/常规/厚
PersonCategory = Literal["child", "adult", "elderly"]  # 小孩/大人/老人
# 舒适基础模式(由舒适推理在三者间选择)
BaseAirMode = Literal["face", "face_feet", "feet"]
# 完整 7 种出风模式 = 3 基础 + 除霜叠加 + 纯除霜(由安全层按起雾风险落定)
AirMode = Literal[
    "face", "face_feet", "feet",
    "face_defrost", "face_feet_defrost", "feet_defrost",
    "defrost",
]


class CabinContext(BaseModel):
    """全局车厢环境(与座位无关)。"""

    ambient_temp: float = Field(..., ge=-40, le=60)    # 车外温度 ℃
    cabin_temp: Optional[float] = Field(None, ge=-40, le=90)  # 车内温度(内温,实测);缺失时由热模型回退估算
    weather: Weather = "sunny"
    humidity: float = Field(50.0, ge=0, le=100)        # 玻璃附近空气相对湿度 %(除雾 Agent 输入)
    # 起雾反馈输入(前风挡区域):玻璃附近空气温度 + 玻璃表面温度;玻璃温度低于露点即起雾
    windshield_air_temp: Optional[float] = Field(None, ge=-40, le=90)
    windshield_glass_temp: Optional[float] = Field(None, ge=-40, le=90)
    # 雨量信号(独立"智能除雾 Agent"前馈输入):无/小雨/中雨/大雨
    rain_level: RainLevel = "none"
    # 太阳辐照仅主驾/副驾两路 W/m²(后排按同侧前排取值)
    sun_driver_wm2: float = Field(0.0, ge=0, le=1500)
    sun_passenger_wm2: float = Field(0.0, ge=0, le=1500)
    soc: float = Field(80.0, ge=0, le=100)
    speed: float = Field(0.0, ge=0, le=300)
    season: Season = "transition"
    # 四个车门开/闭(缺省视为关闭);键为座位
    doors_open: dict[SeatId, bool] = Field(default_factory=dict)
    # 四个车窗开启百分比 0-100(缺省 0=关闭)
    windows_open: dict[SeatId, float] = Field(default_factory=dict)
    # 显式"最大除霜功能"开关(对应车上 MAX 除霜按键):开启才进入纯除霜模式
    max_defrost: bool = False
    timestamp: float = Field(default_factory=lambda: time.time())

    def seat_sun(self, seat_id: str) -> float:
        """该座位太阳辐照:左侧(主驾/左后)用主驾路,右侧用副驾路。"""
        return (self.sun_driver_wm2 if seat_id in ("driver", "rear_left")
                else self.sun_passenger_wm2)

    def window_open_pct(self, seat_id: str) -> float:
        return float(self.windows_open.get(seat_id, 0.0))

    def any_opening(self) -> bool:
        """是否有门开或窗开(影响制冷效果/热负荷)。"""
        return any(self.doors_open.values()) or any(
            v > 0 for v in self.windows_open.values()
        )

    def max_window_open(self) -> float:
        return max(self.windows_open.values(), default=0.0)


class OccupantState(BaseModel):
    """每位乘员(独立预测的单位),含人体特征与状态。"""

    seat_id: SeatId
    user_id: str = "default"
    present: bool = True
    # 人体特征
    age: int = Field(35, ge=0, le=120)
    gender: Gender = "male"
    height_cm: float = Field(170.0, ge=30, le=230)
    weight_kg: float = Field(70.0, ge=3, le=250)
    # 状态:衣着 + 情绪(喜怒哀乐愁)+ 活动/代谢("热状态")
    clothing: Clothing = "medium"
    emotion: Emotion = "neutral"
    activity: Activity = "calm"

    @property
    def bmi(self) -> float:
        h = self.height_cm / 100.0
        return round(self.weight_kg / (h * h), 1) if h > 0 else 0.0

    @property
    def category(self) -> PersonCategory:
        """识别 小孩/大人/老人。"""
        if self.age <= config.COMFORT_DEFAULTS.child_max_age:
            return "child"
        if self.age >= config.COMFORT_DEFAULTS.elderly_min_age:
            return "elderly"
        return "adult"


class SceneInput(BaseModel):
    cabin: CabinContext
    occupants: list[OccupantState]

    @field_validator("occupants")
    @classmethod
    def _at_least_one_present(cls, v: list[OccupantState]) -> list[OccupantState]:
        if not any(o.present for o in v):
            raise ValueError("场景至少需要一位在座乘员")
        return v

    def present_occupants(self) -> list[OccupantState]:
        return [o for o in self.occupants if o.present]


class ZoneSetting(BaseModel):
    """单座位输出:三项设定 + 简短解释。"""

    seat_id: SeatId
    fan_level: int = Field(..., ge=config.FAN_MIN, le=config.FAN_MAX)
    temp_set: float = Field(..., ge=config.TEMP_MIN, le=config.TEMP_MAX)
    air_mode: AirMode
    reasoning: str = ""


class HVACDecision(BaseModel):
    """LLM 结构化输出目标(不含 seat_id,由节点补全)。

    air_mode 仅输出 3 个舒适基础模式;除霜叠加由安全层按起雾风险落定。
    """

    fan_level: int = Field(..., ge=config.FAN_MIN, le=config.FAN_MAX)
    temp_set: float = Field(..., ge=config.TEMP_MIN, le=config.TEMP_MAX)
    air_mode: BaseAirMode
    reasoning: str = Field("", description="一句话说明依据")


class DefogDecision(BaseModel):
    """独立"智能除雾 Agent"的输出:是否除雾 + 紧急度 + 依据 + 诊断。

    舒适 Agent 在确定出风模式时叠加本决策(基础模式 + 除霜 → face_defrost 等)。
    纯除霜仍仅由"最大除霜功能"按键触发,本 Agent 不输出纯除霜。
    """

    need_defog: bool = False
    level: DefogLevel = "none"             # none/mild/strong
    reasoning: str = ""
    source: Literal["llm", "fallback", "rule"] = "rule"  # 决策来源
    # 诊断量(可视化/留痕):露点、玻璃温度、裕度、雨量、前馈/反馈风险
    diagnostics: dict = Field(default_factory=dict)


class CorrectionRecord(BaseModel):
    """一条 (user×seat) 修正记忆 —— 存储完整记忆链条。

    链条 = 输入快照(人员状态 + 车辆状态 + 环境参数)→ 系统推理结果(recommended)
    → 用户最终修正(corrected)。scene_vector 为该输入的归一化特征,用于相似度召回;
    cabin/occupant 为完整可读快照,供审计、展示与未来更细粒度学习。
    """

    user_id: str
    seat_id: SeatId
    scene_vector: list[float]
    recommended: ZoneSetting          # 该输入下空调推理出的结果
    corrected: ZoneSetting            # 用户最终调整成的结果
    season: Season
    # 完整输入快照(环境+车辆在 cabin,人员在 occupant);旧记录可能缺省为 None
    cabin: Optional[CabinContext] = None
    occupant: Optional[OccupantState] = None
    timestamp: float = Field(default_factory=lambda: time.time())


class ComfortMetrics(BaseModel):
    pmv: float
    ppd: float
    eqt: float
    target_temp: float
    cabin_temp: float = 0.0  # 估算的车内空气温度(热模型,非实测)


class DecisionTrace(BaseModel):
    """每座位决策留痕,用于可观测与可解释。"""

    seat_id: SeatId
    user_id: str
    source: Literal["llm", "fallback"] = "llm"  # 专业基准来源
    memory_evidence: int = 0      # 一致历史修正条数(逼近证据)
    approach_weight: float = 0.0  # 本次推荐向用户偏好逼近的权重 w
    learned_preference: Optional[dict] = None  # 学到的偏好代表值
    comfort_metrics: Optional[ComfortMetrics] = None
    comfort_breakdown: dict = Field(default_factory=dict)  # 计算过程中间量(可视化用)
    assumptions: list[str] = Field(default_factory=list)  # PMV 输入中的估算/假设项,透明化
    knowledge_snippets: list[str] = Field(default_factory=list)
    memory_hit_count: int = 0
    llm_raw: Optional[dict] = None
    safety_adjustments: list[str] = Field(default_factory=list)
    fallback_fields: list[str] = Field(default_factory=list)
    final: Optional[ZoneSetting] = None


# ---------------------------------------------------------------------------
# 输入兜底:把可能含 NaN / 缺失的原始 dict 清洗成合法 SceneInput。
# ---------------------------------------------------------------------------
def _clean_number(value: Any, default: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def sanitize_scene(raw: dict) -> tuple[SceneInput, list[str]]:
    """把原始(可能脏)dict 清洗为合法 SceneInput,返回 (场景, 兜底字段列表)。

    缺值/NaN 用季节相关的合理默认回退,并记录被兜底的字段,供 trace 标注。
    """
    fallbacks: list[str] = []
    raw_cabin = dict(raw.get("cabin", {}))
    season = raw_cabin.get("season") or "transition"
    season_default_temp = {"summer": 30.0, "winter": 5.0, "transition": 22.0}[season]

    _raw_amb = raw_cabin.get("ambient_temp")
    _amb_clean = _clean_number(_raw_amb, season_default_temp)
    if _raw_amb is None or _amb_clean != _raw_amb:  # 缺失 / NaN / 非法 → 兜底
        fallbacks.append("cabin.ambient_temp")
    raw_cabin["ambient_temp"] = _amb_clean
    # 车内温度:有效则保留(实测优先),NaN/缺失置 None(交由热模型回退估算)
    for _k in ("cabin_temp", "windshield_air_temp", "windshield_glass_temp"):
        _v = raw_cabin.get(_k)
        if _v is not None and math.isnan(_clean_number(_v, float("nan"))):
            raw_cabin[_k] = None
    raw_cabin["humidity"] = _clean_number(raw_cabin.get("humidity"), 50.0)
    raw_cabin["sun_driver_wm2"] = _clean_number(raw_cabin.get("sun_driver_wm2"), 0.0)
    raw_cabin["sun_passenger_wm2"] = _clean_number(
        raw_cabin.get("sun_passenger_wm2"), 0.0
    )
    raw_cabin["soc"] = _clean_number(raw_cabin.get("soc"), 80.0)
    raw_cabin["speed"] = _clean_number(raw_cabin.get("speed"), 0.0)
    raw_cabin["season"] = season

    cabin = CabinContext(**raw_cabin)

    occupants = [OccupantState(**dict(o)) for o in raw.get("occupants", [])]

    scene = SceneInput(cabin=cabin, occupants=occupants)
    return scene, fallbacks
