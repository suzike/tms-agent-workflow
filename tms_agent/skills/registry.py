"""具体 Skill 实现与默认注册表构建。"""
from __future__ import annotations

from typing import Any, Optional

from ..config import CURRENT_FAN_DEFAULT, TEMP_MAX, TEMP_MIN
from ..knowledge.retriever import TfidfRetriever, build_default_retriever
from ..memory.store import MemoryRecall, MemoryStore
from ..schemas import CabinContext, ComfortMetrics, OccupantState
from ..tools import thermal_comfort as tc
from .base import Skill, SkillRegistry


class ThermalComfortSkill(Skill):
    name = "thermal_comfort"
    description = "按 ISO 7730 计算该座位 PMV/PPD/EQT 与目标舒适温度(含活动代谢与人体特征修正)。"

    def invoke(self, context: dict[str, Any]) -> dict[str, Any]:
        cabin: CabinContext = context["cabin"]
        occ: OccupantState = context["occupant"]
        seat = occ.seat_id
        sun = cabin.seat_sun(seat)
        # 车内温度:实测优先,缺失时热模型回退估算
        ta, ta_measured = tc.seat_air_temp(cabin, seat)
        # 风速:由"当前空调风量档位"查表得到(评估当前热状态)
        cur_fan = int(context.get("current_fan", CURRENT_FAN_DEFAULT))
        vel = tc.air_velocity_from_fan(cur_fan)
        clo = tc.clo_from_clothing(occ.clothing)   # 衣着 → clo(实测量)
        met = tc.met_from_activity(occ.activity)   # 活动 → met
        mrt = tc.mrt_from_sun(ta, sun)             # MRT 由车内温+日照估算
        pmv, ppd = tc.compute_pmv_ppd(ta, mrt, vel, cabin.humidity, met, clo)
        eqt = tc.equivalent_temperature(ta, mrt, vel)
        # 目标温度 = PMV≈0 锚点(用衣着 clo + 当前风速)+ 人体特征启发式偏移
        base_target = tc.target_comfort_temp(
            sun, cabin.humidity, cabin.season, air_velocity=vel, met=met, clo=clo
        )
        offset = tc.comfort_offset(occ.category, occ.gender, occ.bmi)
        target = round(max(TEMP_MIN, min(base_target + offset, TEMP_MAX)) * 2) / 2
        # 瞬态控制:由"车内温度 vs 目标"给出过渡期设定与风量(负荷大→设定更激进+大风量;
        # 趋近稳态→设定回归目标+降风量兼顾 NVH)。作为温度/风量的确定性专业基准。
        t_set, t_fan = tc.transient_setpoint_fan(ta, target)
        load = round(ta - target, 1)
        if abs(load) <= 1.0:
            phase = "steady"      # 稳态:回归最舒适设定 + 最低风量(NVH)
        elif load > 0:
            phase = "cooldown"    # 制冷过渡:车内高于目标,设定更低 + 大风量快速降温
        else:
            phase = "warmup"      # 制热过渡:车内低于目标,设定更高 + 大风量快速升温
        transient = {"setpoint": t_set, "fan": t_fan, "phase": phase, "load": load,
                     "target": target}
        ta_src = "实测" if ta_measured else f"估算(车外 {cabin.ambient_temp:.0f}℃+日照 {sun:.0f}W/m²)"
        assumptions = [
            f"车内温度={ta_src} {ta}℃",
            "平均辐射温度 MRT 由车内温度+日照估算",
            f"风速={vel} m/s(由当前风量 {cur_fan} 档查表)",
        ]
        # 计算过程中间量(供 UI 可视化:车内温度 / PMV / PPD 推导)
        breakdown = {
            "ta": ta, "ta_measured": ta_measured,
            "ambient": cabin.ambient_temp, "sun": sun,
            "window": cabin.window_open_pct(seat), "speed": cabin.speed,
            "mrt": round(mrt, 1), "mrt_gain": round(mrt - ta, 1),
            "fan": cur_fan, "velocity": vel, "rh": cabin.humidity,
            "met": met, "activity": occ.activity, "clo": clo, "clothing": occ.clothing,
            "pmv": pmv, "ppd": ppd, "eqt": eqt,
            "base_target": base_target, "offset": offset, "target": target,
            "transient": transient,
        }
        return {
            "comfort_metrics": ComfortMetrics(
                pmv=pmv, ppd=ppd, eqt=eqt, target_temp=target, cabin_temp=ta
            ),
            "assumptions": assumptions,
            "comfort_breakdown": breakdown,
            "transient": transient,
        }


class KnowledgeSkill(Skill):
    name = "knowledge"
    description = "按场景关键词检索热舒适知识库(ISO/ASHRAE/汽车经验),返回相关原则。"

    def __init__(self, retriever: Optional[TfidfRetriever] = None, k: int = 3):
        self._retriever = retriever or build_default_retriever()
        self._k = k

    @staticmethod
    def _build_query(cabin: CabinContext, occ: OccupantState,
                     has_preference: bool = False) -> str:
        kw: list[str] = ["出风模式", "风量", "温度", "舒适"]
        kw.append(
            {"summer": "夏季 高温 制冷", "winter": "冬季 寒冷 制热 暖脚",
             "transition": "过渡季"}[cabin.season]
        )
        if cabin.seat_sun(occ.seat_id) >= 750:  # 约半量程,强日照
            kw.append("暴晒 日照 辐射 吹脸 降温")
        # 季节性/强负荷场景:检索瞬态控制(快速降温升温→稳态)策略
        if cabin.season in ("summer", "winter") or cabin.seat_sun(occ.seat_id) >= 750:
            kw.append("瞬态 快速降温 快速升温 稳态 风量 NVH 过冲 负荷")
        if cabin.weather in ("rain", "snow") or cabin.humidity >= 70:
            kw.append("除雾 起雾 湿度 雨天 安全")
        if occ.category in ("child", "elderly"):
            kw.append("老人 小孩 婴儿 敏感 偏暖 柔和")
        if occ.activity == "sleeping":
            kw.append("睡眠 安静 低风量 噪声")
        elif occ.activity == "excited":
            kw.append("活动 代谢 偏凉")
        if cabin.any_opening():
            kw.append("开窗 开门 新风 制冷效果 热负荷")
        if cabin.soc < 30:
            kw.append("电量低 节能 续航")
        if has_preference:
            kw.append("逐步逼近 用户偏好 记忆 收敛")
        return " ".join(kw)

    def invoke(self, context: dict[str, Any]) -> dict[str, Any]:
        cabin: CabinContext = context["cabin"]
        occ: OccupantState = context["occupant"]
        query = self._build_query(cabin, occ, context.get("has_preference", False))
        hits = self._retriever.retrieve(query, self._k)
        return {
            "knowledge_snippets": [f"{c.title}:{c.text}" for c, _ in hits],
            "knowledge_titles": [c.title for c, _ in hits],
        }


class WeatherSkill(Skill):
    """Mock 天气数据源(PoC 天气已在 cabin 内;此处演示数据源能力,预留真实接口)。"""

    name = "weather"
    description = "提供天气参数(PoC 直接回显场景中的天气)。"

    def invoke(self, context: dict[str, Any]) -> dict[str, Any]:
        cabin: CabinContext = context["cabin"]
        return {"weather": cabin.weather, "ambient_temp": cabin.ambient_temp,
                "humidity": cabin.humidity}


class VehicleSkill(Skill):
    """Mock 车辆状态数据源(预留 CAN/SOA 真实接口)。"""

    name = "vehicle"
    description = "提供车辆状态(电量/车速等),PoC 回显场景。"

    def invoke(self, context: dict[str, Any]) -> dict[str, Any]:
        cabin: CabinContext = context["cabin"]
        return {"soc": cabin.soc, "speed": cabin.speed}


_STRATEGY_NOTE = {
    "max_defrost": "最大除霜:集中最大气流除雾,牺牲体感优先视线。",
    "defrost_overlay": "起雾风险:在基础模式上叠加除霜,安全优先。",
    "quiet_gentle": "睡眠:安静低风量、避免直吹、略升温。",
    "gentle_warm": "老人/小孩:柔和分散出风、偏暖、不直吹头部。",
    "rapid_cooldown": "夏季/暴晒:先大风量吹面快速降温,达标后回落稳态。",
    "warmup_feet": "冬季:吹脚为主、头凉脚暖、设定不过高。",
    "steady": "过渡季:吹面吹脚均衡、中低风量、平顺保持。",
}


class StrategySkill(Skill):
    """场景 → 高层热管理策略标签(供推理/编排参考)。"""

    name = "strategy"
    description = "根据季节/日照/人群/活动/起雾给出高层空调策略(快速降温/暖脚/柔和/除霜等)。"

    def invoke(self, context: dict[str, Any]) -> dict[str, Any]:
        cabin: CabinContext = context["cabin"]
        occ: OccupantState = context["occupant"]
        ta, _ = tc.seat_air_temp(cabin, occ.seat_id)
        # 结露起雾主要发生在车外低温或雨雪天;高温天不评估起雾,避免误报
        fog_relevant = cabin.ambient_temp <= 18 or cabin.weather in ("rain", "snow")
        risk = tc.fog_risk(ta, cabin.ambient_temp, cabin.humidity) if fog_relevant else "low"
        if cabin.max_defrost:
            tag = "max_defrost"
        elif risk == "high":
            tag = "defrost_overlay"
        elif occ.activity == "sleeping":
            tag = "quiet_gentle"
        elif occ.category in ("child", "elderly"):
            tag = "gentle_warm"
        elif cabin.season == "summer" or cabin.seat_sun(occ.seat_id) >= 750:
            tag = "rapid_cooldown"
        elif cabin.season == "winter":
            tag = "warmup_feet"
        else:
            tag = "steady"
        return {"strategy": tag, "strategy_note": _STRATEGY_NOTE[tag],
                "fog_risk": risk}


class EnergyAdvisorSkill(Skill):
    """节能与内外循环建议。"""

    name = "energy"
    description = "根据电量/日照/天气给出内外循环与节能建议。"

    def invoke(self, context: dict[str, Any]) -> dict[str, Any]:
        cabin: CabinContext = context["cabin"]
        occ: OccupantState = context["occupant"]
        recirc = tc.recirculation_hint(
            cabin.seat_sun(occ.seat_id), cabin.ambient_temp, cabin.soc, cabin.weather
        )
        target = context.get("temp_set", cabin.ambient_temp)
        return {
            "recirculation": recirc,
            "energy_hint": tc.energy_hint(cabin.soc, target, cabin.ambient_temp),
        }


class MemorySkill(Skill):
    name = "memory"
    description = "按 用户×座位 召回历史修正记忆,提供渐进置信与代表修正值。"

    def __init__(self, store: MemoryStore):
        self._store = store

    def invoke(self, context: dict[str, Any]) -> dict[str, Any]:
        recall: MemoryRecall = self._store.recall(
            context["user_id"],
            context["seat_id"],
            context["scene_vector"],
            now=context.get("now"),  # 贯穿 now,保证时间衰减口径一致
        )
        return {"memory": recall}


def build_default_registry(
    store: MemoryStore, retriever: Optional[TfidfRetriever] = None
) -> SkillRegistry:
    reg = SkillRegistry()
    reg.register(ThermalComfortSkill())
    reg.register(KnowledgeSkill(retriever))
    reg.register(StrategySkill())
    reg.register(EnergyAdvisorSkill())
    reg.register(WeatherSkill())
    reg.register(VehicleSkill())
    reg.register(MemorySkill(store))
    return reg
