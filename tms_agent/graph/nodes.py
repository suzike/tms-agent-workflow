"""图节点工厂。每个节点是闭包,绑定其依赖(registry / deciders)。

节点序列(单座位):featurize → recall → comfort → llm_infer → blend → safety
- llm_infer:产出"专业基准"决策(comfort+knowledge,不含记忆)。
- blend    :按持续逼近权重 w,把基准朝用户学到的偏好平滑拉近(渐进,非硬切换)。
"""
from __future__ import annotations

import json
from typing import Any

from ..config import BASE_AIR_MODES, FAN_MAX, FAN_MIN, TEMP_MAX, TEMP_MIN, TEMP_STEP
from ..features import featurize
from ..safety import apply_safety
from ..schemas import DecisionTrace, HVACDecision
from ..tools.thermal_comfort import approach_step, seat_air_temp
from .state import AgentState


def _clamp(v, lo, hi):
    return max(lo, min(v, hi))


def _round_temp(t: float) -> float:
    return round(_clamp(t, TEMP_MIN, TEMP_MAX) / TEMP_STEP) * TEMP_STEP


def _base_of(mode: str | None) -> str | None:
    """剥离除霜叠加,取舒适基础模式;纯除霜无基础(环境驱动),返回 None。"""
    if not mode or mode == "defrost":
        return None
    return mode.replace("_defrost", "")


def _build_payload(state: AgentState) -> dict[str, Any]:
    """组装 LLM 输入:场景 + 人员特征/状态 + 舒适锚点 + 知识(专业基准,个性化由 blend 负责)。"""
    cabin = state["cabin"]
    occ = state["occupant"]
    # 注意:刻意不放 seat_id —— 热舒适只取决于物理输入,与座位无关。
    # 否则主/副驾物理输入相同也会因 payload 不同而被 LLM 当成不同请求(且温度>0 非确定),
    # 导致同输入异输出;去掉后相同输入 → 相同 payload → 缓存命中 → 主副驾一致。
    return {
        "season": cabin.season,
        "weather": cabin.weather,
        "humidity": cabin.humidity,
        "soc": cabin.soc,
        "speed": cabin.speed,
        "any_opening": cabin.any_opening(),       # 门窗开启(影响制冷效果)
        "max_window_open_pct": cabin.max_window_open(),
        "seat_sun_wm2": cabin.seat_sun(occ.seat_id),
        # 人员特征与状态
        "person_category": occ.category,
        "gender": occ.gender,
        "age": occ.age,
        "bmi": occ.bmi,
        "clothing": occ.clothing,
        "emotion": occ.emotion,
        "activity": occ.activity,
        # 车内温度为估算值,随 comfort 一并提供
        "comfort": state["comfort"].model_dump(),
        # 瞬态控制建议(确定性专业基准:负荷大→设定激进+大风量;趋稳→回归舒适+降风量)
        "transient_recommendation": state.get("transient", {}),
        "knowledge": state.get("knowledge", []),
        **_preference_payload(state),
    }


def _preference_payload(state: AgentState) -> dict[str, Any]:
    """有历史偏好时,注入偏好与"逐步逼近"提示,让 LLM 实时推理时真正调用该知识。"""
    mem = state.get("memory")
    if mem is None or not mem.has_preference:
        return {}
    return {
        "user_preference": {
            "temp_set": mem.pref_temp, "fan_level": mem.pref_fan,
            "air_mode": mem.pref_mode,
        },
        "approach_note": (
            "该用户在相似场景有历史偏好。若与专业推荐差异较大,请遵循知识库"
            "「逐步逼近」策略分 2-3 次沿专业推荐向偏好逼近(差值越大首步越大),"
            "不要一次性跳到偏好值。"
        ),
    }


def make_featurize_node():
    def featurize_node(state: AgentState) -> dict:
        cabin = state["cabin"]
        occ = state["occupant"]
        ta, measured = seat_air_temp(cabin, occ.seat_id)
        detail = {
            "座位": occ.seat_id, "用户": occ.user_id,
            "车外温度℃": cabin.ambient_temp,
            "车内温度℃": f"{ta}({'实测' if measured else '估算'})",
            "座位日照W/m²": cabin.seat_sun(occ.seat_id),
            "相对湿度%": cabin.humidity, "天气": cabin.weather, "季节": cabin.season,
            "车速km/h": cabin.speed, "电量%": cabin.soc,
            "车窗开度%": cabin.window_open_pct(occ.seat_id),
            "门窗开启": cabin.any_opening(), "最大除霜": cabin.max_defrost,
            "年龄": occ.age, "性别": occ.gender, "BMI": occ.bmi, "类别": occ.category,
            "衣着": occ.clothing, "活动": occ.activity, "情绪": occ.emotion,
        }
        return {
            "scene_vector": featurize(cabin, occ).tolist(),
            "feature_detail": detail,
        }

    return featurize_node


def make_recall_node(registry):
    skill = registry.get("memory")

    def recall_node(state: AgentState) -> dict:
        occ = state["occupant"]
        out = skill.invoke(
            {
                "user_id": occ.user_id,
                "seat_id": occ.seat_id,
                "scene_vector": state["scene_vector"],
                "now": state.get("now"),
            }
        )
        return {"memory": out["memory"]}

    return recall_node


def make_comfort_node(registry):
    tc_skill = registry.get("thermal_comfort")
    kn_skill = registry.get("knowledge")

    def comfort_node(state: AgentState) -> dict:
        mem = state.get("memory")
        ctx = {"cabin": state["cabin"], "occupant": state["occupant"],
               "current_fan": state.get("current_fan"),
               "has_preference": bool(mem and mem.has_preference)}
        tc_out = tc_skill.invoke(ctx)
        kn = kn_skill.invoke(ctx)
        return {
            "comfort": tc_out["comfort_metrics"],
            "comfort_breakdown": tc_out.get("comfort_breakdown", {}),
            "transient": tc_out.get("transient", {}),
            "assumptions": tc_out.get("assumptions", []),
            "knowledge": kn["knowledge_snippets"],
            "knowledge_titles": kn["knowledge_titles"],
        }

    return comfort_node


def make_llm_node(deciders):
    primary, fallback = deciders
    cache: dict[str, tuple] = {}  # 内容键缓存:相同场景不重复调用云端(随 engine 存活)

    def llm_node(state: AgentState) -> dict:
        payload = _build_payload(state)
        key = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        if key in cache:
            dec, source = cache[key]
            return {"decision": dec, "source": source,
                    "llm_raw": {"cached": True}}
        try:
            dec = primary.decide(payload)
            source = "llm"
            raw = dec.model_dump()
        except Exception as exc:  # 降级链:云端失败/超时 → 规则兜底
            dec = fallback.decide(payload)
            source = "fallback"
            raw = {"error": str(exc)}
        cache[key] = (dec, source)
        return {"decision": dec, "source": source, "llm_raw": raw}

    return llm_node


def make_approach_node():
    """逐步逼近(游标式迭代):每次推理沿"当前游标→用户偏好"迈一步,2-3 次收敛。

    游标在 engine 跨推理持久化(非冷静期才推进),实现 7→5→4→3 式的逐步逼近,
    而非一次到位。LLM 已在 payload 中获知偏好与逼近规则(实时推理);本节点为确定性兜底。
    """

    def approach_node(state: AgentState) -> dict:
        dec: HVACDecision = state["decision"]
        mem = state["memory"]
        if not mem.has_preference:
            return {"approach_weight": 0.0, "learned_preference": None,
                    "approach_cursor_next": None}

        # 游标:首次从专业推荐(LLM 基准)起步,之后用持久化的上一步推荐
        cursor = state.get("approach_cursor") or {
            "temp": dec.temp_set, "fan": dec.fan_level, "mode": dec.air_mode}
        pref_mode = _base_of(mem.pref_mode)
        if pref_mode not in BASE_AIR_MODES:
            pref_mode = dec.air_mode

        next_temp = approach_step(float(cursor["temp"]), float(mem.pref_temp),
                                  TEMP_STEP, lo=TEMP_MIN, hi=TEMP_MAX)
        next_fan = int(approach_step(float(cursor["fan"]), float(mem.pref_fan),
                                     1.0, lo=FAN_MIN, hi=FAN_MAX))
        blended = HVACDecision(
            fan_level=next_fan, temp_set=next_temp, air_mode=pref_mode,
            reasoning=(f"{dec.reasoning};逐步逼近用户偏好"
                       f"({mem.pref_temp}℃/{mem.pref_fan}档),本步 {next_temp}℃/{next_fan}档"),
        )
        start = float(cursor["temp"])
        denom = abs(start - mem.pref_temp) or 1.0
        progress = 1.0 if start == mem.pref_temp else max(
            0.0, round(1 - abs(next_temp - mem.pref_temp) / denom, 2))
        return {
            "decision": blended,
            "approach_weight": progress,
            "learned_preference": {"temp": mem.pref_temp, "fan": mem.pref_fan,
                                   "mode": mem.pref_mode},
            "approach_cursor_next": {"temp": next_temp, "fan": next_fan,
                                     "mode": pref_mode},
        }

    return approach_node


def make_safety_node():
    def safety_node(state: AgentState) -> dict:
        occ = state["occupant"]
        mem = state["memory"]
        setting, adjustments = apply_safety(
            state["decision"], state["cabin"], occ, state.get("defog")
        )
        trace = DecisionTrace(
            seat_id=occ.seat_id,
            user_id=occ.user_id,
            source=state["source"],
            memory_evidence=mem.evidence_count,
            approach_weight=state.get("approach_weight", 0.0),
            learned_preference=state.get("learned_preference"),
            comfort_metrics=state.get("comfort"),
            comfort_breakdown=state.get("comfort_breakdown", {}),
            assumptions=state.get("assumptions", []),
            knowledge_snippets=state.get("knowledge_titles", []),
            memory_hit_count=mem.cluster_size,
            llm_raw=state.get("llm_raw"),
            safety_adjustments=adjustments,
            fallback_fields=state.get("fallback_fields", []),
            final=setting,
        )
        return {"setting": setting, "trace": trace}

    return safety_node
