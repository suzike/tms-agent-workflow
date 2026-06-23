"""LLM 接入:provider 可配置(默认 DeepSeek),结构化输出 + Mock 规则引擎。

抽象 ``Decider.decide(payload) -> HVACDecision``:
- ``CloudDecider``  :LangChain 聊天模型 + with_structured_output,带超时。
- ``MockDecider``   :确定性规则引擎,无 key 即可跑通;同时作为降级链的兜底与测试 oracle。

payload 由 graph 的 llm_infer 节点组装(场景 + 舒适锚点 + 知识 + 记忆建议)。
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional, Protocol

from ..config import FAN_MAX, FAN_MIN, LLM_CONFIG, TEMP_MAX, TEMP_MIN, TEMP_STEP
from ..schemas import HVACDecision


class Decider(Protocol):
    def decide(self, payload: dict[str, Any]) -> HVACDecision: ...


_SYSTEM_PROMPT = (
    "你是汽车座舱空调控制专家,精通 ISO 7730 / ASHRAE 55 热舒适理论。"
    "根据给定的单座位场景、专业热舒适计算结果(PMV/PPD/目标舒适温度)与知识库原则,"
    "给出该座位的专业基准空调设定:风量 fan_level(1-7 整数)、"
    "温度 temp_set(15.5-31.5℃,精度0.5)、舒适基础出风模式 air_mode。"
    "air_mode 只能是三种舒适基础模式之一:face(吹面,夏季制冷)、"
    "feet(吹脚,冬季制热)、face_feet(吹面吹脚,春秋过渡)。"
    "不要输出任何除霜模式——除霜由安全层在起雾风险时自动叠加。"
    "可参考人员特征(老人/小孩偏暖、性别、BMI、衣着)、情绪与活动状态(睡眠宜安静低风量、"
    "兴奋代谢高偏凉),以及门窗开启(制冷效果下降)综合判断。"
    "注意:车内温度为热模型估算值(非实测),请结合不确定性稳健决策。"
    "【瞬态控制(必须遵循)】:负荷越大、车内温度离目标越远,设定温度越激进、风量越大,"
    "以加快收敛——夏季/制冷时车内越热则设定越低、风量越大;冬季/制热时车内越冷则设定越高、"
    "风量越大。随着车内逐步逼近目标,设定温度回归到最舒适目标值、风量降低以兼顾舒适与 NVH;"
    "稳态时取最舒适目标温度 + 最低风量。"
    "输入中的 transient_recommendation 已给出该逻辑的量化基准(setpoint/fan/phase):"
    "你的 temp_set 与 fan_level 必须与之同向且接近(允许按人员特征/情绪做小幅微调,"
    "不得违背瞬态方向,例如车内很热却给出接近目标的高设定或小风量)。"
    "【知识库(必须调用)】:输入的 knowledge 字段是注入的权威热舒适规则"
    "(ISO 7730 / ASHRAE 55 与本项目策略,包括出风模式选择、瞬态风温、人群偏好、节能等),"
    "其准确性高于你的通用常识。推理 温度/风量/出风模式 时必须显式依据这些规则;"
    "当 knowledge 与你的常识冲突时,一律以 knowledge 为准。在 reasoning 里简述所依据的规则。"
    "只输出结构化结果。"
)


def _clamp(v, lo, hi):
    return max(lo, min(v, hi))


def _round_temp(t: float) -> float:
    return round(_clamp(t, TEMP_MIN, TEMP_MAX) / TEMP_STEP) * TEMP_STEP


# ---------------------------------------------------------------------------
# 云端 LLM
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> dict:
    """从模型文本中提取首个 JSON 对象(容忍 ```json 代码块与多余文字)。"""
    cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"未能从输出中解析 JSON:{text[:120]}")
    return json.loads(match.group(0))


class CloudDecider:
    """通用云端决策器:让模型直接输出 JSON 再解析。

    刻意不使用 with_structured_output:其 json_schema(response_format)与
    function_calling(tool_choice)分别被 DeepSeek 普通/思考模型拒绝。手解析最兼容。
    """

    def __init__(self, model: str, api_key: str, base_url: Optional[str],
                 timeout: float):
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": model, "api_key": api_key, "temperature": 0.2,
            "timeout": timeout, "max_retries": 0,
        }
        if base_url:
            kwargs["base_url"] = base_url
        self._llm = ChatOpenAI(**kwargs)

    def decide(self, payload: dict[str, Any]) -> HVACDecision:
        human = (
            "场景与依据(JSON):\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + '\n\n请只输出一个 JSON 对象,不要任何多余文字或解释,格式:'
            + '{"fan_level": 整数1-7, "temp_set": 15.5到31.5之间且为0.5的倍数, '
            + '"air_mode": "face"或"feet"或"face_feet", "reasoning": "简短依据"}'
        )
        resp = self._llm.invoke([("system", _SYSTEM_PROMPT), ("human", human)])
        text = getattr(resp, "content", None) or str(resp)
        return HVACDecision(**_extract_json(text))

    def interpret(self, text: str, current):
        """把乘员自然语言诉求理解为调整后的座位设定(语音转文本接入)。"""
        from ..schemas import ZoneSetting

        sys = ("你是汽车座舱空调助手。把乘员的自然语言诉求理解为调整后的空调设定。"
               "只输出 JSON,不要多余文字。")
        human = (
            f"乘员对{current.seat_id}说:「{text}」。\n"
            f"当前设定:温度 {current.temp_set}℃,风量 {current.fan_level} 档,"
            f"出风 {current.air_mode}。\n请输出调整后的设定 JSON:"
            '{"fan_level": 整数1-7, "temp_set": 15.5-31.5且0.5倍数, '
            '"air_mode": "face"|"face_feet"|"feet"|"face_defrost"|"face_feet_defrost"'
            '|"feet_defrost"|"defrost", "reasoning": "简短依据"}'
        )
        resp = self._llm.invoke([("system", sys), ("human", human)])
        out = getattr(resp, "content", None) or str(resp)
        data = _extract_json(out)
        return ZoneSetting(
            seat_id=current.seat_id,
            fan_level=int(data.get("fan_level", current.fan_level)),
            temp_set=float(data.get("temp_set", current.temp_set)),
            air_mode=data.get("air_mode", current.air_mode),
            reasoning=data.get("reasoning", f"语音指令「{text}」"),
        )


# ---------------------------------------------------------------------------
# Mock 规则引擎(离线兜底 + 测试 oracle)
# ---------------------------------------------------------------------------
class MockDecider:
    """确定性规则引擎(专业基准):以舒适目标温度为基准,按温差定风量,按季节/日照定基础模式。

    只产出 3 个舒适基础模式;不处理除霜(安全层负责)、不融合记忆(逼近节点负责)。
    """

    def decide(self, payload: dict[str, Any]) -> HVACDecision:
        from ..tools.thermal_comfort import transient_setpoint_fan

        comfort = payload.get("comfort", {})
        target = float(comfort.get("target_temp", 24.0))
        pmv = float(comfort.get("pmv", 0.0))  # 冷热感由 PMV 给出(替代主观输入)
        cabin_temp = float(comfort.get("cabin_temp", target))  # 车内温度(实测/估算)
        season = payload.get("season", "transition")
        sun_wm2 = float(payload.get("seat_sun_wm2", 0.0))
        strong_sun = sun_wm2 >= 750.0  # 约半量程

        # 瞬态控制:优先采用 comfort 节点给出的瞬态建议(与注入 LLM 的口径一致),
        # 缺失时即时重算。车内远离目标 → 设定更激进 + 大风量;接近 → 回归目标 + 降风量(NVH)。
        tr = payload.get("transient_recommendation") or {}
        if tr.get("setpoint") is not None and tr.get("fan") is not None:
            temp, fan = float(tr["setpoint"]), int(tr["fan"])
        else:
            temp, fan = transient_setpoint_fan(cabin_temp, target)

        # 舒适基础出风模式:夏/偏热/暴晒→吹面;冬/偏冷→吹脚;过渡→吹面吹脚
        if season == "summer" or strong_sun or pmv > 0.5:
            mode = "face"
        elif season == "winter" or pmv < -0.5:
            mode = "feet"
        else:
            mode = "face_feet"

        reason = (f"目标{target}℃/车内{cabin_temp}℃→瞬态设定{temp}℃·风量{fan}"
                  f",PMV{pmv:+.1f}/{season}定模式{mode}")
        return HVACDecision(fan_level=fan, temp_set=temp, air_mode=mode, reasoning=reason)


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------
def provider_status() -> dict:
    """当前 LLM 接入状态(供 UI 展示)。"""
    cfg = LLM_CONFIG
    p = cfg.provider.lower()
    if p == "mock":
        return {"provider": "mock", "model": "rule-engine", "has_key": False,
                "mode": "离线规则引擎", "base_url": None}
    r = cfg.resolve()
    has = bool(r.get("api_key"))
    return {
        "provider": p, "model": r["model"], "has_key": has,
        "mode": "云端 LLM" if has else "未配置 Key → 离线 Mock 兜底",
        "base_url": r.get("base_url"),
    }


def ping_llm() -> tuple[bool, str]:
    """轻量连通性自检:用最小 payload 调一次,返回 (成功, 说明)。"""
    primary, _ = build_decider()
    if isinstance(primary, MockDecider):
        return False, "当前为 Mock 规则引擎(未配置云端 Key,离线可用)"
    payload = {
        "seat_id": "driver", "season": "transition", "weather": "sunny",
        "humidity": 50.0, "soc": 80.0, "seat_sun_wm2": 0.0,
        "person_category": "adult", "gender": "male", "clothing": "medium",
        "activity": "calm",
        "comfort": {"pmv": 0.0, "ppd": 5.0, "eqt": 24.0, "target_temp": 24.0,
                    "cabin_temp": 24.0},
        "knowledge": [],
    }
    try:
        d = primary.decide(payload)
        return True, f"连接成功 · 返回 {d.temp_set}℃/{d.fan_level}档/{d.air_mode}"
    except Exception as exc:
        return False, f"连接失败:{type(exc).__name__}: {exc}"


def build_decider() -> tuple[Decider, Decider]:
    """返回 (primary, fallback)。fallback 恒为 MockDecider(降级链兜底)。"""
    cfg = LLM_CONFIG
    fallback = MockDecider()
    if cfg.provider.lower() == "mock":
        return fallback, fallback
    resolved = cfg.resolve()
    if not resolved.get("api_key"):
        # 未配置 key → 主决策器也退回 Mock(可离线跑通)
        return fallback, fallback
    primary = CloudDecider(
        model=resolved["model"], api_key=resolved["api_key"],
        base_url=resolved.get("base_url"), timeout=cfg.timeout_seconds,
    )
    return primary, fallback
