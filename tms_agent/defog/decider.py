"""智能除雾 Agent 的决策器:确定性规则 + 云端 LLM,均产出 DefogDecision。

- ``RuleDefogDecider`` :纯物理规则(前馈+反馈),无 key 即可跑通,作降级兜底与测试 oracle。
- ``CloudDefogDecider``:除雾专用提示词的 LLM,直接输出 JSON 再解析(与舒适 Decider 同口径,
  不用 with_structured_output 以兼容 DeepSeek)。
安全取严:Agent 在 decide 节点取规则与 LLM 的更紧急者(见 agent.py),除雾强度不低于规则下限。
"""
from __future__ import annotations

import json
from typing import Any, Optional, Protocol

from ..config import LLM_CONFIG
from ..llm.provider import _extract_json
from ..schemas import DefogDecision
from .tools import defog_urgency


class DefogDecider(Protocol):
    def decide(self, payload: dict[str, Any]) -> DefogDecision: ...


_DEFOG_SYSTEM_PROMPT = (
    "你是汽车前风挡智能除雾安全 Agent,只负责判断是否需要除雾以及紧急度,"
    "不负责温度/风量等舒适设定。判定依据仅四项:玻璃表面温度、玻璃附近空气温度、"
    "玻璃附近相对湿度、雨量信号(无/小雨/中雨/大雨)。"
    "物理原理:玻璃温度低于其附近空气露点即结露起雾;裕度=玻璃温度−露点,越小越危险。"
    "采用前馈(雨量/高湿预判)+反馈(玻璃温度 vs 露点裕度)。除雾关乎视线,安全优先,宜提前介入、判定从严。"
    "输出紧急度 level:none(无需)、mild(轻度预防叠加除霜)、strong(强除雾,需保证气流)。"
    "你只决定是否叠加除霜,绝不输出纯除霜(纯除霜仅由 MAX 按键触发)。只输出结构化结果。"
)


class RuleDefogDecider:
    """确定性物理规则除雾决策器(前馈+反馈)。"""

    def decide(self, payload: dict[str, Any]) -> DefogDecision:
        diag = defog_urgency(
            glass_temp=payload.get("windshield_glass_temp"),
            air_temp=payload.get("windshield_air_temp"),
            humidity=float(payload.get("humidity", 50.0)),
            rain_level=payload.get("rain_level", "none"),
            weather=payload.get("weather", "sunny"),
        )
        return DefogDecision(
            need_defog=diag["need_defog"],
            level=diag["level"],
            source="rule",
            reasoning=f"规则判定:{diag['detail']} → {diag['level']}",
            diagnostics=diag,
        )


class CloudDefogDecider:
    """除雾专用云端 LLM 决策器:直接输出 JSON 再解析。"""

    def __init__(self, model: str, api_key: str, base_url: Optional[str],
                 timeout: float):
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": model, "api_key": api_key, "temperature": 0.1,
            "timeout": timeout, "max_retries": 0,
        }
        if base_url:
            kwargs["base_url"] = base_url
        self._llm = ChatOpenAI(**kwargs)

    def decide(self, payload: dict[str, Any]) -> DefogDecision:
        human = (
            "前风挡除雾场景与物理诊断(JSON):\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + '\n\n请只输出一个 JSON 对象,不要任何多余文字,格式:'
            + '{"need_defog": true或false, "level": "none"或"mild"或"strong", '
            + '"reasoning": "简短依据"}'
        )
        resp = self._llm.invoke(
            [("system", _DEFOG_SYSTEM_PROMPT), ("human", human)]
        )
        text = getattr(resp, "content", None) or str(resp)
        data = _extract_json(text)
        level = data.get("level", "none")
        if level not in ("none", "mild", "strong"):
            level = "strong" if data.get("need_defog") else "none"
        return DefogDecision(
            need_defog=bool(data.get("need_defog", level != "none")),
            level=level,
            source="llm",
            reasoning=data.get("reasoning", ""),
            diagnostics=payload.get("diagnostics", {}),
        )


def build_defog_decider() -> tuple[DefogDecider, DefogDecider]:
    """返回 (primary, fallback)。fallback 恒为规则决策器(降级链兜底)。"""
    cfg = LLM_CONFIG
    fallback = RuleDefogDecider()
    if cfg.provider.lower() == "mock":
        return fallback, fallback
    resolved = cfg.resolve()
    if not resolved.get("api_key"):
        return fallback, fallback
    primary = CloudDefogDecider(
        model=resolved["model"], api_key=resolved["api_key"],
        base_url=resolved.get("base_url"), timeout=cfg.timeout_seconds,
    )
    return primary, fallback
