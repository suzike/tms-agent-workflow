"""智能除雾 Agent:独立于舒适 Agent 的专用 LangGraph(感知→检索→判定)。

- 仅消费起雾相关输入(玻璃温度/玻璃附近空气温度/玻璃附近湿度/雨量信号),与舒适解耦。
- 自带专属工具(defog/tools)、专属知识库(defog/docs)、专属 LLM 决策器(defog/decider)。
- 每场景运行一次(车厢级,非按座位);输出 DefogDecision 供舒适 Agent 的安全层叠加除霜。
- 安全取严:确定性规则与 LLM 取更紧急者,除雾强度不低于物理规则下限。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from ..knowledge.retriever import TfidfRetriever, build_default_retriever
from ..schemas import CabinContext, DefogDecision
from .decider import RuleDefogDecider, build_defog_decider

# 除雾专属知识库目录(独立于舒适知识库)
DEFOG_DOCS_DIR = Path(__file__).resolve().parent / "docs"

_SEV = {"none": 0, "mild": 1, "strong": 2}
_NODE_TITLE = {
    "sense": "① 玻璃环境感知",
    "knowledge": "② 除雾知识检索",
    "decide": "③ 除雾判定(前馈+反馈)",
    "final": "✅ 除雾决策",
}


@dataclass
class DefogStep:
    """除雾 Agent 推理链中的一步(用于可视化)。"""

    node: str
    title: str
    detail: str
    data: Optional[dict] = None


class DefogState(TypedDict, total=False):
    cabin: CabinContext
    now: float
    features: dict
    knowledge: list[str]
    knowledge_titles: list[str]
    decision: DefogDecision
    source: str
    llm_raw: Optional[dict]


def _more_urgent(a: DefogDecision, b: DefogDecision) -> DefogDecision:
    """取更紧急者(紧急度严重者优先)。"""
    return a if _SEV[a.level] >= _SEV[b.level] else b


def build_defog_retriever() -> TfidfRetriever:
    """构建除雾专属知识检索器(仅索引 defog/docs)。"""
    return build_default_retriever(DEFOG_DOCS_DIR)


def _make_sense_node():
    def sense_node(state: DefogState) -> dict:
        c = state["cabin"]
        features = {
            "玻璃表面温度℃": c.windshield_glass_temp,
            "玻璃附近空气温度℃": c.windshield_air_temp,
            "玻璃附近湿度%": c.humidity,
            "雨量信号": c.rain_level,
            "天气": c.weather,
        }
        return {"features": features}

    return sense_node


def _make_knowledge_node(retriever: TfidfRetriever, k: int = 3):
    def knowledge_node(state: DefogState) -> dict:
        c = state["cabin"]
        kw = ["除雾", "起雾", "玻璃", "露点", "裕度"]
        if c.rain_level != "none" or c.weather in ("rain", "snow"):
            kw.append("雨量 前馈 外循环 除湿")
        if c.humidity >= 80:
            kw.append("高湿 湿源 凝结")
        if c.windshield_glass_temp is not None:
            kw.append("玻璃温度 反馈 提前介入")
        if c.season == "winter":
            kw.append("冬季 冷启动 玻璃冷")
        hits = retriever.retrieve(" ".join(kw), k)
        return {
            "knowledge": [f"{ch.title}:{ch.text}" for ch, _ in hits],
            "knowledge_titles": [ch.title for ch, _ in hits],
        }

    return knowledge_node


def _make_decide_node(deciders):
    primary, fallback = deciders
    rule = fallback if isinstance(fallback, RuleDefogDecider) else RuleDefogDecider()
    cache: dict[str, tuple] = {}

    def decide_node(state: DefogState) -> dict:
        c = state["cabin"]
        base = {
            "windshield_glass_temp": c.windshield_glass_temp,
            "windshield_air_temp": c.windshield_air_temp,
            "humidity": c.humidity,
            "rain_level": c.rain_level,
            "weather": c.weather,
            "season": c.season,
        }
        key = json.dumps(base, sort_keys=True, ensure_ascii=False, default=str)
        if key in cache:
            dec, source = cache[key]
            return {"decision": dec, "source": source, "llm_raw": {"cached": True}}

        rule_dec = rule.decide(base)  # 物理规则:安全下限
        if primary is rule or isinstance(primary, RuleDefogDecider):
            cache[key] = (rule_dec, "rule")
            return {"decision": rule_dec, "source": "rule", "llm_raw": None}

        # 云端 LLM(以物理诊断+知识接地)+ 安全取严
        llm_payload = {**base, "diagnostics": rule_dec.diagnostics,
                       "knowledge": state.get("knowledge", [])}
        try:
            llm_dec = primary.decide(llm_payload)
            chosen = _more_urgent(rule_dec, llm_dec)
            note = ("" if chosen is llm_dec
                    else f";安全取严至物理规则下限 {rule_dec.level}")
            final = DefogDecision(
                need_defog=chosen.need_defog, level=chosen.level, source="llm",
                reasoning=(llm_dec.reasoning or "LLM 判定") + note,
                diagnostics=rule_dec.diagnostics,
            )
            raw = {"llm_level": llm_dec.level, "rule_level": rule_dec.level}
            source = "llm"
        except Exception as exc:  # 降级链:LLM 失败 → 规则兜底
            final, source, raw = rule_dec, "fallback", {"error": str(exc)}
        cache[key] = (final, source)
        return {"decision": final, "source": source, "llm_raw": raw}

    return decide_node


def build_defog_graph(retriever: TfidfRetriever, deciders):
    g = StateGraph(DefogState)
    g.add_node("sense", _make_sense_node())
    g.add_node("knowledge", _make_knowledge_node(retriever))
    g.add_node("decide", _make_decide_node(deciders))
    g.add_edge(START, "sense")
    g.add_edge("sense", "knowledge")
    g.add_edge("knowledge", "decide")
    g.add_edge("decide", END)
    return g.compile()


def _summarize_step(node: str, update: dict) -> str:
    if node == "sense":
        f = update.get("features", {})
        g = f.get("玻璃表面温度℃")
        gs = f"玻璃{g}℃" if g is not None else "玻璃温度未接入"
        return (f"{gs} · 玻璃附近湿度{f.get('玻璃附近湿度%')}% · "
                f"雨量{f.get('雨量信号')} · 天气{f.get('天气')}")
    if node == "knowledge":
        titles = update.get("knowledge_titles", [])
        return "检索除雾知识:" + (", ".join(titles[:2]) if titles else "无")
    if node == "decide":
        d = update.get("decision")
        src = {"llm": "LLM", "fallback": "降级", "rule": "规则"}.get(
            update.get("source"), "")
        if d is None:
            return ""
        return (f"判定({src}):{'需要除雾' if d.need_defog else '无需除雾'}"
                f"(紧急度 {d.level});{d.reasoning}")
    return str(update)


class DefogAgent:
    """智能除雾 Agent 封装:构建图、对外 assess()/stream()。"""

    def __init__(self, retriever: Optional[TfidfRetriever] = None, deciders=None):
        self.retriever = retriever or build_defog_retriever()
        self.deciders = deciders or build_defog_decider()
        self.graph = build_defog_graph(self.retriever, self.deciders)

    def assess(self, cabin: CabinContext, now: Optional[float] = None) -> DefogDecision:
        state = self.graph.invoke({"cabin": cabin, "now": now})
        return state["decision"]

    def stream(self, cabin: CabinContext, now: Optional[float] = None):
        """逐节点 yield DefogStep,末尾给出决策(用于推理链可视化)。"""
        decision: Optional[DefogDecision] = None
        for chunk in self.graph.stream({"cabin": cabin, "now": now},
                                       stream_mode="updates"):
            for node, update in chunk.items():
                if node == "decide":
                    decision = update.get("decision")
                data = update.get("features") if node == "sense" else None
                yield DefogStep(node, _NODE_TITLE.get(node, node),
                                _summarize_step(node, update), data=data)
        if decision is not None:
            tail = (f"叠加除霜({decision.level})" if decision.need_defog
                    else "无需除霜,保持舒适基础模式")
            yield DefogStep("final", _NODE_TITLE["final"], tail)
