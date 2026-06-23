"""组装单座位推理状态图。"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from . import nodes
from .state import AgentState


def build_graph(registry, deciders):
    """构建并编译图。registry/deciders 注入,节点为闭包(依赖抽象)。"""
    g = StateGraph(AgentState)

    g.add_node("featurize", nodes.make_featurize_node())
    g.add_node("recall", nodes.make_recall_node(registry))
    g.add_node("comfort", nodes.make_comfort_node(registry))
    g.add_node("llm_infer", nodes.make_llm_node(deciders))
    g.add_node("approach", nodes.make_approach_node())
    g.add_node("safety", nodes.make_safety_node())

    g.add_edge(START, "featurize")
    g.add_edge("featurize", "recall")
    g.add_edge("recall", "comfort")
    g.add_edge("comfort", "llm_infer")  # LLM 产出专业基准(已获知偏好+逼近规则)
    g.add_edge("llm_infer", "approach") # 游标式逐步逼近用户偏好(确定性兜底)
    g.add_edge("approach", "safety")    # 安全层叠加除霜并落定
    g.add_edge("safety", END)

    return g.compile()
