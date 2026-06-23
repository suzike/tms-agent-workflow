"""单座位推理图的状态定义。

图按"单个乘员"运行;engine 对每位在座乘员各跑一次(多温区独立预测)。
"""
from __future__ import annotations

from typing import Any, Optional

from typing_extensions import TypedDict

from ..schemas import (
    CabinContext,
    ComfortMetrics,
    DecisionTrace,
    DefogDecision,
    HVACDecision,
    OccupantState,
    ZoneSetting,
)
from ..memory.store import MemoryRecall


class AgentState(TypedDict, total=False):
    # 输入
    cabin: CabinContext
    occupant: OccupantState
    fallback_fields: list[str]
    now: float          # 推理时刻,贯穿记忆时间衰减口径
    current_fan: int    # 当前空调风量档位(用于评估 PMV 的风速)
    defog: Optional[DefogDecision]  # 独立除雾 Agent 的判定(车厢级,供安全层叠加除霜)
    approach_cursor: Optional[dict[str, Any]]  # 逐步逼近游标(上一步推荐;跨推理持久化)
    # 中间态
    scene_vector: list[float]
    feature_detail: dict
    memory: MemoryRecall
    comfort: ComfortMetrics
    comfort_breakdown: dict
    transient: dict        # 瞬态控制建议(设定/风量/阶段),供 LLM 与兜底遵循
    assumptions: list[str]
    knowledge: list[str]
    knowledge_titles: list[str]
    decision: HVACDecision
    source: str
    llm_raw: Optional[dict[str, Any]]
    approach_weight: float        # 逐步逼近进度(展示用)
    learned_preference: Optional[dict[str, Any]]
    approach_cursor_next: Optional[dict[str, Any]]  # 本次逼近后的新游标
    # 输出
    setting: ZoneSetting
    trace: DecisionTrace
