"""核心引擎:统一对外接口,串联图、记忆、防抖、冷静期与指标。

- infer(scene):对每位在座乘员独立跑图(多温区独立预测),叠加防抖与冷静期,
  返回各座位"实际生效设定"与决策留痕。
- apply_correction(...):记录用户对某座位的修正,写入 (user×seat) 记忆并锁定该座位。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from . import config
from .defog import DefogAgent
from .features import featurize
from .graph.build import build_graph
from .llm.provider import build_decider
from .memory.store import MemoryStore
from .observability import SessionMetrics, get_logger
from .safety import is_significant_change, lock_until, seat_locked
from .schemas import (
    CorrectionRecord,
    DecisionTrace,
    SceneInput,
    ZoneSetting,
)
from .skills.registry import build_default_registry

_log = get_logger()


@dataclass
class InferenceResult:
    """一次推理结果:各座位生效设定 / 留痕 / 是否自动应用。"""

    settings: dict[str, ZoneSetting] = field(default_factory=dict)
    traces: dict[str, DecisionTrace] = field(default_factory=dict)
    applied: dict[str, bool] = field(default_factory=dict)


@dataclass
class ChainStep:
    """推理链中的一步(用于实时可视化)。"""

    seat_id: str
    node: str       # featurize/recall/comfort/llm_infer/blend/safety/final
    title: str      # 中文步骤名
    detail: str     # 该步关键产物可读摘要
    data: Optional[dict] = None  # 结构化中间量(如 comfort 的计算过程 breakdown)


# 节点 → 中文步骤名
_NODE_TITLE = {
    "featurize": "① 特征提取",
    "recall": "② 记忆召回",
    "comfort": "③ 专业舒适计算 + 知识检索",
    "llm_infer": "④ LLM 专业基准",
    "approach": "⑤ 逐步逼近用户偏好",
    "safety": "⑥ 安全落定(除霜叠加)",
    "final": "✅ 最终生效",
}


def _summarize_step(node: str, update: dict) -> str:
    """把某节点的状态增量转为可读摘要,用于推理链展示。"""
    if node == "featurize":
        d = update.get("feature_detail", {})
        vec = update.get("scene_vector", [])
        keys = ["车外温度℃", "车内温度℃", "座位日照W/m²", "相对湿度%", "季节",
                "衣着", "活动", "类别"]
        brief = " · ".join(f"{k}={d[k]}" for k in keys if k in d)
        return f"输入特征:{brief}(完整 {len(d)} 项,特征向量 {len(vec)} 维)"
    if node == "recall":
        m = update.get("memory")
        if m is None or m.cluster_size == 0:
            return "无相似历史修正,本次走纯专业基准。"
        return (f"召回相似修正 {m.cluster_size} 条,逼近证据 {m.evidence_count} 次;"
                f"偏好≈{m.pref_temp:.1f}℃/{m.pref_fan}档/{m.pref_mode}。")
    if node == "comfort":
        c = update.get("comfort")
        titles = update.get("knowledge_titles", [])
        base = (f"PMV {c.pmv} / PPD {c.ppd}% / EQT {c.eqt}℃;"
                f"目标舒适温度锚点 {c.target_temp}℃。" if c else "")
        return base + (f" 知识:{', '.join(titles[:2])}" if titles else "")
    if node == "llm_infer":
        d = update.get("decision")
        src = {"llm": "LLM", "fallback": "降级兜底"}.get(update.get("source"), "")
        return (f"专业基准({src}):{d.temp_set}℃/{d.fan_level}档/{d.air_mode}。"
                if d else "")
    if node == "approach":
        nxt = update.get("approach_cursor_next")
        if not nxt:
            return "无历史偏好,保持专业基准不变。"
        return (f"朝用户偏好逐步逼近,本步 → {nxt['temp']}℃/{nxt['fan']}档"
                f"(进度 {update.get('approach_weight', 0.0):.0%})。")
    if node == "safety":
        s = update.get("setting")
        adj = update.get("trace").safety_adjustments if update.get("trace") else []
        tail = f";{adj[0]}" if adj else ";无需除霜叠加"
        return (f"落定出风模式 {s.air_mode}(风量{s.fan_level}){tail}。"
                if s else "")
    return str(update)


class Engine:
    def __init__(
        self,
        store: Optional[MemoryStore] = None,
        registry=None,
        deciders=None,
        defog_agent: Optional[DefogAgent] = None,
    ):
        self.store = store or MemoryStore()
        self.registry = registry or build_default_registry(self.store)
        self.deciders = deciders or build_decider()
        self.graph = build_graph(self.registry, self.deciders)
        # 独立"智能除雾 Agent"(车厢级,每场景判定一次)
        self.defog_agent = defog_agent or DefogAgent()
        self._defog_cache: dict[str, "DefogDecision"] = {}
        # 会话态:各座位上次实际应用的设定 / 冷静期截止时间
        self.last_applied: dict[str, ZoneSetting] = {}
        self.locked_until: dict[str, float] = {}
        # 逐步逼近游标:(user_id, seat) → 上一步推荐(跨推理持久化,实现迭代收敛)
        self.approach_cursor: dict[tuple, dict] = {}
        self.metrics = SessionMetrics()

    def defog_for(self, cabin, now: Optional[float] = None) -> "DefogDecision":
        """车厢级除雾判定(按车厢指纹缓存,使多座位/流式复用同一决策与 token)。"""
        key = cabin.model_dump_json()
        if key not in self._defog_cache:
            self._defog_cache[key] = self.defog_agent.assess(cabin, now)
        return self._defog_cache[key]

    def stream_defog(self, cabin, now: Optional[float] = None):
        """流式输出除雾 Agent 推理链(供 Web 可视化)。"""
        yield from self.defog_agent.stream(cabin, now)

    def _initial_state(self, scene: SceneInput, occ, now: float) -> dict:
        # 当前空调风量档位 = 该座位上次实际下发档位;首次用默认档
        last = self.last_applied.get(occ.seat_id)
        current_fan = last.fan_level if last is not None else config.CURRENT_FAN_DEFAULT
        return {
            "cabin": scene.cabin,
            "occupant": occ,
            "fallback_fields": [],
            "now": now,
            "current_fan": current_fan,
            "defog": self.defog_for(scene.cabin, now),  # 车厢级除雾判定(缓存复用)
            "approach_cursor": self.approach_cursor.get((occ.user_id, occ.seat_id)),
        }

    def _finalize(self, seat: str, recommended: ZoneSetting, trace: DecisionTrace,
                  now: float) -> tuple[ZoneSetting, bool]:
        """统一的防抖/冷静期判定:返回 (生效设定, 是否自动应用)。"""
        self.metrics.record_recommendation()
        last = self.last_applied.get(seat)
        if seat_locked(self.locked_until.get(seat), now):
            effective = last if last is not None else recommended
            applied = False
            trace.safety_adjustments.append("冷静期内:保持用户设定,不自动应用推荐")
        elif is_significant_change(recommended, last):
            effective = recommended
            self.last_applied[seat] = recommended
            applied = True
        else:
            effective = last
            applied = False
            trace.safety_adjustments.append("防抖:变化未超死区,维持上次设定")
        _log.info(
            "infer seat=%s source=%s applied=%s temp=%.1f fan=%d mode=%s",
            seat, trace.source, applied, effective.temp_set,
            effective.fan_level, effective.air_mode,
        )
        return effective, applied

    # ---- 推理 ----
    def infer(self, scene: SceneInput, now: Optional[float] = None) -> InferenceResult:
        now = time.time() if now is None else now
        result = InferenceResult()
        for occ in scene.present_occupants():
            seat, key = occ.seat_id, (occ.user_id, occ.seat_id)
            locked = seat_locked(self.locked_until.get(seat), now)
            state = self.graph.invoke(self._initial_state(scene, occ, now))
            effective, applied = self._finalize(
                seat, state["setting"], state["trace"], now
            )
            # 仅在非冷静期推进逼近游标(冷静期内保持用户设定,不推进逼近)
            if not locked:
                nxt = state.get("approach_cursor_next")
                if nxt is None:
                    self.approach_cursor.pop(key, None)
                else:
                    self.approach_cursor[key] = nxt
            result.settings[seat] = effective
            result.traces[seat] = state["trace"]
            result.applied[seat] = applied
        return result

    # ---- 实时推理链(流式可视化)----
    def stream_seat(self, scene: SceneInput, occ, now: Optional[float] = None,
                    apply: bool = True):
        """流式执行单座位推理图,逐节点 yield ChainStep,末尾给出结果。

        apply=False:纯可视化,不调用 _finalize(不改会话态/不计指标),复用 LLM 缓存。
        """
        now = time.time() if now is None else now
        seat = occ.seat_id
        setting: Optional[ZoneSetting] = None
        trace: Optional[DecisionTrace] = None
        for chunk in self.graph.stream(
            self._initial_state(scene, occ, now), stream_mode="updates"
        ):
            for node, update in chunk.items():
                if node == "safety":
                    setting = update.get("setting")
                    trace = update.get("trace")
                if node == "comfort":
                    data = update.get("comfort_breakdown")
                elif node == "featurize":
                    data = update.get("feature_detail")
                else:
                    data = None
                yield ChainStep(seat, node, _NODE_TITLE.get(node, node),
                                _summarize_step(node, update), data=data)
        if setting is None or trace is None:
            return
        if apply:
            effective, applied = self._finalize(seat, setting, trace, now)
            detail = (f"{effective.temp_set}℃ / {effective.fan_level}档 / "
                      f"{effective.air_mode} · "
                      f"{'已自动应用' if applied else '维持(防抖/冷静期)'}")
        else:
            detail = (f"推荐 {setting.temp_set}℃ / {setting.fan_level}档 / "
                      f"{setting.air_mode}")
        yield ChainStep(seat, "final", _NODE_TITLE["final"], detail)

    # ---- 自然语言/语音指令 ----
    def apply_command(self, scene: SceneInput, seat_id: str, text: str,
                      now: Optional[float] = None) -> ZoneSetting:
        """接收(语音转写的)文本指令 → 理解为该座位设定调整 → 作为一次修正写入记忆。

        例:文本"太冷了"→ 升温;"天太热了"→ 降温。返回调整后的设定。
        """
        from . import nlu

        now = time.time() if now is None else now
        occ = next((o for o in scene.occupants if o.seat_id == seat_id), None)
        if occ is None:
            raise ValueError(f"场景中不存在座位 {seat_id}")
        # 当前设定:优先上次实际应用,否则现推一次取该座位推荐
        current = self.last_applied.get(seat_id)
        if current is None:
            current = self.infer(scene, now=now).settings[seat_id]
        corrected = nlu.interpret(text, current, self.deciders)
        self.apply_correction(scene, seat_id, current, corrected, now=now)
        _log.info("command seat=%s text=%r -> %s℃/%d档/%s",
                  seat_id, text, corrected.temp_set, corrected.fan_level,
                  corrected.air_mode)
        return corrected

    # ---- 用户修正 ----
    def apply_correction(
        self,
        scene: SceneInput,
        seat_id: str,
        recommended: ZoneSetting,
        corrected: ZoneSetting,
        now: Optional[float] = None,
    ) -> None:
        now = time.time() if now is None else now
        occ = next(
            (o for o in scene.occupants if o.seat_id == seat_id), None
        )
        if occ is None:
            raise ValueError(f"场景中不存在座位 {seat_id}")

        vector = featurize(scene.cabin, occ).tolist()
        record = CorrectionRecord(
            user_id=occ.user_id,
            seat_id=seat_id,
            scene_vector=vector,
            recommended=recommended,
            corrected=corrected,
            season=scene.cabin.season,
            # 完整记忆链条:存入该次输入的人员/车辆/环境完整快照
            cabin=scene.cabin,
            occupant=occ,
            timestamp=now,
        )
        self.store.add(record)
        # 立即生效用户设定,并进入冷静期
        self.last_applied[seat_id] = corrected
        self.locked_until[seat_id] = lock_until(now)
        # 重置逼近游标:冷静期后从专业推荐重新逐步逼近新偏好
        self.approach_cursor.pop((occ.user_id, seat_id), None)
        self.metrics.record_correction()
        _log.info(
            "correction seat=%s user=%s %s -> %s (rate=%.2f)",
            seat_id, occ.user_id, recommended.temp_set, corrected.temp_set,
            self.metrics.correction_rate,
        )
