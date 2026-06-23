"""记忆引擎:按 (user_id × seat_id) 独立存储用户修正,渐进置信学习。

核心机制:
- 召回:同键下,与当前场景"相似"(加权距离<阈值)的历史修正聚成簇。
- 渐进置信:对簇内"方向一致"的修正做时间衰减加权计数,达到 K 才高置信自动套用。
- 矛盾保护:相互矛盾的修正无法形成一致簇,置信不上升,避免被单次误操作带偏。
- 时效:旧修正按半衰期衰减权重,自然老化。
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .. import config
from ..features import distance
from ..schemas import CorrectionRecord


@dataclass(frozen=True)
class MemoryRecall:
    """一次召回结果(不可变)。"""

    user_id: str
    seat_id: str
    cluster_size: int          # 相似历史修正条数
    evidence_count: int        # 方向一致且未老化的修正条数(逼近证据)
    pref_temp: Optional[float] = None     # 学到的偏好代表值(时间加权)
    pref_fan: Optional[int] = None
    pref_mode: Optional[str] = None       # 完整模式代表值(可能含除霜)

    @property
    def has_preference(self) -> bool:
        return self.evidence_count > 0


class MemoryStore:
    """JSON 持久化的记忆库。接口稳定,后续可替换 SQLite/向量库。"""

    def __init__(self, path: Optional[Path] = None):
        self._path = Path(path) if path else config.MEMORY_FILE
        self._records: list[CorrectionRecord] = []
        self._load()

    # ---- 持久化 ----
    def _load(self) -> None:
        if self._path.exists():
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._records = [CorrectionRecord(**r) for r in raw]

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.model_dump() for r in self._records]
        self._path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ---- 写入 ----
    def add(self, record: CorrectionRecord) -> None:
        self._records = [*self._records, record]  # 不可变追加
        self._save()

    # ---- 查询 / 管理 ----
    def records_for(self, user_id: str, seat_id: str) -> list[CorrectionRecord]:
        return [
            r for r in self._records if r.user_id == user_id and r.seat_id == seat_id
        ]

    def delete(self, user_id: str, seat_id: str) -> int:
        before = len(self._records)
        self._records = [
            r
            for r in self._records
            if not (r.user_id == user_id and r.seat_id == seat_id)
        ]
        self._save()
        return before - len(self._records)

    def reset_all(self) -> None:
        self._records = []
        self._save()

    def summary(self) -> dict[tuple[str, str], int]:
        """各 (user×seat) 的记忆条数,供 UI 展示。"""
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for r in self._records:
            counts[(r.user_id, r.seat_id)] += 1
        return dict(counts)

    # ---- 召回(核心) ----
    def _decay_weight(self, record_ts: float, now: float) -> float:
        age_days = max(0.0, (now - record_ts) / 86400.0)
        return 0.5 ** (age_days / config.THRESHOLDS.MEMORY_HALFLIFE_DAYS)

    def recall(
        self,
        user_id: str,
        seat_id: str,
        scene_vector: list[float] | np.ndarray,
        now: Optional[float] = None,
    ) -> MemoryRecall:
        now = time.time() if now is None else now
        th = config.THRESHOLDS
        vec = np.asarray(scene_vector, dtype=float)

        cluster = [
            r
            for r in self.records_for(user_id, seat_id)
            if distance(vec, np.asarray(r.scene_vector)) < th.SIM_THRESHOLD
        ]
        if not cluster:
            return MemoryRecall(user_id, seat_id, 0, 0)

        weights = np.array([self._decay_weight(r.timestamp, now) for r in cluster])
        temps = np.array([r.corrected.temp_set for r in cluster])
        fans = np.array([r.corrected.fan_level for r in cluster])
        wsum = float(weights.sum())

        pref_temp = float(np.dot(weights, temps) / wsum)
        pref_fan = int(round(float(np.dot(weights, fans) / wsum)))

        mode_weight: dict[str, float] = defaultdict(float)
        for r, w in zip(cluster, weights):
            mode_weight[r.corrected.air_mode] += w
        pref_mode = max(mode_weight, key=mode_weight.get)

        # 逼近证据:统计落在偏好代表值容差内、且未老化(权重达门槛)的修正条数。
        # 时间衰减通过 active_floor 实现旧记录的自然老化。证据越多 → 逼近权重越大。
        evidence = 0
        for r, w in zip(cluster, weights):
            if (
                w >= th.MEMORY_ACTIVE_FLOOR
                and abs(r.corrected.temp_set - pref_temp) <= th.TEMP_DEADBAND
                and abs(r.corrected.fan_level - pref_fan) <= th.FAN_DEADBAND
                and r.corrected.air_mode == pref_mode
            ):
                evidence += 1

        return MemoryRecall(
            user_id=user_id,
            seat_id=seat_id,
            cluster_size=len(cluster),
            evidence_count=evidence,
            pref_temp=pref_temp,
            pref_fan=pref_fan,
            pref_mode=pref_mode,
        )
