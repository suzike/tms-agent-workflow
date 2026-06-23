"""触发逻辑:30s 周期 + 输入显著变化事件(纯函数,便于测试)。

实时调度交由上层(CLI/Web)按需循环调用;此处只提供"是否该触发"的判定,
避免把线程/定时器耦合进核心逻辑。
"""
from __future__ import annotations

from typing import Optional

from ..config import THRESHOLDS
from ..schemas import SceneInput


def significant_change(prev: Optional[SceneInput], curr: SceneInput) -> bool:
    """相邻场景是否发生显著变化(事件触发条件)。"""
    if prev is None:
        return True
    c = THRESHOLDS
    cab_p, cab_c = prev.cabin, curr.cabin
    if cab_p.season != cab_c.season or cab_p.weather != cab_c.weather:
        return True
    if abs(cab_p.ambient_temp - cab_c.ambient_temp) > c.EVENT_TEMP_DELTA:
        return True
    # 太阳辐照显著变化(进出隧道/阴影等)
    if (abs(cab_p.sun_driver_wm2 - cab_c.sun_driver_wm2) > 200
            or abs(cab_p.sun_passenger_wm2 - cab_c.sun_passenger_wm2) > 200):
        return True

    prev_by_seat = {o.seat_id: o for o in prev.occupants}
    curr_by_seat = {o.seat_id: o for o in curr.occupants}
    if set(prev_by_seat) != set(curr_by_seat):
        return True
    for seat, oc in curr_by_seat.items():
        op = prev_by_seat[seat]
        if op.present != oc.present or op.activity != oc.activity:
            return True
    return False


def due_for_periodic(last_run_ts: Optional[float], now: float) -> bool:
    """是否到达周期触发时刻。"""
    if last_run_ts is None:
        return True
    return (now - last_run_ts) >= THRESHOLDS.PERIODIC_INTERVAL_SECONDS


def should_trigger(
    prev: Optional[SceneInput],
    curr: SceneInput,
    last_run_ts: Optional[float],
    now: float,
) -> tuple[bool, str]:
    """综合判定。返回 (是否触发, 原因)。事件优先于周期。"""
    if significant_change(prev, curr):
        return True, "event"
    if due_for_periodic(last_run_ts, now):
        return True, "periodic"
    return False, "skip"
