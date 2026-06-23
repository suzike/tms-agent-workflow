"""安全护栏、防抖迟滞、手动冷静期(纯函数,供 graph/engine 使用)。

- 边界:HVACDecision 已由 Pydantic 限定范围;此处额外强制"除雾安全优先"。
- 起雾判定已独立为"智能除雾 Agent"(见 ``defog/``);本层只消费其 DefogDecision 叠加除霜。
- 防抖:变化幅度不超过死区则不更新该座位,避免周期性抖动。
- 冷静期:用户手动覆盖某座位后,该座位在锁定窗口内只记录不自动改。
"""
from __future__ import annotations

from typing import Optional

from .config import DEFOG, FAN_MAX
from .config import THRESHOLDS
from .schemas import (
    CabinContext,
    DefogDecision,
    HVACDecision,
    OccupantState,
    ZoneSetting,
)


def apply_safety(
    decision: HVACDecision,
    cabin: CabinContext,
    occupant: OccupantState,
    defog: Optional[DefogDecision] = None,
) -> tuple[ZoneSetting, list[str]]:
    """把(舒适基础)决策落为该座位完整 ZoneSetting,叠加除雾 Agent 的除霜需求。

    decision.air_mode 为 3 个基础模式之一。除雾职责已交由独立"智能除雾 Agent",
    本层仅消费其 ``DefogDecision``:
    - 纯除霜 ``defrost``:仅当显式开启"最大除霜功能"(cabin.max_defrost)时进入,强制最大除风档。
    - 除霜叠加:除雾 Agent 判定 need_defog 时,基础模式叠加除霜(face_defrost 等);
      strong 级保证最低风量下限,确保气流冲刷玻璃。
    """
    adjustments: list[str] = []
    base_mode = decision.air_mode  # face / face_feet / feet
    fan = decision.fan_level

    if cabin.max_defrost:
        air_mode = "defrost"
        fan = FAN_MAX  # 纯除霜 = 最大除风档
        adjustments.append("最大除霜功能已开启:纯除霜模式 + 最大除风档")
    elif defog is not None and defog.need_defog:
        air_mode = f"{base_mode}_defrost"
        note = (f"除雾 Agent:需要除雾({defog.level})→ {base_mode} 叠加除霜"
                f"→{air_mode}")
        if defog.reasoning:
            note += f";{defog.reasoning}"
        if defog.level == "strong" and fan < DEFOG.fan_floor_strong:
            note += f";强除雾保证风量≥{DEFOG.fan_floor_strong}(原{fan})"
            fan = DEFOG.fan_floor_strong
        adjustments.append(note)
    else:
        air_mode = base_mode

    setting = ZoneSetting(
        seat_id=occupant.seat_id,
        fan_level=fan,
        temp_set=decision.temp_set,
        air_mode=air_mode,
        reasoning=decision.reasoning,
    )
    return setting, adjustments


def is_significant_change(new: ZoneSetting, last: Optional[ZoneSetting]) -> bool:
    """防抖:相对上次已应用设定,变化是否超过死区(超过才值得调整)。"""
    if last is None:
        return True
    th = THRESHOLDS
    return (
        abs(new.temp_set - last.temp_set) > th.TEMP_DEADBAND
        or abs(new.fan_level - last.fan_level) > th.FAN_DEADBAND
        or new.air_mode != last.air_mode
    )


def seat_locked(locked_until: Optional[float], now: float) -> bool:
    """该座位是否处于手动冷静期内。"""
    return locked_until is not None and now < locked_until


def lock_until(now: float) -> float:
    """计算冷静期结束时间戳。"""
    return now + THRESHOLDS.LOCK_WINDOW_SECONDS
