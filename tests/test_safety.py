"""安全/防抖/冷静期单测:除霜叠加(消费除雾 Agent 决策)、最大档纯除霜、死区、锁定。

起雾判定本身已独立到"智能除雾 Agent",见 tests/test_defog.py;
本文件只验证安全层如何"消费 DefogDecision"落定出风模式与风量。
"""
from tms_agent.config import DEFOG, FAN_MAX, THRESHOLDS
from tms_agent.safety import (
    apply_safety,
    is_significant_change,
    lock_until,
    seat_locked,
)
from tms_agent.schemas import (
    CabinContext,
    DefogDecision,
    HVACDecision,
    OccupantState,
    ZoneSetting,
)


def _cabin(**kw):
    base = dict(ambient_temp=5.0, weather="rain", humidity=88.0, season="winter")
    base.update(kw)
    return CabinContext(**base)


def _occ():
    return OccupantState(seat_id="driver")


def _decision(mode="feet", fan=3):
    return HVACDecision(fan_level=fan, temp_set=24.0, air_mode=mode)


def _defog(need=True, level="mild"):
    return DefogDecision(need_defog=need, level=level, source="rule",
                         reasoning="测试")


def test_overlay_stacks_defrost_on_base_mode():
    # 除雾 Agent 判定需要除雾:基础 feet → feet_defrost(叠加,而非纯除霜)
    setting, adj = apply_safety(_decision("feet"), _cabin(), _occ(), _defog())
    assert setting.air_mode == "feet_defrost"
    assert adj


def test_face_overlay():
    setting, _ = apply_safety(_decision("face"), _cabin(), _occ(), _defog())
    assert setting.air_mode == "face_defrost"


def test_strong_defog_enforces_fan_floor():
    # 强除雾:保证风量下限,确保气流冲刷玻璃(安全优先于 NVH)
    setting, adj = apply_safety(
        _decision("feet", fan=2), _cabin(), _occ(), _defog(level="strong")
    )
    assert setting.air_mode == "feet_defrost"
    assert setting.fan_level == DEFOG.fan_floor_strong
    assert any("强除雾" in a for a in adj)


def test_mild_defog_keeps_fan():
    # 轻度除雾:不强制提升风量
    setting, _ = apply_safety(
        _decision("feet", fan=2), _cabin(), _occ(), _defog(level="mild")
    )
    assert setting.fan_level == 2


def test_pure_defrost_only_when_max_defrost_on():
    # 纯除霜仅在显式开启"最大除霜功能"时进入,且强制最大除风档(优先于除雾叠加)
    setting, adj = apply_safety(
        _decision("feet", fan=2), _cabin(max_defrost=True), _occ(), _defog()
    )
    assert setting.air_mode == "defrost"
    assert setting.fan_level == FAN_MAX
    assert adj


def test_no_defog_keeps_base_mode():
    setting, adj = apply_safety(
        _decision("face"), _cabin(weather="sunny", humidity=40, season="summer"),
        _occ(), _defog(need=False, level="none"),
    )
    assert setting.air_mode == "face"
    assert adj == []


def test_defog_none_keeps_base_mode():
    # 未传入除雾决策(defog=None)→ 不叠加除霜
    setting, adj = apply_safety(_decision("face"), _cabin(), _occ())
    assert setting.air_mode == "face"
    assert adj == []


def test_deadband_suppresses_tiny_change():
    last = ZoneSetting(seat_id="driver", fan_level=3, temp_set=24.0, air_mode="face")
    tiny = ZoneSetting(seat_id="driver", fan_level=3, temp_set=24.4, air_mode="face")
    assert is_significant_change(tiny, last) is False


def test_deadband_allows_real_change():
    last = ZoneSetting(seat_id="driver", fan_level=3, temp_set=24.0, air_mode="face")
    big = ZoneSetting(seat_id="driver", fan_level=3, temp_set=25.5, air_mode="face")
    mode = ZoneSetting(seat_id="driver", fan_level=3, temp_set=24.0, air_mode="feet")
    assert is_significant_change(big, last) is True
    assert is_significant_change(mode, last) is True
    assert is_significant_change(last, None) is True


def test_cooldown_lock():
    now = 1000.0
    until = lock_until(now)
    assert until == now + THRESHOLDS.LOCK_WINDOW_SECONDS
    assert seat_locked(until, now + 1) is True
    assert seat_locked(until, until + 1) is False
    assert seat_locked(None, now) is False
