"""触发逻辑单测:事件检测 + 周期 + 综合判定。"""
from tms_agent.config import THRESHOLDS
from tms_agent.runtime.triggers import (
    due_for_periodic,
    should_trigger,
    significant_change,
)
from tms_agent.schemas import CabinContext, OccupantState, SceneInput


def _scene(ambient=30.0, season="summer", activity="calm", present=True):
    return SceneInput(
        cabin=CabinContext(ambient_temp=ambient, season=season, timestamp=1700000000.0),
        occupants=[OccupantState(seat_id="driver", activity=activity, present=present)],
    )


def test_first_scene_triggers():
    assert significant_change(None, _scene()) is True


def test_small_change_not_significant():
    assert significant_change(_scene(ambient=30.0), _scene(ambient=31.0)) is False


def test_large_temp_change_significant():
    assert significant_change(_scene(ambient=30.0), _scene(ambient=34.0)) is True


def test_activity_or_season_change_significant():
    assert significant_change(_scene(activity="calm"), _scene(activity="sleeping")) is True
    assert significant_change(_scene(season="summer"), _scene(season="winter")) is True


def test_periodic_due():
    assert due_for_periodic(None, 100.0) is True
    interval = THRESHOLDS.PERIODIC_INTERVAL_SECONDS
    assert due_for_periodic(100.0, 100.0 + interval) is True
    assert due_for_periodic(100.0, 100.0 + interval - 1) is False


def test_should_trigger_priority():
    # 事件优先
    fired, reason = should_trigger(_scene(ambient=30), _scene(ambient=40), 0.0, 1.0)
    assert fired and reason == "event"
    # 无事件但到周期
    fired, reason = should_trigger(_scene(ambient=30), _scene(ambient=30), 0.0,
                                   THRESHOLDS.PERIODIC_INTERVAL_SECONDS + 1)
    assert fired and reason == "periodic"
    # 无事件未到周期
    fired, reason = should_trigger(_scene(ambient=30), _scene(ambient=30), 0.0, 1.0)
    assert not fired and reason == "skip"
