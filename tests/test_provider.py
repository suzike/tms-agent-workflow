"""LLM provider 单测:Mock 规则引擎只产出舒适基础模式、合法范围、降级工厂。"""
from tms_agent.config import BASE_AIR_MODES, FAN_MAX, FAN_MIN, TEMP_MAX, TEMP_MIN
from tms_agent.llm.provider import MockDecider, build_decider
from tms_agent.schemas import HVACDecision


def _payload(**kw):
    base = {
        "seat_id": "driver", "season": "summer", "weather": "sunny",
        "humidity": 55.0, "soc": 80.0, "seat_sun_wm2": 1200.0,
        "person_category": "adult", "gender": "male", "clothing": "light",
        "activity": "calm",
        "comfort": {"pmv": 1.5, "ppd": 50, "eqt": 40, "target_temp": 23.0,
                    "cabin_temp": 42.0},
        "knowledge": [],
    }
    base.update(kw)
    return base


def test_mock_returns_legal_base_decision():
    d = MockDecider().decide(_payload())
    assert isinstance(d, HVACDecision)
    assert FAN_MIN <= d.fan_level <= FAN_MAX
    assert TEMP_MIN <= d.temp_set <= TEMP_MAX
    assert d.air_mode in BASE_AIR_MODES  # 只产出 3 个舒适基础模式,绝不含除霜


def test_temp_on_half_degree_grid():
    d = MockDecider().decide(_payload(comfort={"target_temp": 23.0}))
    assert (d.temp_set * 2) % 1 == 0  # 0.5℃ 精度


def test_hot_summer_picks_face_high_fan():
    d = MockDecider().decide(_payload())
    assert d.air_mode == "face"
    assert d.fan_level >= 4


def test_winter_cold_picks_feet():
    d = MockDecider().decide(_payload(
        season="winter", seat_sun_wm2=0.0,
        comfort={"target_temp": 23.0, "pmv": -1.0, "cabin_temp": 2.0}))
    assert d.air_mode == "feet"


def test_transition_neutral_picks_face_feet():
    d = MockDecider().decide(_payload(
        season="transition", seat_sun_wm2=200.0,
        comfort={"target_temp": 23.0, "pmv": 0.0, "cabin_temp": 24.0}))
    assert d.air_mode == "face_feet"


def test_build_decider_falls_back_to_mock_without_key():
    primary, fallback = build_decider()
    assert isinstance(fallback, MockDecider)
    assert isinstance(primary.decide(_payload()), HVACDecision)
