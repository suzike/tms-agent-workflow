"""自然语言/语音指令单测:规则解析 + 端到端 apply_command。"""
from tms_agent.engine import Engine
from tms_agent.memory.store import MemoryStore
from tms_agent.nlu import rule_interpret
from tms_agent.schemas import CabinContext, OccupantState, SceneInput, ZoneSetting

CUR = ZoneSetting(seat_id="driver", fan_level=4, temp_set=24.0, air_mode="face")


def test_too_cold_raises_temp():
    assert rule_interpret("太冷了", CUR).temp_set > CUR.temp_set


def test_too_hot_lowers_temp():
    assert rule_interpret("天太热了", CUR).temp_set < CUR.temp_set


def test_wind_too_strong_lowers_fan():
    assert rule_interpret("风太大", CUR).fan_level < CUR.fan_level


def test_explicit_temperature():
    assert rule_interpret("帮我调到22度", CUR).temp_set == 22.0


def test_defog_command_sets_defrost():
    assert rule_interpret("玻璃起雾了", CUR).air_mode == "defrost"


def _scene():
    return SceneInput(
        cabin=CabinContext(ambient_temp=10, cabin_temp=12, season="winter",
                           sun_driver_wm2=0, sun_passenger_wm2=0, timestamp=1_700_000_000.0),
        occupants=[OccupantState(seat_id="driver", user_id="alice", clothing="medium")],
    )


def test_apply_command_records_memory(tmp_path):
    eng = Engine(store=MemoryStore(tmp_path / "m.json"))
    scene = _scene()
    corrected = eng.apply_command(scene, "driver", "太冷了", now=1_700_000_000.0)
    assert isinstance(corrected, ZoneSetting)
    assert eng.store.summary().get(("alice", "driver"), 0) >= 1  # 已写入记忆
