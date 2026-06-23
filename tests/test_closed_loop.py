"""端到端闭环回归:持续逼近学习 + 多温区独立 + 专业约束 + 鲁棒性(Mock 决策器确定性)。"""
import json

import pytest

from tms_agent.config import PROJECT_ROOT, THRESHOLDS
from tms_agent.engine import Engine
from tms_agent.llm.provider import MockDecider
from tms_agent.memory.store import MemoryStore
from tms_agent.schemas import (
    CabinContext,
    OccupantState,
    SceneInput,
    ZoneSetting,
    sanitize_scene,
)

T = 1_700_000_000.0
PREF = ZoneSetting(seat_id="driver", fan_level=6, temp_set=18.5, air_mode="face_feet")


def _scene(driver_user="alice", season="summer", sun_driver=1200.0, sun_pass=150.0):
    # 主驾侧强日照→偏热;副驾侧弱日照,后排取副驾侧→偏凉,体现多温区差异
    return SceneInput(
        cabin=CabinContext(ambient_temp=36, season=season, sun_driver_wm2=sun_driver,
                           sun_passenger_wm2=sun_pass, humidity=55, soc=75, timestamp=T),
        occupants=[
            OccupantState(seat_id="driver", user_id=driver_user, clothing="light"),
            OccupantState(seat_id="rear_right", user_id="bob", clothing="light"),
        ],
    )


def _teach(engine, scene, times, now0=T):
    for i in range(times):
        r = engine.infer(scene, now=now0 + i)
        engine.apply_correction(scene, "driver", r.settings["driver"], PREF,
                                now=now0 + i)


def _measure(path, scene):
    """用全新引擎(无锁/无防抖态)读取共享记忆,反映图的纯推荐。"""
    return Engine(store=MemoryStore(path)).infer(scene, now=T + 10_000)


# 0) 实时推理链:流式输出节点序列,末步给出最终生效
def test_stream_seat_yields_chain(tmp_path):
    eng = Engine(store=MemoryStore(tmp_path / "m.json"))
    scene = _scene()
    steps = list(eng.stream_seat(scene, scene.occupants[0], now=T))
    nodes = [s.node for s in steps]
    assert nodes == ["featurize", "recall", "comfort", "llm_infer", "approach",
                     "safety", "final"]
    assert all(s.detail for s in steps)          # 每步都有可读摘要
    assert steps[-1].node == "final"


# 1) 多乘员独立预测
def test_multi_occupant_independent(tmp_path):
    res = Engine(store=MemoryStore(tmp_path / "m.json")).infer(_scene(), now=T)
    assert set(res.settings) == {"driver", "rear_right"}
    assert res.settings["driver"].temp_set < res.settings["rear_right"].temp_set


# 2) 逐步逼近:一次修正后,冷静期外的后续推理沿专业推荐迭代收敛到偏好(非一次到位)
def test_gradual_approach_iterates(tmp_path):
    eng = Engine(store=MemoryStore(tmp_path / "m.json"))
    scene = _scene()  # 夏季强日照 → 专业基准风量高(约 7 档)
    r0 = eng.infer(scene, now=T)
    base_fan = r0.settings["driver"].fan_level
    target = ZoneSetting(seat_id="driver", fan_level=3, temp_set=18.5,
                         air_mode="face_feet")
    eng.apply_correction(scene, "driver", r0.settings["driver"], target, now=T)

    # 冷静期之后,逐次推理观察风量逐步逼近 3 档
    t0 = T + THRESHOLDS.LOCK_WINDOW_SECONDS + 10
    fans = [eng.infer(scene, now=t0 + i).settings["driver"].fan_level
            for i in range(6)]
    assert base_fan > 3                       # 基准远高于目标
    assert fans[0] > 3                         # 不是一次到位
    assert fans[-1] == 3                       # 最终收敛到偏好
    assert fans == sorted(fans, reverse=True)  # 单调递减
    assert len(set(fans)) >= 2                 # 确有逐步过程(多步)


# 2b) 后排独立:主驾的偏好不影响后排
def test_other_seat_unaffected_by_preference(tmp_path):
    path = tmp_path / "m.json"
    _teach(Engine(store=MemoryStore(path)), _scene(), 3)
    res = _measure(path, _scene())
    assert res.traces["rear_right"].memory_evidence == 0


# 3) 不相似场景 / 不同季节 不受影响
def test_dissimilar_or_other_season_unaffected(tmp_path):
    path = tmp_path / "m.json"
    _teach(Engine(store=MemoryStore(path)), _scene(), 6)
    res = _measure(path, _scene(season="winter", sun_driver=0.0))
    assert res.traces["driver"].memory_evidence == 0
    assert res.traces["driver"].approach_weight == 0.0


# 4) 多用户隔离
def test_multi_user_isolation(tmp_path):
    path = tmp_path / "m.json"
    _teach(Engine(store=MemoryStore(path)), _scene(driver_user="alice"), 6)
    res = _measure(path, _scene(driver_user="dave"))
    assert res.traces["driver"].memory_evidence == 0


# 5) 专业约束:舒适数值与知识注入,温度落在合理带
def test_professional_grounding(tmp_path):
    res = Engine(store=MemoryStore(tmp_path / "m.json")).infer(_scene(), now=T)
    td = res.traces["driver"]
    assert td.comfort_metrics is not None and td.knowledge_snippets
    # 瞬态控制下设定可在目标±瞬态偏置(≤4℃)内,接近稳态回归目标
    assert abs(res.settings["driver"].temp_set - td.comfort_metrics.target_temp) <= 4.5


# 5b) 瞬态控制:负荷越大设定越激进+风量越大,趋近稳态回归舒适+降风量
def test_transient_setpoint_fan_direction(tmp_path):
    def _drive(cabin_temp, season, i):
        eng = Engine(store=MemoryStore(tmp_path / f"m{season}{i}.json"))
        sc = SceneInput(
            cabin=CabinContext(ambient_temp=36 if season == "summer" else -5,
                               cabin_temp=cabin_temp, season=season,
                               sun_driver_wm2=1000 if season == "summer" else 100,
                               humidity=50, timestamp=T),
            occupants=[OccupantState(seat_id="driver", clothing="light")])
        return eng.infer(sc, now=T).settings["driver"]

    # 夏季:车内更热 → 设定更低、风量不更小
    hot, cool = _drive(48, "summer", 0), _drive(27, "summer", 1)
    assert hot.temp_set < cool.temp_set
    assert hot.fan_level >= cool.fan_level
    # 冬季:车内更冷 → 设定更高、风量不更小
    cold, warm = _drive(-5, "winter", 2), _drive(18, "winter", 3)
    assert cold.temp_set > warm.temp_set
    assert cold.fan_level >= warm.fan_level


# 6) 鲁棒性:云端 LLM 失败 → 降级链产出合法设定
def test_llm_failure_degrades_gracefully(tmp_path):
    class FailingDecider:
        def decide(self, payload):
            raise RuntimeError("network down")

    eng = Engine(store=MemoryStore(tmp_path / "m.json"),
                 deciders=(FailingDecider(), MockDecider()))
    res = eng.infer(_scene(), now=T)
    s = res.settings["driver"]
    assert res.traces["driver"].source == "fallback"
    assert 15.5 <= s.temp_set <= 31.5 and 1 <= s.fan_level <= 7


# 7) 鲁棒性:场景稳定时反复推理会收敛,收敛后防抖抑制重复应用
def test_deadband_suppresses_after_convergence(tmp_path):
    eng = Engine(store=MemoryStore(tmp_path / "m.json"))
    scene = _scene()
    flags = [eng.infer(scene, now=T + i * 5).applied["driver"] for i in range(4)]
    assert flags[0] is True       # 首次必应用
    assert flags[-1] is False     # 收敛后防抖维持,不再重复应用


# 8) 脏输入兜底
def test_sanitize_handles_dirty_input():
    raw = {
        "cabin": {"ambient_temp": float("nan"), "season": "summer"},
        "occupants": [{"seat_id": "driver", "local_cabin_temp": None}],
    }
    scene, fallbacks = sanitize_scene(raw)
    assert scene.cabin.ambient_temp == 30.0          # 夏季默认
    assert "cabin.ambient_temp" in fallbacks


# 9) 典型工况回归:专业出风模式与温度方向正确
@pytest.mark.parametrize("case", json.loads(
    (PROJECT_ROOT / "data" / "scenario_set.json").read_text(encoding="utf-8")
))
def test_scenario_set_regression(tmp_path, case):
    scene, _ = sanitize_scene(case["scene"])
    res = Engine(store=MemoryStore(tmp_path / "m.json")).infer(scene, now=T)
    exp = case["expect"]
    s = res.settings[exp["seat_id"]]
    if "air_mode" in exp:
        assert s.air_mode == exp["air_mode"], case["name"]
    if "fan_is" in exp:
        assert s.fan_level == exp["fan_is"], case["name"]
    if "temp_below" in exp:
        assert s.temp_set < exp["temp_below"], case["name"]
    if "temp_above" in exp:
        assert s.temp_set > exp["temp_above"], case["name"]
    if "temp_between" in exp:
        lo, hi = exp["temp_between"]
        assert lo <= s.temp_set <= hi, case["name"]
