"""Skills 单测:注册表 + 各能力独立调用返回结构化结果。"""
from tms_agent.memory.store import MemoryStore
from tms_agent.schemas import CabinContext, ComfortMetrics, OccupantState
from tms_agent.skills.registry import build_default_registry


def _ctx(season="summer", sun=1200.0, activity="calm", clothing="light"):
    return {
        "cabin": CabinContext(ambient_temp=36.0, season=season, sun_driver_wm2=sun,
                              sun_passenger_wm2=sun, humidity=55.0, soc=80.0),
        "occupant": OccupantState(seat_id="driver", activity=activity,
                                  clothing=clothing),
    }


def test_registry_registers_all(tmp_path):
    reg = build_default_registry(MemoryStore(tmp_path / "m.json"))
    assert set(reg.names()) == {
        "thermal_comfort", "knowledge", "strategy", "energy",
        "weather", "vehicle", "memory"
    }


def test_strategy_skill_tags_scene(tmp_path):
    reg = build_default_registry(MemoryStore(tmp_path / "m.json"))
    out = reg.get("strategy").invoke(_ctx())  # 夏季强日照
    assert out["strategy"] == "rapid_cooldown" and out["strategy_note"]


def test_energy_skill_advises(tmp_path):
    reg = build_default_registry(MemoryStore(tmp_path / "m.json"))
    out = reg.get("energy").invoke(_ctx())
    assert out["recirculation"] in ("inner", "outer", "auto") and out["energy_hint"]


def test_thermal_comfort_skill_returns_metrics(tmp_path):
    reg = build_default_registry(MemoryStore(tmp_path / "m.json"))
    out = reg.get("thermal_comfort").invoke(_ctx())
    m = out["comfort_metrics"]
    assert isinstance(m, ComfortMetrics)
    assert m.pmv > 0  # 38℃暴晒 → 偏热
    assert 15.5 <= m.target_temp <= 31.5


def test_knowledge_skill_returns_relevant_snippets(tmp_path):
    reg = build_default_registry(MemoryStore(tmp_path / "m.json"))
    out = reg.get("knowledge").invoke(_ctx())
    titles = out["knowledge_titles"]
    assert titles and any("暴晒" in t or "夏季" in t for t in titles)


def test_memory_skill_recall_empty(tmp_path):
    reg = build_default_registry(MemoryStore(tmp_path / "m.json"))
    ctx = {"user_id": "u1", "seat_id": "driver",
           "scene_vector": [0.0] * 11}
    out = reg.get("memory").invoke(ctx)
    assert out["memory"].cluster_size == 0
