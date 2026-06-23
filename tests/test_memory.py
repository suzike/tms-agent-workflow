"""记忆引擎单测:逼近证据累计、矛盾保护、时间衰减、user×seat 隔离、逼近权重。"""
from tms_agent.config import THRESHOLDS
from tms_agent.memory.store import MemoryStore
from tms_agent.schemas import CorrectionRecord, ZoneSetting

VEC = [0.3, 0.36, 0.0, 0.1, 0.0, 0.0, 0.33, 0.0, 2.0, 0.0, 0.0]
NOW = 1_700_000_000.0


def _rec(user, seat, corr_temp, corr_fan=4, mode="face", ts=NOW, vec=None):
    return CorrectionRecord(
        user_id=user, seat_id=seat, scene_vector=vec or VEC,
        recommended=ZoneSetting(seat_id=seat, fan_level=2, temp_set=24.0, air_mode="face"),
        corrected=ZoneSetting(seat_id=seat, fan_level=corr_fan, temp_set=corr_temp,
                              air_mode=mode),
        season="summer", timestamp=ts,
    )


def test_evidence_accumulates(tmp_path):
    store = MemoryStore(tmp_path / "m.json")
    for _ in range(3):
        store.add(_rec("u1", "driver", corr_temp=21.0))
    r = store.recall("u1", "driver", VEC, now=NOW)
    assert r.evidence_count == 3
    assert r.has_preference is True
    assert abs(r.pref_temp - 21.0) < 0.5


def test_first_record_creates_preference(tmp_path):
    store = MemoryStore(tmp_path / "m.json")
    store.add(_rec("u1", "driver", corr_temp=21.0))
    r = store.recall("u1", "driver", VEC, now=NOW)
    assert r.evidence_count == 1 and r.has_preference is True


def test_contradictory_corrections_yield_no_evidence(tmp_path):
    store = MemoryStore(tmp_path / "m.json")
    for _ in range(2):
        store.add(_rec("u1", "driver", corr_temp=17.0))
    for _ in range(2):
        store.add(_rec("u1", "driver", corr_temp=29.0))
    r = store.recall("u1", "driver", VEC, now=NOW)
    assert r.evidence_count == 0  # 互相矛盾 → 无一致证据,逼近权重为 0


def test_time_decay_ages_out(tmp_path):
    store = MemoryStore(tmp_path / "m.json")
    old_ts = NOW - 400 * 86400
    for _ in range(3):
        store.add(_rec("u1", "driver", corr_temp=21.0, ts=old_ts))
    r = store.recall("u1", "driver", VEC, now=NOW)
    assert r.evidence_count == 0  # 旧记录衰减老化,不再计入逼近证据


def test_user_seat_isolation(tmp_path):
    store = MemoryStore(tmp_path / "m.json")
    for _ in range(3):
        store.add(_rec("u_driver", "driver", corr_temp=21.0))
    assert store.recall("u_other", "driver", VEC, now=NOW).cluster_size == 0
    assert store.recall("u_driver", "rear_left", VEC, now=NOW).cluster_size == 0


def test_dissimilar_scene_misses(tmp_path):
    store = MemoryStore(tmp_path / "m.json")
    for _ in range(3):
        store.add(_rec("u1", "driver", corr_temp=21.0))
    far_vec = [v + 5.0 for v in VEC]
    assert store.recall("u1", "driver", far_vec, now=NOW).cluster_size == 0


def test_delete_and_persist(tmp_path):
    path = tmp_path / "m.json"
    store = MemoryStore(path)
    store.add(_rec("u1", "driver", corr_temp=21.0))
    assert store.delete("u1", "driver") == 1
    assert MemoryStore(path).records_for("u1", "driver") == []
