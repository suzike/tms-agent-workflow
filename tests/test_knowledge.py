"""知识库单测:场景召回准确性 + 导入接口。"""
import json

from tms_agent.knowledge.loader import load_chunks
from tms_agent.knowledge.retriever import TfidfRetriever, build_default_retriever


def _ids(top):
    return [c.id for c, _ in top]


def test_summer_sun_retrieves_face_first():
    r = build_default_retriever()
    assert "summer_sun_face_first" in _ids(r.retrieve("夏季 暴晒 高温 进车 制冷 降温", k=3))


def test_defog_scene_retrieves_defrost():
    r = build_default_retriever()
    assert "defrost_priority" in _ids(r.retrieve("车窗起雾 除雾 湿度 雨天", k=3))


def test_winter_retrieves_feet_first():
    r = build_default_retriever()
    assert "winter_feet_first" in _ids(r.retrieve("冬季 寒冷 制热 暖脚", k=3))


def test_default_kb_nonempty():
    assert len(load_chunks()) >= 8


def test_import_interface_picks_up_new_docs(tmp_path):
    # 用户放入新的专有资料 → 无需改代码即被索引并可召回
    (tmp_path / "custom.json").write_text(
        json.dumps(
            [
                {
                    "id": "custom_rule",
                    "title": "企业专有标定",
                    "tags": ["专有", "标定"],
                    "text": "婴儿在车时避免冷风直吹,采用柔和分散出风并提高设定温度。",
                }
            ]
        ),
        encoding="utf-8",
    )
    chunks = load_chunks(tmp_path)
    assert any(c.id == "custom_rule" for c in chunks)
    top = TfidfRetriever(chunks).retrieve("婴儿 冷风 直吹", k=1)
    assert top and top[0][0].id == "custom_rule"
