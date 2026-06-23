"""智能除雾 Agent 单测:前馈(雨量/高湿)+ 反馈(玻璃温度 vs 露点)判定、知识检索、安全取严。

测试环境 LLM_PROVIDER=mock(conftest),除雾 Agent 走规则决策器,确定性。
"""
from tms_agent.defog import DefogAgent, build_defog_retriever
from tms_agent.defog.decider import RuleDefogDecider, build_defog_decider
from tms_agent.defog.tools import (
    dew_point_margin,
    defog_urgency,
    rain_fog_pressure,
)
from tms_agent.schemas import CabinContext, DefogDecision


def _cabin(**kw):
    base = dict(ambient_temp=5.0, weather="sunny", humidity=50.0, season="winter")
    base.update(kw)
    return CabinContext(**base)


# ---- 确定性工具 ----
def test_rain_pressure_levels():
    assert rain_fog_pressure("none") == 0
    assert rain_fog_pressure("light") == 1
    assert rain_fog_pressure("moderate") == 2
    assert rain_fog_pressure("heavy") == 3
    # 天气兜底:即使雨量信号缺省,weather=rain 也至少中雨级
    assert rain_fog_pressure("none", "rain") == 2


def test_dew_point_margin_sign():
    # 玻璃比露点高 → 正裕度;玻璃低于露点 → 负裕度(已结露)
    assert dew_point_margin(20.0, 22.0, 60.0) > 0
    assert dew_point_margin(2.0, 20.0, 90.0) < 0


def test_urgency_feedback_strong_when_glass_below_dewpoint():
    diag = defog_urgency(glass_temp=2.0, air_temp=20.0, humidity=90.0)
    assert diag["level"] == "strong" and diag["need_defog"] is True


def test_urgency_preemptive_small_margin():
    # 玻璃仅略高于露点(裕度小)→ 提前介入(强)
    diag = defog_urgency(glass_temp=12.0, air_temp=22.0, humidity=60.0)
    assert diag["need_defog"] is True


def test_urgency_none_when_dry_and_warm_glass():
    diag = defog_urgency(glass_temp=30.0, air_temp=24.0, humidity=40.0,
                         rain_level="none", weather="sunny")
    assert diag["level"] == "none" and diag["need_defog"] is False


def test_urgency_feedforward_heavy_rain():
    # 大雨即使玻璃温度未接入,也判定需要除雾
    diag = defog_urgency(glass_temp=None, air_temp=None, humidity=85.0,
                         rain_level="heavy", weather="rain")
    assert diag["need_defog"] is True and diag["level"] == "strong"


def test_urgency_compounding_rain_and_margin():
    # 中雨 + 玻璃裕度小 → 叠加升级为强
    diag = defog_urgency(glass_temp=11.0, air_temp=15.0, humidity=92.0,
                         rain_level="moderate", weather="rain")
    assert diag["level"] == "strong"


# ---- 规则决策器 ----
def test_rule_decider_outputs_decision():
    dec = RuleDefogDecider().decide(
        {"windshield_glass_temp": 2.0, "windshield_air_temp": 18.0,
         "humidity": 90.0, "rain_level": "none", "weather": "cloudy"}
    )
    assert isinstance(dec, DefogDecision)
    assert dec.need_defog is True and dec.source == "rule"
    assert "margin" in dec.diagnostics


# ---- Agent 端到端 ----
def test_agent_assess_high_humidity_rain():
    agent = DefogAgent()
    dec = agent.assess(_cabin(weather="rain", humidity=93.0, rain_level="moderate"))
    assert dec.need_defog is True


def test_agent_assess_dry_summer_no_defog():
    agent = DefogAgent()
    dec = agent.assess(CabinContext(ambient_temp=35.0, weather="sunny",
                                    humidity=45.0, season="summer"))
    assert dec.need_defog is False and dec.level == "none"


def test_agent_stream_yields_chain():
    agent = DefogAgent()
    steps = list(agent.stream(_cabin(weather="rain", humidity=93.0,
                                     rain_level="heavy")))
    nodes = [s.node for s in steps]
    assert nodes == ["sense", "knowledge", "decide", "final"]
    assert all(s.detail for s in steps)


def test_defog_retriever_hits_defog_kb():
    r = build_defog_retriever()
    hits = r.retrieve("玻璃温度 露点 裕度 起雾", 3)
    assert hits  # 除雾专属知识库可检索
    titles = [c.title for c, _ in hits]
    assert any("露点" in t or "裕度" in t or "前馈" in t for t in titles)


def test_build_defog_decider_returns_rule_under_mock():
    primary, fallback = build_defog_decider()
    assert isinstance(primary, RuleDefogDecider)
    assert isinstance(fallback, RuleDefogDecider)
