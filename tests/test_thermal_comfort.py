"""热舒适计算单测:对照 ISO 7730 文献值 + 物理单调性。"""
from tms_agent.tools.thermal_comfort import (
    air_velocity_from_fan,
    approach_step,
    clo_from_clothing,
    comfort_offset,
    compute_pmv_ppd,
    equivalent_temperature,
    estimate_cabin_temp,
    met_from_activity,
    mrt_from_sun,
    target_comfort_temp,
)


def test_approach_step_fan_sequence():
    # 风量 7 → 偏好 3:逐步逼近 7→5→4→3(差值越大首步越大,2-3 步到位)
    seq, cur = [], 7
    for _ in range(5):
        cur = int(approach_step(cur, 3, 1.0, lo=1, hi=7))
        seq.append(cur)
        if cur == 3:
            break
    assert seq == [5, 4, 3]


def test_approach_step_monotonic_no_overshoot():
    # 升温方向同样单调收敛且不超调
    cur = 18.0
    seen = []
    for _ in range(10):
        cur = approach_step(cur, 24.0, 0.5, lo=15.5, hi=31.5)
        seen.append(cur)
        if cur == 24.0:
            break
    assert seen == sorted(seen) and seen[-1] == 24.0 and max(seen) <= 24.0


def test_estimate_cabin_temp_rises_with_sun():
    cold = estimate_cabin_temp(30.0, 0.0)
    hot = estimate_cabin_temp(30.0, 1000.0)
    assert cold == 30.0 and hot > cold        # 日照升温
    # 开窗通风 → 拉回接近车外
    vented = estimate_cabin_temp(30.0, 1000.0, window_open_pct=100.0)
    assert vented < hot


def test_clo_from_clothing_ordering():
    assert clo_from_clothing("light") < clo_from_clothing("medium") < clo_from_clothing("heavy")


def test_fog_risk_and_dew_point():
    from tms_agent.tools.thermal_comfort import dew_point, fog_risk
    assert dew_point(25.0, 90.0) > dew_point(25.0, 30.0)   # 湿度高露点高
    assert fog_risk(20.0, 2.0, 95.0) == "high"             # 车内暖湿+车外冷 → 高风险
    assert fog_risk(24.0, 30.0, 40.0) == "low"             # 干爽 → 低风险


def test_comfort_temp_band_brackets_target():
    from tms_agent.tools.thermal_comfort import comfort_temp_band
    lo, hi = comfort_temp_band(0.0, 50.0, "summer")
    assert 15.5 <= lo <= hi <= 31.5 and (hi - lo) >= 0.5


def test_transient_setpoint_fan_cooldown_then_steady():
    from tms_agent.tools.thermal_comfort import transient_setpoint_fan
    sp_hot, fan_hot = transient_setpoint_fan(40.0, 24.0)     # 大负荷
    sp_steady, fan_steady = transient_setpoint_fan(24.0, 24.0)  # 稳态
    assert sp_hot < sp_steady          # 热时设定更激进(更低)以快速降温
    assert fan_hot > fan_steady        # 热时风量更大
    # 稳态回归目标;风量落脚点为 2 档(常规舒适/NVH 基准,不取过小的 1 档)
    assert sp_steady == 24.0 and fan_steady == 2


def test_met_from_activity_ordering():
    assert met_from_activity("sleeping") < met_from_activity("calm") < met_from_activity("excited")


def test_comfort_offset_elderly_child_warmer():
    assert comfort_offset("elderly", "male", 24.0) > 0     # 老人偏暖
    assert comfort_offset("child", "male", 24.0) > 0       # 小孩偏暖
    assert comfort_offset("adult", "male", 24.0) == 0.0    # 标准成年男性无偏移
    assert comfort_offset("adult", "female", 22.0) > 0     # 女性偏暖
    assert comfort_offset("adult", "male", 30.0) < 0       # 高 BMI 偏凉


def test_pmv_matches_iso_reference_cool():
    # ISO 7730 验证点:文献 PMV≈-0.75, PPD≈17
    pmv, ppd = compute_pmv_ppd(22, 22, 0.1, 60, met=1.2, clo=0.5)
    assert abs(pmv - (-0.75)) < 0.05
    assert abs(ppd - 17) < 1.5


def test_pmv_matches_iso_reference_warm():
    # ISO 7730 验证点:文献 PMV≈+0.77, PPD≈17
    pmv, ppd = compute_pmv_ppd(27, 27, 0.1, 60, met=1.2, clo=0.5)
    assert abs(pmv - 0.77) < 0.05
    assert abs(ppd - 17) < 1.5


def test_ppd_minimized_at_neutral():
    _, ppd = compute_pmv_ppd(24.5, 24.5, 0.1, 50, met=1.2, clo=0.5)
    assert ppd < 6  # 中性点 PPD 接近理论下限 5%


def test_pmv_monotonic_in_temp():
    cool = compute_pmv_ppd(20, 20, 0.1, 50, met=1.2, clo=0.5)[0]
    warm = compute_pmv_ppd(30, 30, 0.1, 50, met=1.2, clo=0.5)[0]
    assert cool < 0 < warm


def test_target_temp_lower_under_strong_sun():
    no_sun = target_comfort_temp(0.0, 50, "summer")        # 0 W/m²
    strong = target_comfort_temp(1350.0, 50, "summer")     # 强暴晒 W/m²
    assert strong < no_sun  # 暴晒需更低设定温度补偿辐射
    assert 15.5 <= strong <= 31.5 and 15.5 <= no_sun <= 31.5


def test_winter_target_in_range():
    t = target_comfort_temp(0.0, 40, "winter")
    assert 20 <= t <= 26  # 冬季着装更厚,中性温度偏低且合理


def test_helpers_monotonic():
    assert air_velocity_from_fan(1) < air_velocity_from_fan(7)
    assert mrt_from_sun(30, 0.0) < mrt_from_sun(30, 1500.0)
    # 气流增大 → 当量温度下降(夏季制冷感更强)
    assert equivalent_temperature(30, 30, 1.0) < equivalent_temperature(30, 30, 0.1)
