"""特征工程单测:相似性、季节隔离、活动/人员/衣着区分。"""
from tms_agent.config import THRESHOLDS
from tms_agent.features import distance, featurize
from tms_agent.schemas import CabinContext, OccupantState

TS = 1_700_000_000.0


def _cabin(ambient=30.0, season="summer", sun=800.0, **kw):
    return CabinContext(ambient_temp=ambient, season=season, sun_driver_wm2=sun,
                        sun_passenger_wm2=sun, timestamp=TS, **kw)


def _occ(activity="calm", clothing="medium", age=35, seat="driver"):
    return OccupantState(seat_id=seat, activity=activity, clothing=clothing, age=age)


def test_identical_scene_zero_distance():
    v = featurize(_cabin(), _occ())
    assert distance(v, v) == 0.0


def test_near_scene_is_similar():
    v1 = featurize(_cabin(ambient=30.0, sun=800.0), _occ())
    v2 = featurize(_cabin(ambient=31.0, sun=820.0), _occ())
    assert distance(v1, v2) < THRESHOLDS.SIM_THRESHOLD


def test_season_isolation():
    assert distance(featurize(_cabin(season="summer"), _occ()),
                    featurize(_cabin(season="winter"), _occ())) > THRESHOLDS.SIM_THRESHOLD


def test_activity_separates():
    assert distance(featurize(_cabin(), _occ(activity="sleeping")),
                    featurize(_cabin(), _occ(activity="excited"))) > THRESHOLDS.SIM_THRESHOLD


def test_person_category_separates():
    assert distance(featurize(_cabin(), _occ(age=6)),
                    featurize(_cabin(), _occ(age=70))) > THRESHOLDS.SIM_THRESHOLD


def test_large_ambient_gap_not_similar():
    assert distance(featurize(_cabin(ambient=20.0, sun=100.0), _occ()),
                    featurize(_cabin(ambient=44.0, sun=1400.0), _occ())) > THRESHOLDS.SIM_THRESHOLD
