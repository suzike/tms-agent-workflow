"""全局配置与可调阈值集中收口。

所有"魔法数"统一在此定义,便于调参与测试覆盖。环境变量经 python-dotenv 加载。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # python-dotenv 未安装时静默跳过,使用进程环境变量
    pass


# ---- 路径 ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MEMORY_FILE = DATA_DIR / "memory.json"
KNOWLEDGE_DIR = PROJECT_ROOT / "tms_agent" / "knowledge" / "docs"


# ---- 业务取值范围 ----
TEMP_MIN = 15.5
TEMP_MAX = 31.5
TEMP_STEP = 0.5
FAN_MIN = 1
FAN_MAX = 7
FAN_STEP = 1
# 舒适基础模式(由舒适推理在三者间选择)
BASE_AIR_MODES = ("face", "face_feet", "feet")
# 完整 7 种出风模式 = 3 基础 + 除霜叠加 + 纯除霜(仅最大除风档)
AIR_MODES = (
    "face", "face_feet", "feet",
    "face_defrost", "face_feet_defrost", "feet_defrost",
    "defrost",
)
SEATS = ("driver", "front_passenger", "rear_left", "rear_right")
SEASONS = ("summer", "winter", "transition")
# 雨量信号等级:无/小雨/中雨/大雨(供独立"智能除雾 Agent"前馈判定)
RAIN_LEVELS = ("none", "light", "moderate", "heavy")
# 除雾紧急度等级:无/轻度/强(强度叠加到出风模式,strong 保证最低风量)
DEFOG_LEVELS = ("none", "mild", "strong")

# 太阳辐照强度(W/m²),按座位独立;0~1500
SUN_WM2_MIN = 0.0
SUN_WM2_MAX = 1500.0

# 人员状态枚举
GENDERS = ("male", "female")
EMOTIONS = ("happy", "angry", "sad", "joy", "worried", "neutral")  # 喜怒哀乐愁
ACTIVITIES = ("excited", "calm", "sleeping")  # "热状态":活动/代谢状态
CLOTHINGS = ("light", "medium", "heavy")      # 衣着:薄/常规/厚
PERSON_CATEGORIES = ("child", "adult", "elderly")  # 小孩/大人/老人


@dataclass(frozen=True)
class Thresholds:
    """记忆/学习/防抖等核心阈值(不可变)。"""

    # 记忆相似度:加权欧氏距离 < SIM_THRESHOLD 视为"相似场景"
    SIM_THRESHOLD: float = 1.0
    # 逐步逼近比例 k:每次推理沿"当前游标→用户偏好"迈出 k×差值的一步(比例控制,
    # 差值越大首步越大);接近一格内时贴齐偏好。n 次推理迭代收敛,而非一次到位。
    APPROACH_RATE: float = 0.5
    # 记忆时间衰减半衰期(天):越久的修正权重越低
    MEMORY_HALFLIFE_DAYS: float = 30.0
    # 有效记录门槛:衰减权重低于此值(约超过一个半衰期)的旧记录不再计入
    # 一致计数(自然老化),但仍参与偏好代表值的加权。
    MEMORY_ACTIVE_FLOOR: float = 0.5

    # 防抖迟滞:仅抑制亚网格噪声/零变化;单步逼近(0.5℃ / 1 档)应能生效,
    # 否则会吃掉逐步逼近的中间步。
    TEMP_DEADBAND: float = 0.4   # 0.5℃ 步进 > 0.4 → 生效;<0.4 噪声被抑制
    FAN_DEADBAND: int = 0        # 任何整档变化(≥1)都生效

    # 手动覆盖后的冷静期(秒):该座位只记录不自动改
    LOCK_WINDOW_SECONDS: float = 600.0

    # 事件触发:关键维度变化超过阈值则立即触发推理
    EVENT_TEMP_DELTA: float = 2.0
    # 周期触发间隔(秒)
    PERIODIC_INTERVAL_SECONDS: float = 30.0

    # 结果缓存 TTL(秒)
    CACHE_TTL_SECONDS: float = 60.0


@dataclass(frozen=True)
class DefogThresholds:
    """独立"智能除雾 Agent"的判定阈值(全部集中于此,便于调参)。

    判定输入:玻璃表面温度、玻璃附近空气温度、玻璃附近相对湿度、雨量信号。
    反馈核心:玻璃温度 - 露点 = 裕度;裕度越小越易结露起雾。
    """

    # 反馈裕度阈值(玻璃温度 - 露点,℃)
    margin_strong: float = 2.0   # 裕度 ≤2℃(含已结露的负值)→ 强除雾(提前介入)
    margin_mild: float = 4.0     # 2℃ < 裕度 ≤4℃ → 轻度除雾(预警)
    # 前馈:玻璃附近相对湿度高于此值视为高湿(易起雾)
    humidity_high: float = 90.0
    humidity_mid: float = 80.0
    # 强除雾时保证的最低风量档(安全优先于 NVH/体感,确保气流冲刷玻璃)
    fan_floor_strong: int = 4


@dataclass(frozen=True)
class FeatureWeights:
    """特征各维度权重:温度/人热状态/日照/季节权重高,车速等低。"""

    ambient_temp: float = 1.0
    local_cabin_temp: float = 1.2
    local_sun: float = 1.0          # 太阳辐照(归一化)
    humidity: float = 0.5
    activity: float = 0.8           # 活动/代谢("热状态")
    clothing: float = 0.6           # 衣着
    person_category: float = 0.8    # 小孩/大人/老人
    season: float = 2.0             # 高权重 → 季节隔离
    time_of_day: float = 0.4
    speed: float = 0.2


@dataclass(frozen=True)
class ComfortDefaults:
    """热舒适计算默认人体参数(缺值兜底用)。"""

    met: float = 1.1  # 代谢率:静坐驾驶约 1.1
    clo_summer: float = 0.5  # 季节兜底着装热阻(无衣着输入时)
    clo_winter: float = 1.0
    clo_transition: float = 0.7

    # 衣着 → 服装热阻 clo(有衣着输入时优先,使 clo 成为实测量而非假设)
    clo_light: float = 0.4    # 薄(短袖/夏装)
    clo_medium: float = 0.7   # 常规(长袖)
    clo_heavy: float = 1.1    # 厚(外套/冬装)

    # 车厢热模型:满日照(1000 W/m²)密闭静止时,车内相对车外的稳态升温(℃)
    # 仅在"车内温度"输入缺失时作回退估算使用
    soak_gain_at_1kw: float = 22.0

    # 活动("热状态")→ 代谢率 met
    met_sleeping: float = 0.8
    met_calm: float = 1.1
    met_excited: float = 1.5

    # 人体特征 → 目标舒适温度启发式偏移(℃,正=偏暖)
    child_offset: float = 0.5    # 小孩对冷更敏感,略偏暖
    elderly_offset: float = 1.0  # 老人代谢低、循环弱,偏暖
    female_offset: float = 0.3   # 统计上女性偏好略暖
    bmi_high_offset: float = -0.3  # BMI≥28 体脂保温好,偏凉
    bmi_low_offset: float = 0.3    # BMI<18.5 偏瘦,偏暖
    bmi_high: float = 28.0
    bmi_low: float = 18.5
    child_max_age: int = 12
    elderly_min_age: int = 65

    # 瞬态控制(快速降温/升温 → 回归稳态):
    # 车内离目标越远,设定温度越激进(制冷更低/制热更高)以加快收敛,随接近回归目标。
    cooldown_overshoot_rate: float = 0.3   # 设定偏置 = rate × (车内温度-目标)
    cooldown_overshoot_max: float = 4.0    # 偏置上限(℃)
    fan_per_degree: float = 3.0            # 每 N℃ 温差增加 1 档风量
    # 稳态专业基准风量落脚点(兼顾舒适与 NVH):常规取 2 档(临近稳态可能为 2~3 档)。
    # 1 档过小,不作为专业基准;仅当个性化记忆中用户习惯稳态调 1 档时,由逐步逼近学到 1。
    fan_steady_min: int = 2


# 风量档位 → 乘员处等效风速 (m/s) 查表(PMV 用)。1~7 档。
FAN_VELOCITY_TABLE = {1: 0.10, 2: 0.20, 3: 0.35, 4: 0.55, 5: 0.80, 6: 1.10, 7: 1.45}
# 评估 PMV 时若不知当前风量,采用的默认当前档位(AUTO 中低档)
CURRENT_FAN_DEFAULT = 3


@dataclass(frozen=True)
class LLMConfig:
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "deepseek"))
    timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("LLM_TIMEOUT_SECONDS", "5"))
    )

    def resolve(self) -> dict:
        """按 provider 返回 {model, api_key, base_url}。"""
        p = self.provider.lower()
        if p == "deepseek":
            return {
                "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
                "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            }
        if p == "openai":
            return {
                "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "api_key": os.getenv("OPENAI_API_KEY", ""),
                "base_url": os.getenv("OPENAI_BASE_URL", None),
            }
        if p == "glm":
            return {
                "model": os.getenv("GLM_MODEL", "glm-4-flash"),
                "api_key": os.getenv("GLM_API_KEY", ""),
                "base_url": os.getenv(
                    "GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"
                ),
            }
        if p == "claude":
            return {
                "model": os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6"),
                "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
                "base_url": os.getenv("ANTHROPIC_BASE_URL", None),
            }
        return {"model": "mock", "api_key": "", "base_url": None}


THRESHOLDS = Thresholds()
DEFOG = DefogThresholds()
WEIGHTS = FeatureWeights()
COMFORT_DEFAULTS = ComfortDefaults()
LLM_CONFIG = LLMConfig()
