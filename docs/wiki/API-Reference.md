# API 参考

## Engine (engine.py)

核心引擎，管理推理、记忆与决策。

```python
from tms_agent.engine import Engine

engine = Engine()

# 推理（返回决策 + 可选推理链）
state = engine.infer(capture_chain=True)

# 应用用户修正
engine.apply_correction(scene, seat, corrected)

# 语音/对话入口
engine.apply_command(scene, seat, text)

# 除雾决策
defog_decision = engine.defog_for(cabin)
```

### `engine.infer(capture_chain=True)`

推理所有座位的设定。

- **capture_chain**: 是否捕获推理链供 Web 展示
- **返回**: `dict[seat_position, dict]`, 包含 temp/fan/mode 及推理链

### `engine.apply_correction(scene, seat, corrected)`

记录用户修正，写入记忆。

- **scene**: 输入快照（CabinContext + OccupantState）
- **seat**: 座位位置
- **corrected**: 用户修正后的设定

## Safety (safety.py)

安全约束层。

```python
from tms_agent.safety import apply_safety

final = apply_safety(comfort_setting, defog_decision, cabin)
# 返回: 落定的 temp/fan/mode（7 态）
```

### 设定域

| 参数 | 范围 | 精度 |
|------|------|------|
| 温度 | 15.5 ~ 31.5 ℃ | 0.5 |
| 风量 | 1 ~ 7 | 1 |
| 模式 | 7 种 | — |

出风模式：
- `face` — 吹面（制冷）
- `feet` — 吹脚（制热）
- `face_feet` — 吹面吹脚（过渡）
- `face_defrost` — 吹面 + 除霜
- `feet_defrost` — 吹脚 + 除霜
- `face_feet_defrost` — 吹面吹脚 + 除霜
- `defrost` — 纯除霜（仅 max_defrost 模式）

## Memory (memory/store.py)

```python
from tms_agent.memory.store import MemoryStore

store = MemoryStore()

# 回忆
memories = store.recall(user_id, seat, scene_vector, now)

# 录入修正
store.record(correction_record)

# 获取游标
cursor = store.get_cursor(user_id, seat)
```

## Tools (tools/thermal_comfort.py)

```python
from tms_agent.tools.thermal_comfort import (
    ThermalComfortSkill,
    pmv_ppd,
    comfort_target_temp,
)

# PMV/PPD 计算
pmv, ppd = pmv_ppd(ta, mrt, vel, rh, met, clo)

# 舒适目标温度
target = comfort_target_temp(season, occupant)
```

## Config (config.py)

所有阈值集中管理：

```python
# 舒适默认值
ComfortDefaults.target_temp_summer = 24.0
ComfortDefaults.target_temp_winter = 22.0
ComfortDefaults.fan_steady_min = 2

# 逐步逼近
Thresholds.APPROACH_RATE = 0.5
Thresholds.FAN_DEADBAND = 0
Thresholds.TEMP_DEADBAND = 0.4

# 除雾阈值
DefogThresholds.margin_strong = 2.0
DefogThresholds.margin_mild = 4.0
DefogThresholds.fan_floor_strong = 4
```
