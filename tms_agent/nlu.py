"""自然语言指令理解(语音转文本后接入)。

把乘员的口语诉求(如"太冷了""天太热了""风太大")转成对该座位的空调设定调整。
优先用 LLM 实时理解(provider 支持时),离线/失败回退关键词规则。结果作为一次修正
写入记忆,既即时生效又被学习。
"""
from __future__ import annotations

import re

from .config import FAN_MAX, FAN_MIN, TEMP_MAX, TEMP_MIN, TEMP_STEP
from .schemas import ZoneSetting


def _clamp_temp(t: float) -> float:
    return round(max(TEMP_MIN, min(t, TEMP_MAX)) / TEMP_STEP) * TEMP_STEP


def _clamp_fan(f: int) -> int:
    return int(max(FAN_MIN, min(f, FAN_MAX)))


def rule_interpret(text: str, current: ZoneSetting) -> ZoneSetting:
    """关键词规则解析(离线兜底):相对当前设定做调整。"""
    temp, fan, mode = current.temp_set, current.fan_level, current.air_mode
    reason = f"语音指令「{text}」"

    if any(k in text for k in ("太冷", "冷", "暖", "热一点", "升温", "凉")):
        if "太热" not in text and "热死" not in text:
            temp += 2.0
    if any(k in text for k in ("太热", "热", "凉快", "凉一点", "降温", "闷热")):
        if "太冷" not in text:
            temp -= 2.0
    if any(k in text for k in ("风太大", "风大了", "风小一点", "小风", "降低风量")):
        fan -= 2
    if any(k in text for k in ("风太小", "风大一点", "大风", "加大风量", "风力不够")):
        fan += 2
    if "闷" in text:
        fan += 1
    if any(k in text for k in ("吹脸", "对着脸", "脸")):
        mode = "face"
    elif any(k in text for k in ("吹脚", "暖脚", "脚")):
        mode = "feet"
    if any(k in text for k in ("除雾", "起雾", "除霜", "雾")):
        mode = "defrost"

    m = (re.search(r"(\d{2}(?:\.\d)?)\s*度", text)
         or re.search(r"调到\s*(\d{2}(?:\.\d)?)", text))
    if m:
        temp = float(m.group(1))

    return ZoneSetting(
        seat_id=current.seat_id, fan_level=_clamp_fan(fan),
        temp_set=_clamp_temp(temp), air_mode=mode, reasoning=reason,
    )


def interpret(text: str, current: ZoneSetting, deciders) -> ZoneSetting:
    """理解指令 → 调整后的座位设定。LLM 优先,失败/Mock 回退规则。"""
    primary, _ = deciders
    if hasattr(primary, "interpret"):
        try:
            return primary.interpret(text, current)
        except Exception:
            pass
    return rule_interpret(text, current)
