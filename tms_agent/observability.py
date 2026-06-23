"""可观测与评估:日志 + 会话级修正率指标。"""
from __future__ import annotations

import logging
from dataclasses import dataclass

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def get_logger(name: str = "tms_agent") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@dataclass
class SessionMetrics:
    """代理指标:用户修正率越低,说明推荐越贴合(可解释的产品质量信号)。"""

    recommendations: int = 0
    corrections: int = 0

    def record_recommendation(self) -> None:
        self.recommendations += 1

    def record_correction(self) -> None:
        self.corrections += 1

    @property
    def correction_rate(self) -> float:
        return self.corrections / self.recommendations if self.recommendations else 0.0
