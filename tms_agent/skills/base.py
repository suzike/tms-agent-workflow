"""Skill 统一抽象:把 Tools / Knowledge / Memory 封装为可注册、可描述、可调用的能力。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Skill(ABC):
    """能力基类。name/description 供编排理解,invoke 执行并返回结构化结果。"""

    name: str = "skill"
    description: str = ""

    @abstractmethod
    def invoke(self, context: dict[str, Any]) -> dict[str, Any]:
        """执行能力。context 由调用方提供(如 cabin/occupant),返回结构化 dict。"""
        raise NotImplementedError


class SkillRegistry:
    """Skill 注册表:新增能力只需注册,不改图。"""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        return self._skills[name]

    def names(self) -> list[str]:
        return list(self._skills)

    def describe(self) -> dict[str, str]:
        return {name: s.description for name, s in self._skills.items()}
