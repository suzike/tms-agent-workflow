"""命令行界面(Typer + Rich):演示推荐 / 修正 / 自动纠正闭环。

记忆持久化于 data/memory.json,故跨命令调用可复现完整闭环:
    infer 0  →  correct 0 driver 19 6 face_feet  →  (重复几次)  →  infer 0 看自动纠正
"""
from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import PROJECT_ROOT
from .engine import Engine, InferenceResult
from .schemas import SceneInput, ZoneSetting, sanitize_scene

app = typer.Typer(help="座舱智慧空调 Agent —— CLI 演示")
console = Console()

_SCENES_FILE = PROJECT_ROOT / "data" / "mock_scenes.json"


def _load_scenes() -> list[dict]:
    return json.loads(_SCENES_FILE.read_text(encoding="utf-8"))


def _scene_at(index: int) -> tuple[str, SceneInput]:
    scenes = _load_scenes()
    if not 0 <= index < len(scenes):
        raise typer.BadParameter(f"场景下标越界,可用 0..{len(scenes) - 1}")
    item = scenes[index]
    scene, _ = sanitize_scene(item["scene"])
    return item["name"], scene


_CAT_ZH = {"child": "小孩", "adult": "大人", "elderly": "老人"}
_ACT_ZH = {"sleeping": "睡眠", "calm": "平静", "excited": "兴奋"}
_GENDER_ZH = {"male": "男", "female": "女"}


def _occupant_brief(user: str, occ) -> str:
    """乘员摘要:用户·类别性别年龄·BMI·活动。"""
    if occ is None:
        return user
    return (f"{user}·{_CAT_ZH[occ.category]}{_GENDER_ZH[occ.gender]}{occ.age}"
            f"·BMI{occ.bmi}·{_ACT_ZH[occ.activity]}")


def _render(name: str, result: InferenceResult, scene: SceneInput) -> None:
    occ_by_seat = {o.seat_id: o for o in scene.occupants}
    table = Table(title=f"推荐结果 · {name}", expand=False)
    for col in ("座位", "乘员(用户·类别性别龄·BMI·活动)", "车内估℃", "温度℃", "风量",
                "出风", "来源", "PMV", "目标℃", "应用", "知识依据"):
        table.add_column(col, overflow="fold")
    for seat, s in result.settings.items():
        t = result.traces[seat]
        m = t.comfort_metrics
        src = t.source + (f"+逼近{t.approach_weight:.2f}" if t.approach_weight > 0 else "")
        table.add_row(
            seat, _occupant_brief(t.user_id, occ_by_seat.get(seat)),
            f"{m.cabin_temp}" if m else "-",
            f"{s.temp_set}", str(s.fan_level), s.air_mode,
            src, f"{m.pmv}" if m else "-",
            f"{m.target_temp}" if m else "-",
            "✓" if result.applied[seat] else "·",
            (t.knowledge_snippets[0] if t.knowledge_snippets else "-"),
        )
    console.print(table)


def _render_defog(eng: Engine, scene: SceneInput) -> None:
    """打印车厢级智能除雾 Agent 决策(与 Web 顶部横幅一致)。"""
    d = eng.defog_for(scene.cabin)
    if d.need_defog:
        console.print(f"🌫 [yellow]智能除雾 Agent:需要除雾({d.level})[/] · "
                      f"来源 {d.source} · {d.reasoning}")
    else:
        console.print(f"🌫 [green]智能除雾 Agent:无需除雾[/] · 来源 {d.source}")


@app.command("list")
def list_scenes() -> None:
    """列出内置演示场景。"""
    table = Table(title="演示场景")
    table.add_column("#"); table.add_column("名称"); table.add_column("乘员")
    for i, item in enumerate(_load_scenes()):
        occ = ",".join(o["seat_id"] for o in item["scene"]["occupants"])
        table.add_row(str(i), item["name"], occ)
    console.print(table)


@app.command()
def infer(index: int = typer.Argument(..., help="场景下标")) -> None:
    """对某场景推理并展示各座位推荐(含除雾 Agent 决策、舒适指标与知识依据)。"""
    name, scene = _scene_at(index)
    eng = Engine()
    _render_defog(eng, scene)
    _render(name, eng.infer(scene), scene)


@app.command()
def say(index: int, seat: str, text: str) -> None:
    """语音/对话指令(与 Web 对话框一致):接收转写文本 → 理解调整 → 写入记忆。

    例:app_cli say 0 driver 太冷了
    """
    name, scene = _scene_at(index)
    eng = Engine()
    corrected = eng.apply_command(scene, seat, text)
    console.print(
        f"[green]{seat}「{text}」→[/] {corrected.temp_set}℃/"
        f"{corrected.fan_level}档/{corrected.air_mode}(已理解并写入记忆)"
    )


@app.command()
def chain(index: int = typer.Argument(0, help="场景下标")) -> None:
    """多 Agent 实时推理链(与 Web「实时推理链」一致):除雾 Agent + 各座位舒适 Agent。"""
    name, scene = _scene_at(index)
    eng = Engine()
    console.print(f"[bold]🌫 智能除雾 Agent 推理链(车厢级)[/] · {name}")
    for s in eng.stream_defog(scene.cabin):
        console.print(f"  {s.title} — {s.detail}")
    res = eng.infer(scene, capture_chain=True)
    for seat, steps in res.chains.items():
        console.print(f"[bold]🌡 舒适 Agent · {seat}[/]")
        for s in steps:
            console.print(f"  {s.title} — {s.detail}")


@app.command()
def memory() -> None:
    """查看学习记忆完整链条(与 Web「学习记忆」一致):输入快照 → 推理 → 用户修正。"""
    eng = Engine()
    summ = eng.store.summary()
    if not summ:
        console.print("[dim]暂无学习记忆。先 correct / say 几次后再看。[/]")
        return
    for (user, seat), cnt in sorted(summ.items()):
        console.print(f"[bold cyan]{user} × {seat}[/] · {cnt} 条记忆")
        for i, rec in enumerate(eng.store.records_for(user, seat), 1):
            c, o = rec.cabin, rec.occupant
            env = (f"{c.season}/外{c.ambient_temp:.0f}℃/"
                   f"内{('%.0f℃' % c.cabin_temp) if c.cabin_temp is not None else '估'}/"
                   f"日照{c.seat_sun(seat):.0f}/湿{c.humidity:.0f}%/雨{c.rain_level}"
                   if c else "-")
            per = (f"{_CAT_ZH[o.category]}{_GENDER_ZH[o.gender]}{o.age}/"
                   f"{o.clothing}/{_ACT_ZH[o.activity]}" if o else "-")
            console.print(
                f"  #{i} 环境「{env}」人「{per}」"
                f"推理 {rec.recommended.temp_set}℃/{rec.recommended.fan_level}档/"
                f"{rec.recommended.air_mode} → "
                f"[yellow]修正 {rec.corrected.temp_set}℃/{rec.corrected.fan_level}档/"
                f"{rec.corrected.air_mode}[/]"
            )


@app.command()
def correct(
    index: int, seat: str, temp: float, fan: int, mode: str,
    user: Optional[str] = typer.Option(None, help="覆盖该座位用户名"),
) -> None:
    """对某座位手动修正(写入记忆并锁定该座位)。"""
    name, scene = _scene_at(index)
    eng = Engine()
    rec = eng.infer(scene).settings[seat]
    corrected = ZoneSetting(seat_id=seat, fan_level=fan, temp_set=temp, air_mode=mode)
    eng.apply_correction(scene, seat, rec, corrected)
    console.print(
        f"[green]已记录修正[/]:{name} · {seat} {rec.temp_set}→{temp}℃ "
        f"(修正率 {eng.metrics.correction_rate:.2f})"
    )


@app.command()
def teach(index: int = typer.Argument(0, help="场景下标"),
          seat: str = "driver", temp: float = 19.0,
          fan: int = 6, mode: str = "face_feet") -> None:
    """自动演示闭环:连续修正同一座位,再推理展示自动纠正。"""
    name, scene = _scene_at(index)
    eng = Engine()
    corrected = ZoneSetting(seat_id=seat, fan_level=fan, temp_set=temp, air_mode=mode)
    console.print(f"[bold]教学闭环[/]:{name} · 连续修正 {seat} → {temp}℃/{mode}")
    for _ in range(3):
        rec = eng.infer(scene).settings[seat]
        eng.apply_correction(scene, seat, rec, corrected)
    console.print("再次推理(应自动纠正):")
    _render(name, eng.infer(scene), scene)


@app.command()
def reset() -> None:
    """清空所有学习记忆。"""
    Engine().store.reset_all()
    console.print("[yellow]已清空记忆。[/]")


if __name__ == "__main__":
    app()
