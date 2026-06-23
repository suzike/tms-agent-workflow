"""座舱智慧空调 Agent · 智能座舱 HMI(Streamlit)。

设计语言:暗色驾驶舱、青(冷)↔琥珀(热)双色霓虹、玻璃拟态面板、Orbitron 数显字体。
中枢是"车厢俯视图 HUD",每个座位按推荐温度发光着色,直观呈现多温区结果。

运行:streamlit run tms_agent/app_web.py
"""
from __future__ import annotations

import time

import streamlit as st
import streamlit.components.v1 as components

from tms_agent.engine import Engine
from tms_agent.llm.provider import ping_llm, provider_status
from tms_agent.schemas import CabinContext, OccupantState, SceneInput, ZoneSetting

_GENDERS = ["male", "female"]
_EMOTIONS = ["neutral", "happy", "angry", "sad", "joy", "worried"]
_ACTIVITIES = ["excited", "calm", "sleeping"]
_CLOTHINGS = ["light", "medium", "heavy"]
_SEATS = ["driver", "front_passenger", "rear_left", "rear_right"]
_SEASONS = ["summer", "winter", "transition"]
_WEATHER = ["sunny", "cloudy", "rain", "snow"]
_RAIN = ["none", "light", "moderate", "heavy"]
_RAIN_ZH = {"none": "无雨", "light": "小雨", "moderate": "中雨", "heavy": "大雨"}
_DEFOG_ZH = {"none": "无需除雾", "mild": "轻度除雾", "strong": "强除雾"}
_AIR_MODES = ["face", "face_feet", "feet", "face_defrost", "face_feet_defrost",
              "feet_defrost", "defrost"]

_SEAT_ZH = {"driver": "主驾", "front_passenger": "副驾",
            "rear_left": "左后", "rear_right": "右后"}
_MODE_ZH = {"face": "吹面", "face_feet": "面+脚", "feet": "吹脚",
            "face_defrost": "吹面·除霜", "face_feet_defrost": "面脚·除霜",
            "feet_defrost": "吹脚·除霜", "defrost": "纯除霜"}
_CAT_ZH = {"child": "小孩", "adult": "大人", "elderly": "老人"}
_ACT_ZH = {"sleeping": "睡眠", "calm": "平静", "excited": "兴奋"}

# 俯视图座位坐标(viewBox 680x780)
_SEAT_XY = {"driver": (240, 352), "front_passenger": (440, 352),
            "rear_left": (240, 582), "rear_right": (440, 582)}


# --------------------------------------------------------------------------- #
# 主题 / 配色
# --------------------------------------------------------------------------- #
def _inject_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600;700&family=Noto+Sans+SC:wght@400;500;700&display=swap');

:root{
  --cyan:#00E5FF; --amber:#FF8A3D; --ink:#070B12; --panel:#0E1622;
  --line:rgba(0,229,255,.18); --txt:#D7E3EE; --muted:#7E93A8;
}
html,body,[class*="css"]{ font-family:'Rajdhani','Noto Sans SC',sans-serif; }
.stApp{
  background:
    radial-gradient(900px 600px at 18% -5%, rgba(0,229,255,.10), transparent 60%),
    radial-gradient(800px 600px at 95% 0%, rgba(255,138,61,.08), transparent 55%),
    linear-gradient(180deg,#070B12 0%, #050810 100%);
  color:var(--txt);
}
/* 顶部 HUD 标题条 */
.hud-top{
  border:1px solid var(--line); border-radius:14px; padding:14px 20px; margin-bottom:6px;
  background:linear-gradient(120deg, rgba(0,229,255,.06), rgba(13,22,34,.5));
  box-shadow:0 0 30px rgba(0,229,255,.08) inset, 0 0 22px rgba(0,0,0,.4);
  display:flex; align-items:center; gap:16px; flex-wrap:wrap;
}
.hud-title{ font-family:'Orbitron',sans-serif; font-weight:900; letter-spacing:3px;
  font-size:26px; color:#EAF6FF; text-shadow:0 0 18px rgba(0,229,255,.55); margin:0;}
.hud-sub{ color:var(--muted); letter-spacing:1px; font-size:13px; }
.chip{ font-family:'Rajdhani'; font-weight:600; font-size:12.5px; letter-spacing:1px;
  padding:4px 11px; border-radius:999px; border:1px solid var(--line);
  background:rgba(0,229,255,.06); color:var(--cyan); }
.chip.warn{ color:var(--amber); border-color:rgba(255,138,61,.35); background:rgba(255,138,61,.07);}
.chip.dim{ color:var(--muted); border-color:rgba(126,147,168,.3); background:rgba(126,147,168,.06);}
h1,h2,h3{ font-family:'Rajdhani','Noto Sans SC'; letter-spacing:1px; color:#EAF6FF; }
[data-testid="stMetricValue"]{ font-family:'Orbitron'; color:var(--cyan);
  text-shadow:0 0 14px rgba(0,229,255,.4);}
[data-testid="stMetricLabel"]{ color:var(--muted); letter-spacing:1px;}
/* 卡片容器 */
[data-testid="stVerticalBlockBorderWrapper"]{
  background:linear-gradient(160deg, rgba(14,22,34,.7), rgba(8,12,20,.6));
  border:1px solid var(--line)!important; border-radius:14px;
  box-shadow:0 0 24px rgba(0,0,0,.35), 0 0 18px rgba(0,229,255,.05) inset;
}
/* 按钮:霓虹描边 */
.stButton>button{
  font-family:'Rajdhani'; font-weight:700; letter-spacing:1.5px;
  background:rgba(0,229,255,.08); color:var(--cyan);
  border:1px solid var(--line); border-radius:10px; transition:.18s;
}
.stButton>button:hover{ background:rgba(0,229,255,.18); border-color:var(--cyan);
  box-shadow:0 0 18px rgba(0,229,255,.35); color:#EAF6FF; }
/* Tabs */
.stTabs [data-baseweb="tab-list"]{ gap:6px; border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{ font-family:'Rajdhani'; font-weight:600; letter-spacing:1px;}
.stTabs [aria-selected="true"]{ color:var(--cyan)!important; }
/* 推理链控制台 */
.chain-line{ font-family:'Rajdhani','Noto Sans SC'; border-left:2px solid var(--cyan);
  padding:7px 14px; margin:6px 0; background:rgba(0,229,255,.04); border-radius:0 8px 8px 0;}
.chain-final{ border-left-color:var(--amber); background:rgba(255,138,61,.07);
  box-shadow:0 0 16px rgba(255,138,61,.12);}
.chain-node{ color:var(--cyan); font-weight:700; letter-spacing:1px;}
.readout{ font-family:'Orbitron'; font-size:13px; color:var(--muted); letter-spacing:1px;}
.calc-box{ border:1px solid var(--line); border-radius:8px; padding:8px 10px;
  background:rgba(0,229,255,.04);}
.calc-row{ font-family:'Rajdhani','Noto Sans SC'; font-size:13px; color:#bcd;
  padding:3px 0; border-bottom:1px dashed rgba(0,229,255,.12);}
.calc-row:last-child{ border-bottom:none; }
/* 座舱 HUD 气流/热晕动画(全局定义,作用于内联 SVG 元素) */
.flow{ fill:none; stroke-linecap:round; animation-name:flowdash;
  animation-timing-function:linear; animation-iteration-count:infinite; }
@keyframes flowdash{ to{ stroke-dashoffset:-44; } }
.aura{ animation:breathe 3.4s ease-in-out infinite; }
@keyframes breathe{ 0%,100%{ opacity:.6; } 50%{ opacity:1; } }
section[data-testid="stSidebar"]{ background:linear-gradient(180deg,#0A0F18,#070B12);
  border-right:1px solid var(--line);}
::-webkit-scrollbar{width:9px;height:9px;} ::-webkit-scrollbar-thumb{
  background:rgba(0,229,255,.25); border-radius:6px;}
/* 语音/对话输入框:高亮描边 + 呼吸光晕(精准命中 key 前缀 cmd_,不影响其它输入) */
[class*="st-key-cmd_"] label p{ color:var(--cyan)!important; font-weight:600; letter-spacing:.5px;}
[class*="st-key-cmd_"] div[data-baseweb="input"]{
  border:1.6px solid rgba(0,229,255,.7)!important;
  background:rgba(0,229,255,.06)!important; border-radius:10px!important;
  box-shadow:0 0 0 1px rgba(0,229,255,.15), 0 0 16px rgba(0,229,255,.18);
  animation:cmdGlow 2.6s ease-in-out infinite;
}
[class*="st-key-cmd_"] div[data-baseweb="input"]:focus-within{
  border-color:var(--cyan)!important;
  box-shadow:0 0 0 2px rgba(0,229,255,.5), 0 0 22px rgba(0,229,255,.42)!important;
  animation:none;
}
[class*="st-key-cmd_"] input{ color:#EAF6FF!important; font-weight:500; }
[class*="st-key-cmd_"] input::placeholder{ color:#86b9c8!important; opacity:.95; }
@keyframes cmdGlow{
  0%,100%{ box-shadow:0 0 0 1px rgba(0,229,255,.14), 0 0 11px rgba(0,229,255,.13);}
  50%{ box-shadow:0 0 0 1px rgba(0,229,255,.32), 0 0 22px rgba(0,229,255,.32);}
}
/* 学习状态徽标 + 手动调节区标题 */
.learn-badge{ display:inline-block; margin:4px 0 6px; padding:3px 11px; border-radius:999px;
  font-family:'Rajdhani'; font-size:12.5px; font-weight:700; color:#06121a;
  background:linear-gradient(90deg,#00E5FF,#2ede96); letter-spacing:.5px;}
.learn-badge.dim{ color:var(--muted); background:rgba(126,147,168,.12);
  border:1px solid rgba(126,147,168,.28); }
.adjust-head{ margin-top:10px; font-family:'Rajdhani'; font-weight:700; letter-spacing:.5px;
  color:var(--amber); border-top:1px solid var(--line); padding-top:8px; font-size:13.5px;}
.cooldown{ display:inline-block; margin:2px 0 4px; padding:3px 11px; border-radius:999px;
  font-family:'Rajdhani'; font-size:12.5px; font-weight:700; letter-spacing:.5px;
  color:var(--amber); background:rgba(255,138,61,.10);
  border:1px solid rgba(255,138,61,.4); animation:cdPulse 1.2s ease-in-out infinite;}
@keyframes cdPulse{ 0%,100%{ box-shadow:0 0 0 rgba(255,138,61,0);}
  50%{ box-shadow:0 0 12px rgba(255,138,61,.35);} }
</style>
        """,
        unsafe_allow_html=True,
    )


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _ramp(temp: float, stops: list) -> str:
    """在颜色停靠点之间线性插值,返回 #rrggbb。"""
    if temp <= stops[0][0]:
        r, g, b = stops[0][1]
    elif temp >= stops[-1][0]:
        r, g, b = stops[-1][1]
    else:
        r = g = b = 0
        for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
            if t0 <= temp <= t1:
                f = (temp - t0) / (t1 - t0)
                r, g, b = (int(_lerp(c0[i], c1[i], f)) for i in range(3))
                break
    return f"#{r:02x}{g:02x}{b:02x}"


def _temp_color(temp: float) -> str:
    """设定温度域(15.5–31.5℃)→ 颜色:冷青 → 中绿 → 暖琥珀(用于送风/设定值)。"""
    return _ramp(temp, [(15.5, (0, 229, 255)), (23.5, (46, 222, 150)),
                        (31.5, (255, 138, 61))])


def _cabin_color(temp: float) -> str:
    """车内实测/估算温度域 → 颜色:越热越暖(用于座舱整体热氛围着色)。

    冷(≤12℃)深青 → 舒适(~23℃)绿 → 热(30℃)琥珀 → 高热(≥42℃)红。
    """
    return _ramp(temp, [(12, (0, 183, 216)), (23, (46, 222, 150)),
                        (30, (255, 176, 60)), (42, (226, 75, 74))])


# --------------------------------------------------------------------------- #
# 车厢俯视图 HUD
# --------------------------------------------------------------------------- #
def _fan_dots(cx: float, y: float, color: str, fan: int) -> str:
    """风量 7 格点阵:点亮数 = 风量档。"""
    n, gap = 7, 12
    x0 = cx - (n - 1) * gap / 2
    out = ""
    for i in range(n):
        c = color if i < fan else "rgba(120,140,160,.22)"
        out += f'<circle cx="{x0 + i*gap:.0f}" cy="{y:.0f}" r="3.4" fill="{c}"/>'
    return out


def _airflow(cx: float, y_start: float, y_end: float, color: str, fan: int) -> str:
    """一束动态气流(简洁版):风量决定线数(2~3)与速度;中心流末端单箭头指向流向。"""
    n = 3 if fan >= 4 else 2
    dur = max(0.5, round(1.5 - fan * 0.12, 2))
    up = y_end < y_start
    span = abs(y_end - y_start)
    xs = [cx - 24 + 48 * i / (n - 1) for i in range(n)]
    out = ""
    for i, x in enumerate(xs):
        amp = 6 if i % 2 == 0 else -6
        c1y = y_start + (-span * 0.35 if up else span * 0.35)
        c2y = y_end + (span * 0.35 if up else -span * 0.35)
        d = (f"M {x:.0f} {y_start:.0f} C {x+amp:.0f} {c1y:.0f} "
             f"{x-amp:.0f} {c2y:.0f} {x:.0f} {y_end:.0f}")
        out += (f'<path class="flow" d="{d}" stroke="{color}" stroke-width="2" '
                f'stroke-dasharray="8 12" opacity="0.7" '
                f'style="animation-duration:{dur}s"/>')
    wy = y_end + (8 if up else -8)
    out += (f'<path d="M {cx-7:.0f} {wy:.0f} L {cx:.0f} {y_end:.0f} '
            f'L {cx+7:.0f} {wy:.0f}" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round" opacity="0.85"/>')
    return out


def _seat_pod(seat: str, occ, setting, cabin_temp) -> str:
    cx, cy = _SEAT_XY[seat]
    front = seat in ("driver", "front_passenger")
    name = _SEAT_ZH[seat]
    pw, ph = 164, 150
    x, y = cx - pw / 2, cy - ph / 2
    if setting is None or occ is None:
        return f"""
<g opacity="0.3">
  <rect x="{x:.0f}" y="{y:.0f}" width="{pw}" height="{ph}" rx="26" fill="rgba(18,26,38,.5)"
        stroke="rgba(126,147,168,.4)" stroke-dasharray="7 7"/>
  <text x="{cx}" y="{cy-12}" fill="#7E93A8" font-size="17" text-anchor="middle"
        font-family="Rajdhani">{name}</text>
  <text x="{cx}" y="{cy+24}" fill="#5b6c7e" font-size="22" text-anchor="middle"
        font-family="Rajdhani">空席</text>
</g>"""
    # 设定温度色(送风/设定值);车内温度色(整体热氛围:越热越暖)
    set_color = _temp_color(setting.temp_set)
    ct = cabin_temp if cabin_temp is not None else setting.temp_set
    cab_color = _cabin_color(ct)
    mode = setting.air_mode
    base = mode.replace("_defrost", "") or "face"
    has_face = base in ("face", "face_feet")
    has_feet = base in ("feet", "face_feet")
    fan = setting.fan_level
    far = 66 if front else 60
    face_air = _airflow(cx, y - 6, y - far, set_color, fan) if has_face else ""
    feet_air = _airflow(cx, y + ph + 6, y + ph + far, set_color, fan) if has_feet else ""
    vent = ""
    if has_face:
        vent += (f'<text x="{cx}" y="{y-far-8:.0f}" fill="{set_color}" font-size="12" '
                 f'text-anchor="middle" font-family="Rajdhani" opacity="0.8">吹面 ▲</text>')
    if has_feet:
        vent += (f'<text x="{cx}" y="{y+ph+far+16:.0f}" fill="{set_color}" font-size="12" '
                 f'text-anchor="middle" font-family="Rajdhani" opacity="0.8">吹脚 ▼</text>')
    return f"""
<g>
  <ellipse cx="{cx}" cy="{cy}" rx="108" ry="100" fill="url(#aura_{seat})" class="aura"/>
  {face_air}{feet_air}
  <rect x="{cx-30:.0f}" y="{y-15:.0f}" width="60" height="22" rx="10"
        fill="rgba(10,16,24,.92)" stroke="{cab_color}" stroke-width="1.8" opacity="0.92"/>
  <g filter="url(#soft)">
    <rect x="{x:.0f}" y="{y:.0f}" width="{pw}" height="{ph}" rx="26"
          fill="rgba(10,16,24,.88)" stroke="{cab_color}" stroke-width="2.6"/>
    <rect x="{x:.0f}" y="{y:.0f}" width="{pw}" height="6" rx="3" fill="{cab_color}"/>
  </g>
  {vent}
  <text x="{cx}" y="{cy-44}" fill="#cfe0ee" font-size="15" text-anchor="middle"
        font-family="Rajdhani" letter-spacing="1">{name} · {occ.user_id}</text>
  <text x="{cx-8}" y="{cy+8}" fill="{set_color}" font-size="50" font-weight="900"
        text-anchor="middle" font-family="Orbitron">{setting.temp_set:.1f}</text>
  <text x="{cx+54}" y="{cy-12}" fill="{set_color}" font-size="15"
        text-anchor="middle" font-family="Orbitron">℃</text>
  <text x="{cx}" y="{cy+30}" fill="{cab_color}" font-size="13" text-anchor="middle"
        font-family="Rajdhani" letter-spacing="0.5">内温 {ct:.0f}℃</text>
  {_fan_dots(cx, cy+50, set_color, fan)}
  <text x="{cx}" y="{cy+72}" fill="#9fb3c6" font-size="13" text-anchor="middle"
        font-family="Rajdhani">{_MODE_ZH.get(mode, mode)} · 风量 {fan}</text>
</g>"""


def _cabin_hud(result, scene: SceneInput) -> str:
    occ_by_seat = {o.seat_id: o for o in scene.occupants}
    defrost = any("defrost" in s.air_mode for s in result.settings.values())
    ws_color = "#FF8A3D" if defrost else "rgba(0,229,255,.55)"
    ws_label = "❄ 除霜中 · DEFOG" if defrost else "前风挡 WINDSHIELD"

    def _cabin_t(seat):
        tr = result.traces.get(seat)
        m = tr.comfort_metrics if tr else None
        s = result.settings.get(seat)
        return m.cabin_temp if m else (s.temp_set if s else None)

    # 每个在座座位生成"热晕"径向渐变(颜色=车内温度,越热越暖、偏离中性越强)
    auras = ""
    for seat in _SEATS:
        s, occ = result.settings.get(seat), occ_by_seat.get(seat)
        if s is None or occ is None:
            continue
        ct = _cabin_t(seat)
        c = _cabin_color(ct if ct is not None else s.temp_set)
        dev = min(1.0, abs((ct if ct is not None else 23) - 23) / 15.0)
        op = round(0.18 + 0.34 * dev, 2)
        auras += (f'<radialGradient id="aura_{seat}" cx="50%" cy="50%" r="50%">'
                  f'<stop offset="0" stop-color="{c}" stop-opacity="{op}"/>'
                  f'<stop offset="55%" stop-color="{c}" stop-opacity="{op*0.4:.2f}"/>'
                  f'<stop offset="100%" stop-color="{c}" stop-opacity="0"/>'
                  f'</radialGradient>')
    # 除霜时前风挡的暖色扫描动画
    sweep = ""
    if defrost:
        for yy in (130, 152, 174):
            sweep += (f'<line class="flow" x1="160" y1="{yy}" x2="520" y2="{yy}" '
                      f'stroke="#FF8A3D" stroke-width="2.2" stroke-dasharray="12 18" '
                      f'opacity="0.6" style="animation-duration:0.85s"/>')
    pods = "".join(
        _seat_pod(seat, occ_by_seat.get(seat), result.settings.get(seat),
                  _cabin_t(seat))
        for seat in _SEATS
    )
    return f"""<svg viewBox="0 0 680 780" width="100%" style="max-width:560px;height:auto;display:block;margin:0 auto;">
  <defs>
    <filter id="soft" x="-35%" y="-35%" width="170%" height="170%">
      <feGaussianBlur stdDeviation="3" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <linearGradient id="body" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#101c2c"/><stop offset="1" stop-color="#080d16"/>
    </linearGradient>
    <linearGradient id="scale" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#00B7D8"/><stop offset="0.45" stop-color="#2ede96"/>
      <stop offset="0.75" stop-color="#FFB03C"/><stop offset="1" stop-color="#E24B4A"/>
    </linearGradient>
    {auras}
  </defs>
  <rect x="44" y="28" width="592" height="724" rx="120" fill="url(#body)"
        stroke="rgba(0,229,255,.22)" stroke-width="1.5"/>
  <rect x="44" y="28" width="592" height="724" rx="120" fill="none"
        stroke="rgba(0,229,255,.06)" stroke-width="12"/>
  <path d="M120 122 Q340 74 560 122 L524 196 Q340 158 156 196 Z"
        fill="rgba(0,229,255,.05)" stroke="{ws_color}" stroke-width="2"/>
  {sweep}
  <text x="340" y="158" fill="{ws_color}" font-size="15" text-anchor="middle"
        font-family="Rajdhani" letter-spacing="2">{ws_label}</text>
  <line x1="340" y1="214" x2="340" y2="704" stroke="rgba(0,229,255,.10)"
        stroke-dasharray="4 9"/>
  {pods}
  <rect x="220" y="734" width="240" height="9" rx="4.5" fill="url(#scale)"/>
  <text x="212" y="743" fill="#7E93A8" font-size="12" text-anchor="end"
        font-family="Orbitron">冷</text>
  <text x="468" y="743" fill="#7E93A8" font-size="12" text-anchor="start"
        font-family="Orbitron">热</text>
  <text x="340" y="766" fill="#5b6c7e" font-size="12" text-anchor="middle"
        font-family="Orbitron" letter-spacing="3">座椅色=内温 · 气流=出风/设定 · 风量=气流密度</text>
</svg>"""


# 富 SVG(滤镜/渐变/动画)经 st.markdown 会被 Markdown 段落化截断,改用 iframe 组件保真渲染。
_SVG_ANIM_STYLE = (
    "<style>html,body{margin:0;padding:0;background:transparent;overflow:hidden;}"
    ".flow{fill:none;stroke-linecap:round;animation:flowdash linear infinite;}"
    "@keyframes flowdash{to{stroke-dashoffset:-44;}}"
    ".aura{animation:breathe 3.4s ease-in-out infinite;}"
    "@keyframes breathe{0%,100%{opacity:.6;}50%{opacity:1;}}</style>"
)


def _svg_iframe(svg: str, height: int) -> None:
    """在隔离 iframe 中渲染 SVG(透明背景,内联动画样式),绕过 Markdown 清洗。"""
    components.html(_SVG_ANIM_STYLE + svg, height=height)


# --------------------------------------------------------------------------- #
# PMV 计算逻辑 + 可视化模块(座舱 HUD 下方)
# --------------------------------------------------------------------------- #
def _pmv_scale(result) -> str:
    """多温区 PMV 标尺:冷(-3)↔热(+3),舒适带 [-0.5,0.5] + 各区当前 PMV(每区独占一行,防压字)。"""
    LO, HI = 56, 624          # 色条左右端 x
    BAR_Y, BAR_H = 112, 20    # 色条位置/高度
    def x_of(p: float) -> float:
        return LO + (max(-3.0, min(p, 3.0)) + 3) / 6 * (HI - LO)
    bx0, bx1 = x_of(-0.5), x_of(0.5)
    rows = ""
    items = [(s, result.traces[s].comfort_metrics) for s in result.settings]
    for i, (seat, m) in enumerate(items):
        if not m:
            continue
        xb = x_of(m.pmv)
        color = _temp_color(result.settings[seat].temp_set)
        ly = 26 + i * 18                       # 每区独占一行,绝不重叠
        lx = min(max(xb, 64), 616)
        anchor = "start" if lx < 120 else ("end" if lx > 560 else "middle")
        flag = "(超出量程)" if abs(m.pmv) > 3 else ""
        rows += (
            f'<text x="{lx:.0f}" y="{ly}" fill="{color}" font-size="14" '
            f'text-anchor="{anchor}" font-family="Rajdhani" font-weight="600">'
            f'{_SEAT_ZH[seat]} {m.pmv:+.2f}{flag}</text>'
            f'<line x1="{xb:.0f}" y1="{ly+5}" x2="{xb:.0f}" y2="{BAR_Y-2}" stroke="{color}" '
            f'stroke-width="1" stroke-dasharray="2 3" opacity="0.55"/>'
            f'<path d="M {xb-7:.0f} {BAR_Y-11} L {xb+7:.0f} {BAR_Y-11} L {xb:.0f} {BAR_Y-1} Z" '
            f'fill="{color}"/>')
    ticks = ""
    for p in (-3, -2, -1, 0, 1, 2, 3):
        tx = x_of(p)
        ticks += (f'<line x1="{tx:.0f}" y1="{BAR_Y}" x2="{tx:.0f}" y2="{BAR_Y+BAR_H+5}" '
                  f'stroke="rgba(255,255,255,.18)"/>'
                  f'<text x="{tx:.0f}" y="{BAR_Y+BAR_H+24}" fill="#7E93A8" font-size="13" '
                  f'text-anchor="middle" font-family="Orbitron">{p:+d}</text>')
    return f"""<svg viewBox="0 0 680 168" width="100%" style="max-width:660px;height:auto;display:block;margin:0 auto;">
  <defs><linearGradient id="pmvbar" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#00B7D8"/><stop offset="0.5" stop-color="#2ede96"/>
    <stop offset="1" stop-color="#E24B4A"/></linearGradient></defs>
  <rect x="{bx0:.0f}" y="{BAR_Y-5}" width="{bx1-bx0:.0f}" height="{BAR_H+10}" rx="7"
        fill="rgba(46,222,150,.16)" stroke="rgba(46,222,150,.5)" stroke-width="1"/>
  <text x="{(bx0+bx1)/2:.0f}" y="{BAR_Y-12}" fill="#2ede96" font-size="12.5"
        text-anchor="middle" font-family="Rajdhani">舒适带 PPD&lt;10%</text>
  <rect x="{LO}" y="{BAR_Y}" width="{HI-LO}" height="{BAR_H}" rx="10" fill="url(#pmvbar)"/>
  {ticks}{rows}
  <text x="{LO}" y="{BAR_Y-12}" fill="#00B7D8" font-size="12.5" text-anchor="start"
        font-family="Rajdhani">偏冷</text>
  <text x="{HI}" y="{BAR_Y-12}" fill="#E24B4A" font-size="12.5" text-anchor="end"
        font-family="Rajdhani">偏热</text>
</svg>"""


def _pmv_panel(result, scene: SceneInput) -> None:
    st.markdown("#### 🧊 PMV 计算逻辑与可视化")
    logic = (
        "<div class='calc-box'>"
        "<div class='calc-row'>① 六输入:<b>ta</b> 车内空气温度 · <b>MRT</b> 平均辐射温度(日照)"
        " · <b>v</b> 风速(风量档查表) · <b>RH</b> 相对湿度 · <b>met</b> 代谢率(活动) · "
        "<b>clo</b> 服装热阻(衣着)</div>"
        "<div class='calc-row'>② <b>PMV</b> = Fanger(ta, MRT, v, RH, met, clo) "
        "(ISO 7730 热平衡迭代)</div>"
        "<div class='calc-row'>③ <b>PPD</b> = 100 − 95·e^(−0.03353·PMV⁴ − 0.2179·PMV²)</div>"
        "<div class='calc-row'>④ 判读:PMV&lt;0 偏冷 · ≈0 中性最舒适(PPD≈5%) · &gt;0 偏热;"
        "推荐 <b style='color:#2ede96'>PMV∈[−0.5,+0.5]</b></div>"
        "</div>"
    )
    st.markdown(logic, unsafe_allow_html=True)
    _svg_iframe(_pmv_scale(result), height=180)
    # 各区域 PMV 变化曲线(跨推理累积)
    st.markdown("<div class='adjust-head' style='color:#00E5FF'>各区域 PMV 变化曲线"
                "</div>", unsafe_allow_html=True)
    hist = st.session_state.setdefault("pmv_hist", {})
    present = list(result.settings.keys())
    for seat in present:
        m = result.traces[seat].comfort_metrics
        if m:
            hist.setdefault(seat, []).append(round(m.pmv, 2))
            hist[seat] = hist[seat][-40:]
    lens = [len(hist[s]) for s in present if hist.get(s)]
    lmin = min(lens) if lens else 0
    if lmin >= 2:
        series = {_SEAT_ZH[s]: hist[s][-lmin:] for s in present if hist.get(s)}
        st.line_chart(series, height=220)
        st.caption("纵轴 PMV(0 最舒适);随输入变化/多次推理累积。每区独立一条。")
    else:
        st.caption("调节侧栏输入或多次推理后,这里将累积各区域 PMV 变化曲线…")


# --------------------------------------------------------------------------- #
# 侧栏:LLM 状态 + 场景构建
# --------------------------------------------------------------------------- #
def _provider_sidebar() -> None:
    st.sidebar.markdown("### ⚙ 大模型接入")
    s = provider_status()
    badge = "🟢" if s["has_key"] else "🟡"
    st.sidebar.markdown(
        f"{badge} **{s['provider'].upper()}** · `{s['model']}`  \n"
        f"<span style='color:#7E93A8'>状态:{s['mode']}</span>",
        unsafe_allow_html=True,
    )
    if st.sidebar.button("🔌 测试连接", use_container_width=True):
        ok, msg = ping_llm()
        (st.sidebar.success if ok else st.sidebar.warning)(msg)
    st.sidebar.caption("配置:编辑项目根目录 `.env` 填 `DEEPSEEK_API_KEY`,重启生效。")
    st.sidebar.divider()


def _build_scene() -> SceneInput:
    st.sidebar.markdown("### 🌡 车厢环境")
    season = st.sidebar.selectbox("季节", _SEASONS)
    weather = st.sidebar.selectbox("天气", _WEATHER)
    ambient = st.sidebar.slider("车外温度 ℃", -30.0, 50.0, 32.0)
    cabin_t = st.sidebar.slider("车内温度(内温)℃", -20.0, 80.0, 38.0,
                                help="车内温感实测值,作为 PMV 的空气温度 ta")
    humidity = st.sidebar.slider("相对湿度 %", 0.0, 100.0, 55.0)
    sun_d = st.sidebar.slider("主驾日照 W/m²", 0.0, 1500.0, 900.0, 50.0)
    sun_p = st.sidebar.slider("副驾日照 W/m²", 0.0, 1500.0, 700.0, 50.0)
    soc = st.sidebar.slider("电量 %", 0.0, 100.0, 70.0)
    speed = st.sidebar.slider("车速 km/h", 0.0, 200.0, 0.0, 5.0)
    max_defrost = st.sidebar.checkbox("最大除霜功能(MAX)", value=False,
                                      help="开启才进入纯除霜模式 + 最大除风档")
    with st.sidebar.expander("🌫 智能除雾 Agent 输入", expanded=False):
        st.caption("独立除雾 Agent 仅凭以下输入判定起雾(与舒适解耦)。")
        rain_level = st.selectbox("雨量信号", _RAIN,
                                  format_func=lambda x: _RAIN_ZH[x],
                                  help="无/小雨/中雨/大雨;前馈起雾风险")
        ws_air = st.slider("玻璃附近空气温度 ℃", -30.0, 60.0, float(cabin_t))
        ws_glass = st.slider("前风挡玻璃表面温度 ℃", -30.0, 60.0, float(ambient),
                             help="玻璃温度接近/低于露点 → 提前介入除霜")
        st.caption("注:相对湿度滑块即「玻璃附近空气湿度」,与上两项共同决定露点裕度。")
    st.sidebar.caption("ℹ PMV 风速由当前空调风量档位查表;MRT 由内温+日照估算;"
                       "起雾判定 = 独立除雾 Agent(前馈雨量/高湿 + 反馈玻璃温度 vs 露点)。")

    st.sidebar.markdown("### 🧑‍🤝‍🧑 乘员")
    n = st.sidebar.slider("在座人数", 1, 4, 2)
    occupants, windows_open = [], {}
    for i in range(n):
        with st.sidebar.expander(f"座位 {i + 1}", expanded=i == 0):
            seat = st.selectbox("座位", _SEATS, index=i, key=f"seat{i}")
            user = st.text_input("用户", value=["alice", "bob", "carol", "dave"][i],
                                 key=f"user{i}")
            age = st.slider("年龄", 1, 90, 35, key=f"age{i}")
            gender = st.selectbox("性别", _GENDERS, key=f"g{i}")
            height = st.slider("身高 cm", 80, 200, 170, key=f"h{i}")
            weight = st.slider("体重 kg", 10, 130, 65, key=f"w{i}")
            clothing = st.selectbox("衣着", _CLOTHINGS, index=1, key=f"cl{i}")
            emotion = st.selectbox("情绪(喜怒哀乐愁)", _EMOTIONS, key=f"em{i}")
            activity = st.selectbox("热状态/活动", _ACTIVITIES, index=1, key=f"ac{i}")
            win = st.slider("车窗开启 %", 0, 100, 0, key=f"win{i}")
            if win > 0:
                windows_open[seat] = float(win)
            occ = OccupantState(
                seat_id=seat, user_id=user, age=age, gender=gender,
                height_cm=float(height), weight_kg=float(weight), clothing=clothing,
                emotion=emotion, activity=activity)
            st.caption(f"BMI {occ.bmi} · 识别为 {_CAT_ZH[occ.category]} · 衣着 {clothing}")
            occupants.append(occ)

    return SceneInput(
        cabin=CabinContext(ambient_temp=ambient, cabin_temp=cabin_t, weather=weather,
                           humidity=humidity, sun_driver_wm2=sun_d,
                           sun_passenger_wm2=sun_p, soc=soc, speed=speed, season=season,
                           windshield_air_temp=ws_air, windshield_glass_temp=ws_glass,
                           rain_level=rain_level,
                           windows_open=windows_open, max_defrost=max_defrost,
                           timestamp=time.time()),
        occupants=occupants,
    )


# --------------------------------------------------------------------------- #
# 主面板
# --------------------------------------------------------------------------- #
def _header(scene: SceneInput) -> None:
    c = scene.cabin
    chips = (
        f"<span class='chip'>季节 {c.season}</span>"
        f"<span class='chip'>外温 {c.ambient_temp:.0f}℃</span>"
        f"<span class='chip'>日照 主{c.sun_driver_wm2:.0f}/副{c.sun_passenger_wm2:.0f}</span>"
        f"<span class='chip'>湿度 {c.humidity:.0f}%</span>"
        f"<span class='chip'>SOC {c.soc:.0f}%</span>"
    )
    if c.max_defrost:
        chips += "<span class='chip warn'>MAX 除霜</span>"
    if c.any_opening():
        chips += "<span class='chip warn'>门窗开启</span>"
    st.markdown(
        f"""<div class="hud-top">
        <div><div class="hud-title">TMS · COCKPIT AI</div>
        <div class="hud-sub">座舱智慧空调 Agent · 多温区热舒适推理</div></div>
        <div style="flex:1"></div><div style="display:flex;gap:8px;flex-wrap:wrap">{chips}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _send_command(seat: str, scene: SceneInput) -> None:
    """对话/语音指令回调:理解文本 → 调整该座位设定并写入记忆。"""
    eng = st.session_state.engine
    text = (st.session_state.get(f"cmd_{seat}") or "").strip()
    if not text:
        return
    corrected = eng.apply_command(scene, seat, text)
    st.session_state[f"cmd_{seat}"] = ""  # 清空输入框
    st.toast(
        f"{_SEAT_ZH[seat]}「{text}」→ {corrected.temp_set}℃/{corrected.fan_level}档/"
        f"{_MODE_ZH.get(corrected.air_mode, corrected.air_mode)}",
        icon="🗣",
    )


@st.fragment(run_every=1.0)
def _cooldown_chip(seat: str) -> None:
    """卡片上的冷静期倒计时(每秒自刷新该片段,不重算整页)。归零→整页刷新开始逐步逼近。"""
    eng = st.session_state.engine
    rem = int(max(0, (eng.locked_until.get(seat) or 0) - time.time()))
    if rem > 0:
        st.markdown(
            f"<span class='cooldown'>❄ 冷静期剩余 {rem}s · 期间保持你的设定,不自动更改"
            "</span>", unsafe_allow_html=True)
    else:
        st.rerun()  # 冷静期结束:整页刷新,恢复逐步逼近,并停止本计时片段


def _record_override(seat: str, scene: SceneInput, recommended) -> None:
    """控件改动回调:把手动设定自动写入记忆(无需按钮)。只在用户实际改动时触发。"""
    eng = st.session_state.engine
    corrected = ZoneSetting(
        seat_id=seat,
        temp_set=float(st.session_state[f"ct_{seat}"]),
        fan_level=int(st.session_state[f"cf_{seat}"]),
        air_mode=st.session_state[f"cm_{seat}"],
    )
    if (recommended is not None
            and abs(corrected.temp_set - recommended.temp_set) < 1e-6
            and corrected.fan_level == recommended.fan_level
            and corrected.air_mode == recommended.air_mode):
        return  # 与推荐一致,无需记忆
    eng.apply_correction(scene, seat, recommended, corrected)
    st.toast(
        f"{_SEAT_ZH[seat]} 已自动记忆:{corrected.temp_set}℃/{corrected.fan_level}档/"
        f"{_MODE_ZH.get(corrected.air_mode, corrected.air_mode)}",
        icon="🧠",
    )


def _defog_banner(scene: SceneInput, eng: Engine) -> None:
    """独立"智能除雾 Agent"决策横幅(车厢级)。"""
    d = eng.defog_for(scene.cabin)
    diag = d.diagnostics or {}
    if d.need_defog:
        margin = diag.get("margin")
        mtxt = f" · 玻璃裕度 {margin}℃" if margin is not None else ""
        st.warning(
            f"🌫 **智能除雾 Agent**:{_DEFOG_ZH[d.level]}(来源 {d.source})"
            f" · 雨量 {_RAIN_ZH.get(scene.cabin.rain_level, scene.cabin.rain_level)}"
            f" · 玻璃附近湿度 {scene.cabin.humidity:.0f}%{mtxt}  \n"
            f"→ 舒适 Agent 出风模式将叠加除霜;{d.reasoning}"
        )
    else:
        st.success(
            f"🌫 **智能除雾 Agent**:{_DEFOG_ZH[d.level]}(来源 {d.source})"
            f" · 雨量 {_RAIN_ZH.get(scene.cabin.rain_level, scene.cabin.rain_level)}"
            f" · 不叠加除霜,保持舒适基础出风模式"
        )


def _dashboard(scene: SceneInput, eng: Engine) -> None:
    result = eng.infer(scene)
    _defog_banner(scene, eng)
    left, right = st.columns([5, 6], gap="large")
    with left:
        _svg_iframe(_cabin_hud(result, scene), height=650)
        st.divider()
        _pmv_panel(result, scene)
    with right:
        for seat, s in result.settings.items():
            t = result.traces[seat]
            m = t.comfort_metrics
            occ = next((o for o in scene.occupants if o.seat_id == seat), None)
            color = _temp_color(s.temp_set)
            with st.container(border=True):
                head = (f"<span style='font-family:Orbitron;color:{color};"
                        f"font-size:15px;letter-spacing:1px'>{_SEAT_ZH[seat]}</span>"
                        f" · {t.user_id}")
                if occ:
                    head += (f" <span class='readout'>{_CAT_ZH[occ.category]}/"
                             f"{occ.age}/BMI{occ.bmi}/{_ACT_ZH[occ.activity]}</span>")
                st.markdown(head, unsafe_allow_html=True)
                st.text_input(
                    f"🎙 {_SEAT_ZH[seat]}语音/对话输入(语音转文本,直接说人话)",
                    key=f"cmd_{seat}",
                    placeholder="如:太冷了 / 天太热了 / 风太大 / 调到22度 / 吹脚",
                    on_change=_send_command, args=(seat, scene),
                )
                st.caption("↑ 输入后回车即生效:系统理解诉求 → 调整该座位设定 → 写入记忆")
                k1, k2, k3 = st.columns(3)
                k1.metric("温度", f"{s.temp_set}℃")
                k2.metric("风量", f"{s.fan_level}/7")
                k3.metric("出风", _MODE_ZH.get(s.air_mode, s.air_mode))
                if m:
                    badge = ("已自动应用" if result.applied[seat] else "维持/锁定")
                    st.markdown(
                        f"<span class='readout'>车内≈{m.cabin_temp}℃(估) · PMV {m.pmv} · "
                        f"PPD {m.ppd}% · 目标 {m.target_temp}℃ · 来源 {t.source}"
                        + (f" · 逼近 {t.approach_weight:.2f}" if t.approach_weight > 0 else "")
                        + f" · {badge}</span>", unsafe_allow_html=True)
                # 冷静期倒计时(仅当该座位处于手动覆盖锁定窗口内时显示并自动倒数)
                if int((eng.locked_until.get(seat) or 0) - time.time()) > 0:
                    _cooldown_chip(seat)
                if t.knowledge_snippets:
                    st.caption("📚 " + " ｜ ".join(t.knowledge_snippets[:2]))
                for adj in t.safety_adjustments:
                    st.caption("🛡 " + adj)
                # 学习状态:让"记忆学习确实在生效"可见(证据/偏好/逼近进度)
                if t.memory_evidence > 0 and t.learned_preference:
                    lp = t.learned_preference
                    extra = (f" · 逼近中 {t.approach_weight:.0%}"
                             if t.approach_weight > 0 else " · 已逼近偏好")
                    st.markdown(
                        f"<span class='learn-badge'>🧠 已学习 {t.memory_evidence} 次 · "
                        f"偏好 {lp['temp']}℃/{lp['fan']}档{extra}</span>",
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<span class='learn-badge dim'>🧠 该用户暂无记忆 · "
                        "调下方设定或说一句即开始学习</span>", unsafe_allow_html=True)
                if t.assumptions:
                    with st.expander("ⓘ 估算与假设(PMV 输入透明化)"):
                        for a in t.assumptions:
                            st.caption("• " + a)
                # 手动调节(常显,不折叠):改动任一项即记为用户主动修正 → 写入记忆并逐步逼近
                st.markdown(
                    f"<div class='adjust-head'>✎ 手动调节 {_SEAT_ZH[seat]}(改动即记忆学习)"
                    "</div>", unsafe_allow_html=True)
                # 预置控件值为当前生效设定;让控件跟随系统,用户一改动即写入记忆
                st.session_state[f"ct_{seat}"] = float(s.temp_set)
                st.session_state[f"cf_{seat}"] = int(s.fan_level)
                st.session_state[f"cm_{seat}"] = s.air_mode
                args = (seat, scene, t.final)  # t.final = 系统推荐,作为修正基准
                a1, a2 = st.columns(2)
                with a1:
                    st.number_input("温度 ℃", 15.5, 31.5, step=0.5, key=f"ct_{seat}",
                                    on_change=_record_override, args=args)
                with a2:
                    st.slider("风量", 1, 7, key=f"cf_{seat}",
                              on_change=_record_override, args=args)
                st.selectbox("出风模式", _AIR_MODES, key=f"cm_{seat}",
                             on_change=_record_override, args=args)
                st.caption("调整任一项 → 立即作为主动修正写入记忆,系统下次相似场景持续逼近")
    st.metric(
        "会话修正率(越低越贴合)", f"{eng.metrics.correction_rate:.2f}",
        help="= 用户修正次数 ÷ 系统推荐次数。衡量推荐与你需求的贴合度:你手动/语音改得越少,"
             "说明 Agent 推得越准,该值越低越好;随记忆学习生效会逐步下降。",
    )


_STEP_ICON = {"featurize": "🧩", "recall": "🧠", "comfort": "📐",
              "llm_infer": "🤖", "approach": "🎯", "safety": "🛡", "final": "✅",
              "sense": "🪟", "knowledge": "📚", "decide": "🌫"}


def _render_comfort_calc(bd: dict) -> str:
    """车内温度 / PMV / PPD 计算过程可视化(展示公式与代入数值)。"""
    if bd.get("ta_measured"):
        ta_line = f"车内温度 ta = <b>{bd['ta']}℃</b>(实测内温)"
    else:
        ta_line = (f"车内温度 ta = 车外 {bd['ambient']:.0f}℃ + 日照升温 = "
                   f"<b>{bd['ta']}℃</b>(热模型估算)")
    rows = [
        ta_line,
        f"平均辐射温度 MRT = ta + 日照增量 {bd['mrt_gain']} = <b>{bd['mrt']}℃</b>"
        f"(日照 {bd['sun']:.0f} W/m²)",
        f"风速 v = 风量 {bd['fan']} 档 → <b>{bd['velocity']} m/s</b>(查表)",
        f"clo = {bd['clo']}(衣着 {bd['clothing']}) · met = {bd['met']}(活动 "
        f"{bd['activity']}) · RH = {bd['rh']:.0f}%",
        f"<b>PMV</b> = Fanger(ta, MRT, v, RH, met, clo) = "
        f"<b style='color:#FF8A3D'>{bd['pmv']}</b>",
        f"<b>PPD</b> = 100 − 95·e^(−0.03353·PMV⁴ − 0.2179·PMV²) = "
        f"<b style='color:#FF8A3D'>{bd['ppd']}%</b>",
        f"目标温度 = 锚点 {bd['base_target']} + 人体偏移 {bd['offset']:+} = "
        f"<b style='color:#00E5FF'>{bd['target']}℃</b>",
    ]
    tr = bd.get("transient")
    if tr:
        phase_zh = {"cooldown": "制冷过渡(快速降温)", "warmup": "制热过渡(快速升温)",
                    "steady": "稳态(回归舒适+最低风量/NVH)"}.get(tr["phase"], tr["phase"])
        rows.append(
            f"瞬态控制:负荷 {tr['load']:+}℃ → <b>{phase_zh}</b> → "
            f"建议设定 <b style='color:#FF8A3D'>{tr['setpoint']}℃</b> / 风量 "
            f"<b style='color:#FF8A3D'>{tr['fan']}档</b>"
        )
    inner = "".join(f"<div class='calc-row'>{i + 1}. {r}</div>"
                    for i, r in enumerate(rows))
    return f"<div class='calc-box'>{inner}</div>"


def _render_feature_detail(d: dict) -> str:
    rows = "".join(f"<div class='calc-row'>{k}:<b>{v}</b></div>" for k, v in d.items())
    return f"<div class='calc-box'>{rows}</div>"


def _defog_chain(scene: SceneInput, eng: Engine) -> None:
    """独立除雾 Agent 推理链(车厢级,与舒适 Agent 并列展示多 Agent 协作)。"""
    st.markdown("#### 🌫 智能除雾 Agent 推理链(车厢级)")
    st.caption("独立 Agent:仅凭玻璃温度 / 玻璃附近空气温度 / 玻璃附近湿度 / 雨量信号判定,"
               "结论叠加到下方各座位舒适出风模式。")
    for step in eng.stream_defog(scene.cabin):
        icon = _STEP_ICON.get(step.node, "•")
        cls = "chain-line chain-final" if step.node == "final" else "chain-line"
        st.markdown(
            f"<div class='{cls}'>{icon} <span class='chain-node'>{step.title}"
            f"</span><br><span class='readout'>{step.detail}</span></div>",
            unsafe_allow_html=True)
        if step.node == "sense" and step.data:
            with st.expander("🪟 除雾输入特征"):
                st.markdown(_render_feature_detail(step.data),
                            unsafe_allow_html=True)
    st.divider()


def _chain_panel(scene: SceneInput, eng: Engine) -> None:
    st.markdown("#### ⚡ 实时推理链(多 Agent)")
    st.caption("随场景实时更新。除雾 Agent(车厢级)与舒适 Agent(按座位)并行协作;"
               "展开「③ 舒适计算」可见车内温度 / PMV / PPD 的完整计算过程。")
    _defog_chain(scene, eng)
    st.markdown("##### 🌡 舒适 Agent 推理链(按座位)")
    occs = scene.present_occupants()
    cols = st.columns(len(occs))
    for col, occ in zip(cols, occs):
        with col:
            st.markdown(f"##### 🚗 {_SEAT_ZH[occ.seat_id]} · {occ.user_id} "
                        f"({_CAT_ZH[occ.category]}/{occ.gender}/{occ.age})")
            for step in eng.stream_seat(scene, occ, apply=False):
                icon = _STEP_ICON.get(step.node, "•")
                cls = "chain-line chain-final" if step.node == "final" else "chain-line"
                st.markdown(
                    f"<div class='{cls}'>{icon} <span class='chain-node'>{step.title}"
                    f"</span><br><span class='readout'>{step.detail}</span></div>",
                    unsafe_allow_html=True)
                if step.node == "featurize" and step.data:
                    with st.expander("🧩 全部输入特征"):
                        st.markdown(_render_feature_detail(step.data),
                                    unsafe_allow_html=True)
                if step.node == "comfort" and step.data:
                    with st.expander("📐 计算过程:车内温度 / PMV / PPD"):
                        st.markdown(_render_comfort_calc(step.data),
                                    unsafe_allow_html=True)


def _render_memory_chain(rec) -> str:
    """渲染一条完整记忆链条:人员/车辆/环境 输入 → 推理 → 用户修正。"""
    cab, occ = rec.cabin, rec.occupant
    if cab is not None:
        env = (f"季节 {cab.season} · 外温 {cab.ambient_temp:.0f}℃ · "
               f"内温 {('%.0f' % cab.cabin_temp+'℃') if cab.cabin_temp is not None else '估算'} · "
               f"日照 {cab.seat_sun(rec.seat_id):.0f}W/m² · 湿度 {cab.humidity:.0f}% · "
               f"天气 {cab.weather} · 雨量 {_RAIN_ZH.get(cab.rain_level, cab.rain_level)}")
        veh = (f"电量 {cab.soc:.0f}% · 车速 {cab.speed:.0f}km/h · "
               f"门窗 {'有开启' if cab.any_opening() else '关闭'} · "
               f"MAX除霜 {'开' if cab.max_defrost else '关'}")
    else:
        env = veh = "(旧记录无完整快照)"
    if occ is not None:
        per = (f"{_CAT_ZH.get(occ.category, occ.category)}/{occ.gender}/{occ.age}岁 · "
               f"BMI {occ.bmi} · 衣着 {occ.clothing} · "
               f"活动 {_ACT_ZH.get(occ.activity, occ.activity)} · 情绪 {occ.emotion}")
    else:
        per = "(旧记录无人员快照)"
    rc, cc = rec.recommended, rec.corrected
    rows = [
        f"🌡 环境:{env}",
        f"🚗 车辆:{veh}",
        f"🧑 人员:{per}",
        f"🤖 推理:{rc.temp_set}℃ / {rc.fan_level}档 / {_MODE_ZH.get(rc.air_mode, rc.air_mode)}",
        f"✋ 修正:<b style='color:#FF8A3D'>{cc.temp_set}℃ / {cc.fan_level}档 / "
        f"{_MODE_ZH.get(cc.air_mode, cc.air_mode)}</b>",
    ]
    inner = "".join(f"<div class='calc-row'>{r}</div>" for r in rows)
    return f"<div class='calc-box'>{inner}</div>"


def _memory_panel(eng: Engine) -> None:
    st.markdown("#### 🧠 学习记忆(完整链条:输入 → 推理 → 修正)")
    st.caption("每条记忆都记录:在何种人员/车辆/环境输入下,空调推理出什么、用户最终调成什么。"
               "下次相似场景据此逐步逼近。")
    summary = eng.store.summary()
    if not summary:
        st.info("暂无学习记忆。在「座舱总览」里调节设定或说一句指令后再来看。")
    else:
        for (user, seat), cnt in sorted(summary.items()):
            cols = st.columns([4, 1])
            cols[0].markdown(f"**{user}** × {_SEAT_ZH.get(seat, seat)} · {cnt} 条修正")
            if cols[1].button("删除", key=f"del{user}{seat}", use_container_width=True):
                eng.store.delete(user, seat)
                st.rerun()
            recs = eng.store.records_for(user, seat)
            with st.expander(f"查看 {user}×{_SEAT_ZH.get(seat, seat)} 的 {cnt} 条记忆链条"):
                for i, rec in enumerate(reversed(recs), 1):  # 最新在前
                    st.markdown(f"**#{cnt - i + 1}**", unsafe_allow_html=True)
                    st.markdown(_render_memory_chain(rec), unsafe_allow_html=True)
    if st.button("清空所有记忆", use_container_width=True):
        eng.store.reset_all()
        st.rerun()


def _engine() -> Engine:
    if "engine" not in st.session_state:
        st.session_state.engine = Engine()
    return st.session_state.engine


def main() -> None:
    st.set_page_config(page_title="TMS · Cockpit AI", layout="wide", page_icon="❄")
    _inject_css()
    eng = _engine()
    _provider_sidebar()
    scene = _build_scene()
    _header(scene)

    tab1, tab2, tab3 = st.tabs(["座舱总览", "实时推理链", "学习记忆"])
    with tab1:
        _dashboard(scene, eng)
    with tab2:
        _chain_panel(scene, eng)
    with tab3:
        _memory_panel(eng)


if __name__ == "__main__":
    main()
