---
name: tms-cockpit-hvac
description: >-
  Use when working on the TMS cockpit smart-HVAC agent (this repo,
  E:/AI+热管理项目实战/TMS_Agent_Workflow) or any automotive cabin thermal-comfort /
  air-conditioning reasoning task. Covers ISO 7730 PMV/PPD, the project's input model,
  transient cooldown control, the standalone defog agent (multi-agent), multi-zone
  independent prediction, gradual-approach memory learning, and exactly how to extend the
  project's knowledge / tools / skills without breaking conventions. Trigger on: 座舱空调/
  热舒适/PMV/PPD/除雾/起雾/风量温度推荐/记忆逼近, or editing files under tms_agent/.
---

# 座舱智慧空调 Agent · 专家技能

本技能封装"汽车座舱热舒适专业知识 + 本项目工程规范"。在改动本仓库或处理座舱热管理
推理任务时遵循本技能。

## 1. 何时使用

- 在 `tms_agent/` 下增改功能(知识/工具/技能/图节点/记忆/UI)。
- 需要基于热舒适给出/解释空调风量、温度、出风模式设定。
- 涉及 PMV/PPD、除雾、瞬态降温、记忆逼近、多温区等。

## 2. 领域核心(设定空调参数的依据)

- **PMV/PPD(ISO 7730)**:目标 PMV≈0(PPD 最低≈5%),推荐 PMV∈[-0.5,+0.5](PPD<10%)。
  六输入:空气温度 ta、平均辐射温度 MRT、风速、相对湿度、代谢率 met、服装热阻 clo。
- **本项目输入来源(诚实对待,写入 DecisionTrace.assumptions)**:
  ta=车内温度实测(缺失→车外温+日照热模型估算);风速=当前风量档查表;clo=衣着映射;
  met=活动映射;MRT=车内温+日照估算(唯一纯估算项);RH=实测。
- **出风模式 = 3 基础 + 除霜叠加**:夏/制冷→吹面 face,冬/制热→吹脚 feet(头凉脚暖),
  过渡→吹面吹脚 face_feet;起雾风险→在基础上叠加除霜;纯除霜仅"最大除霜功能"开启时。
- **瞬态控制(快速降温→稳态)**:车内远离目标→设定更激进(制冷更低/制热更高)+大风量;
  逐步接近→设定回调向目标+降风量;稳态→最舒适目标温度+低档风量(常规 2~3 档,默认 fan_steady_min=2,兼顾 NVH)。
  见 `tools/thermal_comfort.transient_setpoint_fan`。
- **起雾=独立「智能除雾 Agent」(多 Agent)**:已从舒适侧拆出为车厢级独立 LangGraph(`defog/agent.py`:
  sense→knowledge→decide),仅凭**玻璃表面温度 / 玻璃附近空气温度 / 玻璃附近湿度 / 雨量信号 `rain_level`** 判定。
  前馈=雨量/高湿,反馈=玻璃温度 vs 露点裕度(≤2℃强、≤4℃轻度);安全取严(规则与 LLM 取更紧急者)。
  舒适 Agent 出基础模式,`safety.apply_safety` 消费 `DefogDecision` 叠加除霜(strong 级保证风量下限)。
  专属工具 `defog/tools.py`、专属知识库 `defog/docs/`、专属决策器 `defog/decider.py`。
- **人群/状态微调**:老人/小孩/婴儿偏暖、柔和、不直吹;睡眠安静低风量略升温;兴奋/运动后偏凉;
  女性略暖、高 BMI 略凉;低电量节能(收敛设定、降风量、多内循环)。
- **设定域(硬约束)**:温度 15.5–31.5℃/0.5;风量 1–7;7 出风模式。

## 3. 记忆 / 逐步逼近(人调节后如何学习)

- 手动/语音修改 → 作为一次修正写入 (user×seat) 记忆,进入冷静期、重置逼近游标。
- 冷静期外的后续推理由 `approach` 节点**游标式逐步逼近**:差值越大首步越大,贴齐网格,
  2-3 步收敛(例:风量 7→5→4→3),绝不一步跳变。
- 方向一致才累计证据;矛盾不学;旧记忆半衰期老化;季节隔离;用户×座位独立。
- **坑**:防抖 `FAN_DEADBAND=0`、`TEMP_DEADBAND=0.4`,否则单步过渡被吃掉。

## 4. 如何扩展本项目(扩展点,加东西只在这几处)

- **加舒适知识**:在 `tms_agent/knowledge/docs/` 放 `.json`(结构化 `{id,title,tags,text}`)
  或 `.md`(整文件一条 chunk,标题取首个标题行),重启即被 sklearn TF-IDF + chromadb 索引。
- **加舒适工具**:在 `tools/thermal_comfort.py` 写纯函数 + 加进 `THERMAL_TOOLS`(StructuredTool)。
- **加运行时技能**:在 `skills/registry.py` 继承 `Skill` 实现 `invoke(context)->dict`,
  在 `build_default_registry` 注册。这是运行时能力(执行 Python),非提示型 Markdown skill。
- **改/扩除雾 Agent**:工具加在 `defog/tools.py`、知识加在 `defog/docs/*.json`、
  判定逻辑在 `defog/decider.py` / `defog/agent.py`,阈值在 `config.DEFOG`。与舒适侧解耦,勿混。

## 5. 工程约束(改代码前必读)

- Python 3.14;已启用 scikit-learn + chromadb。
- LLM 直接输出 JSON 自解析(勿用 with_structured_output:DeepSeek 拒 json_schema/tool_choice);
  无 key 退回 MockDecider;超时降级;结果按 payload 缓存。默认模型 `deepseek-v4-flash`(小写)。
- `now` 必须贯穿 engine.infer→graph state→recall→store.recall(时间衰减口径)。
- 阈值集中在 `config.py`(`Thresholds` + `DefogThresholds`);不可变风格(节点返回 state 增量);注释用中文。
- 中文引号在 JSON/Python 字符串内必须用「」,误用 ASCII " 会破坏 JSON/语法。
- 舒适图序列:featurize→recall→comfort→llm_infer→approach→safety;除雾图序列:sense→knowledge→decide。
- **LLM payload 不含 seat_id**(热舒适与座位无关,否则同输入异输出);**总览与推理链用同一次
  `infer(capture_chain=True)`**(勿分别推理,避免 last_applied/游标副作用致不一致)。
- **Web 富 SVG 用 `st.components.v1.html`(iframe)渲染**,勿用 `st.markdown`(会被 Markdown 段落化截断)。

## 6. 运行与验证

```bash
.venv/Scripts/python -m pytest -q                          # 全量测试(应全绿)
.venv/Scripts/python -m streamlit run tms_agent/app_web.py # 座舱 HMI(http://127.0.0.1:8501)
.venv/Scripts/python -m tms_agent.app_cli teach 0          # CLI 闭环演示
```

## 7. 关键文件

`config.py`(阈值/枚举/查表 + `DefogThresholds`)、`schemas.py`(数据模型+兜底,含 `DefogDecision`/`rain_level`)、
`tools/thermal_comfort.py`(PMV/PPD/EQT/目标温度/瞬态/露点/逼近步)、`knowledge/`(舒适 RAG)、`skills/registry.py`、
`llm/provider.py`(provider+NLU interpret)、`nlu.py`、`graph/`、`safety.py`(消费 `DefogDecision` 叠加)、
`memory/store.py`、`engine.py`(`defog_for`/`stream_defog`)、`app_web.py`/`app_cli.py`;
**独立除雾 Agent:`defog/`(`agent.py` 图、`tools.py` 工具、`decider.py` 决策、`docs/defog_kb.json` 知识)**。
方案见 PLAN.md,硬约束见 CLAUDE.md。
