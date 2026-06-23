# CLAUDE.md — 座舱智慧空调 Agent 开发需求(权威约束)

> 本文件固化开发需求与硬约束,供每次会话自动加载,避免上下文压缩丢失。
> 详细方案见 [PLAN.md](PLAN.md),使用说明见 [README.md](README.md)。改动需求时更新本文件。

## 产品目标

基于 LangGraph + LangChain 的座舱热感 Agent:输入环境/人员/车辆/天气,经**专业热舒适
知识(ISO 7730 / ASHRAE 55)约束的 LLM** 推理,为**每位乘员独立**给出空调设定;用户
手动修改被记忆,**下次相似场景持续逼近用户偏好**。本期为 PC/Python PoC,架构为产品化预留接口。

## 输入模型(细粒度)与 PMV 输入来源(务必诚实对待)

- **环境(CabinContext)**:**车外温度**、相对湿度(=玻璃附近空气湿度)、车速、SOC、天气、季节;
  **太阳辐照 `sun_driver_wm2` / `sun_passenger_wm2`(仅主驾/副驾两路,W/m² 0–1500;后排按同侧前排取值,见 `seat_sun()`)**;
  四门 `doors_open`、四窗 `windows_open`(0–100%);`max_defrost`。
- **智能除雾 Agent 专属输入**:`windshield_glass_temp`(玻璃表面温度)、`windshield_air_temp`(玻璃附近空气温度)、
  `humidity`(玻璃附近空气湿度)、`rain_level`(雨量信号:none/light/moderate/heavy = 无/小雨/中雨/大雨)。
- **车内温度 `cabin_temp` 是输入(实测优先)**;缺失/NaN 时才由 `estimate_cabin_temp(车外温,座位日照,门窗,车速)` 热模型回退估算(见 `seat_air_temp`)。
- **人员(OccupantState,每座位)**:年龄、性别、身高、体重、BMI(派生)、
  **衣着 clothing(light/medium/heavy)**、情绪 emotion(喜怒哀乐愁)、活动 activity(兴奋/平静/睡眠);识别小孩/大人/老人(category)。**无主观冷热输入,冷热由 PMV 计算。**
- **PMV 六输入来源(透明化,写入 `DecisionTrace.assumptions`)**:
  ta=车内温度(**实测**,缺失则估算)、MRT=车内温+日照估算、**风速=当前空调风量档查表**(`FAN_VELOCITY_TABLE`)、
  RH=实测、met=活动映射、**clo=衣着映射(实测量)**。唯一纯估算项为 MRT。
- 接入:活动→met;衣着→clo;人体特征(老人/小孩/性别/BMI)→目标温度偏移(`comfort_offset`);
  情绪/门窗→注入 LLM 与知识检索(不做确定性公式)。
- **诚实原则**:PMV/PPD/EQT 是"基于上述估算/假设的模型值",不得当作实测真值表述;UI/CLI 需展示估算车内温与 assumptions。

## 硬约束:设定域(务必遵守)

- **温度**:15.5 ~ 31.5 ℃,精度 **0.5℃**。
- **风量**:**1 ~ 7 档**(无 0 档),精度 1。
- **出风模式 = 7 种 = 3 个舒适基础模式 + 除霜叠加**:
  - 舒适基础(由舒适推理在三者间选):`face` 吹面(夏季制冷)、`feet` 吹脚(冬季制热)、
    `face_feet` 吹面吹脚(春秋过渡)。
  - **除霜叠加由独立「智能除雾 Agent」判定**(`defog/`),安全层消费其 `DefogDecision` 叠加,
    不由舒适推理输出:`face_defrost` / `face_feet_defrost` / `feet_defrost`。
  - **纯除霜 `defrost` 仅当显式开启「最大除霜功能」**(`CabinContext.max_defrost=True`,
    对应车上 MAX 除霜按键)时进入,并强制风量=7;**除雾 Agent 绝不输出纯除霜**。
  - LLM/MockDecider **只输出 3 个基础模式**;7 态由 `safety.apply_safety` 落定。

## 硬约束:多 Agent —— 智能除雾 Agent(独立于舒适 Agent)

- **职责分离**:舒适 Agent(按座位 LangGraph)出三项基础设定;**智能除雾 Agent(车厢级,独立 LangGraph
  `defog/agent.py`:sense→knowledge→decide)** 仅凭**玻璃表面温度 / 玻璃附近空气温度 / 玻璃附近湿度 / 雨量信号**
  判定是否除雾及紧急度(none/mild/strong)。
- **判定 = 前馈(雨量/高湿)+ 反馈(玻璃温度 vs 露点裕度)**;裕度 ≤2℃→强、≤4℃→轻度(阈值在 `config.DEFOG`)。
- **叠加合成**:除雾 Agent 判 `need_defog` → 舒适基础模式叠加除霜;`strong` 级保证风量 ≥ `DEFOG.fan_floor_strong`(=4)。
- **安全取严**:确定性物理规则与 LLM 取**更紧急者**,除雾强度不低于规则下限(`defog/decider.py`)。
- **专属配置**:工具 `defog/tools.py`、知识库 `defog/docs/defog_kb.json`、决策器 `defog/decider.py`,均独立于舒适侧。
- **接入**:`engine.defog_for(cabin)` 按车厢指纹缓存(多座位/流式复用同一决策),注入各座位图 state['defog']。
- 除雾图序列:`sense → knowledge → decide`。

## 硬约束:学习策略 = 逐步逼近(游标式,非硬切换)

- 一次修正建立偏好(写入 user×seat 记忆)、重置游标、进入冷静期。
- **冷静期(600s,保留)外的每次推理**:`approach` 节点从游标(首次=专业基准)按
  `tools/thermal_comfort.approach_step` 比例(`APPROACH_RATE`=0.5)朝偏好迈一步、贴齐网格
  (温度 0.5℃ / 风量 1 档),**差值越大首步越大,2-3 步到位**(例:风量 7→5→4→3),绝不一步跳变。
- 游标在 `engine.approach_cursor[(user,seat)]` 跨推理持久化,**仅非冷静期推进**;修正时重置。
- **LLM 实时推理 + 确定性兜底**:payload 注入用户偏好 + 逐步逼近提示,知识库有
  `gradual_preference_approach` 等条目(has_preference 时检索);approach 节点为确定性兜底保证收敛。
- **关键坑**:防抖 `FAN_DEADBAND=0`、`TEMP_DEADBAND=0.4`,否则单档/单步过渡值被防抖吃掉。
- 记忆按 **(user×seat)** 独立;矛盾修正不产生一致证据;旧记忆按半衰期老化;证据计数用整数条数。
- **每条记忆存完整链条**(`CorrectionRecord`):输入快照(人员 `occupant` + 车辆/环境 `cabin`)+ 归一化特征
  `scene_vector` + 系统推理 `recommended` + 用户修正 `corrected`。语音与手动调节**都**算用户主动修正,
  统一经 `engine.apply_correction` 写入并落盘;Web「学习记忆」页可展开查看每条链条。
- 图序列:`featurize → recall → comfort → llm_infer → approach → safety`。

## 决策优先级

安全(起雾强制/叠加除霜)> 用户偏好持续逼近 > 专业舒适锚点(PMV≈0 目标温度)> LLM 自由决策。

## 硬约束:瞬态控制(温度/风量随负荷,必须遵循)

- **夏季/制冷**:车内离目标越远(冷负荷越大)→ 设定温度越低、风量越大;趋近稳态 → 设定回升到
  最舒适目标、风量降低兼顾 NVH。**冬季/制热对称**:车内越冷 → 设定越高、风量越大;趋稳 → 回落到
  舒适目标、风量降低。**稳态风量落脚点 = 低档(常规 2~3 档,默认 `fan_steady_min=2`,不取过弱的 1 档);
  仅当用户个性化记忆显示其稳态习惯 1 档时,由逐步逼近学到 1 档。**
- 实现:`tools/thermal_comfort.transient_setpoint_fan(cabin_temp,target)`;`ThermalComfortSkill`
  在 comfort 节点算出 `transient`{setpoint,fan,phase,load} → 注入 `state['transient']` →
  `_build_payload` 的 `transient_recommendation`。
- **关键坑(已修)**:此前仅 MockDecider 用它,**CloudDecider(LLM)提示词未含瞬态、反而要求贴近锚点**,
  接真实 Key 时不遵循。现 `_SYSTEM_PROMPT` 强制按 `transient_recommendation` 同向给出 temp/fan;
  MockDecider 也改用注入的 transient,口径统一。回归测试 `test_transient_setpoint_fan_direction`。

## 其他确认决策

- 多乘员:**每乘员独立预测 → 多温区**,每位独立一组三项设定与独立记忆。
- 触发:30s 周期 + 输入显著变化事件(`runtime/triggers.py`)。
- 输出:自动应用 + 用户可覆盖(覆盖即记忆并进入冷静期);防抖迟滞抑制抖动。
- LLM:provider 可配置,默认 DeepSeek(模型名小写 `deepseek-v4-flash`);**直接输出 JSON 自解析**
  (不用 with_structured_output:json_schema/tool_choice 被 DeepSeek 拒);无 key 退回 MockDecider;
  超时/失败走降级链(云→Mock→沿用上次+舒适锚点);LLM 结果按 payload 内容缓存。
- 知识:计算工具 + RAG(**sklearn TfidfVectorizer + chromadb 向量库**,`knowledge/docs/*.json` 可导入)。
- **语音/对话输入**:接收文本(语音转写在外部),每座位一个入口;`engine.apply_command(scene,seat,text)`
  经 `nlu.interpret`(LLM `CloudDecider.interpret` + 关键词 `rule_interpret` 兜底)→ 修正并记忆。
- **预装能力**:Skills = ThermalComfort/Knowledge/Strategy/Energy/Weather/Vehicle/Memory;
  Tools 增 fog_risk/dew_point/comfort_temp_band/recirculation_hint/energy_hint;知识库共 66 条(舒适 56 + 除雾 10)。
- **除雾 Agent 专属能力**:Tools = `defog/tools.py`(dew_point_margin/rain_fog_pressure/defog_urgency);
  知识库 = `defog/docs/defog_kb.json`(10 条除雾原理/前馈反馈/雨量分级/裕度策略/气流除湿);决策器 = `defog/decider.py`。

## 技术约束(改代码前必读)

1. **Python 3.14 环境**。scikit-learn / chromadb 已验证可安装并启用:知识检索用
   sklearn ``TfidfVectorizer`` 向量化 + chromadb 向量库(见 `knowledge/retriever.py`)。
2. `now` 必须贯穿 `engine.infer → graph state['now'] → recall → store.recall`,否则时间衰减口径错乱。
3. 所有阈值集中在 `config.py` 的 `Thresholds`/`DefogThresholds`/`ComfortDefaults`,调参改这里,勿散落。
4. 不可变风格:节点返回 state 增量,记忆记录只增不改。
5. 代码注释用中文,与现有代码库一致。
6. **LLM payload 不含 `seat_id`**:热舒适只取决于物理输入,与座位无关;否则主/副驾同输入也会被当成
   不同请求(且 LLM 温度>0 非确定)→ 同输入异输出。见 `graph/nodes._build_payload`。
7. **总览与推理链必须同源**:Web 用 `engine.infer(capture_chain=True)` 一次推理同时喂两处,
   勿分别推理(`infer` 有 `last_applied`/游标副作用,二次推理会导致温度不一致)。
8. **Web 富 SVG 用 `st.components.v1.html`(iframe)渲染**,勿用 `st.markdown`——多行 SVG 会被
   Markdown 段落化截断(座椅/文字被丢弃),且滤镜/渐变/动画需 iframe 内联样式。诊断用 Playwright
   `iframe.contentDocument` 查 svg 子元素数。

## 运行与验证

```bash
.venv/Scripts/python -m pytest -q                       # 全量测试(应全绿)
.venv/Scripts/python -m tms_agent.app_cli teach 0       # CLI 闭环演示
.venv/Scripts/python -m streamlit run tms_agent/app_web.py
```
接云端 LLM:复制 `.env.example`→`.env` 填 `DEEPSEEK_API_KEY`。

## 非目标(本期不做,已用接口隔离)

真实车辆信号(CAN/SOA)、域控嵌入式部署、本地小模型、**语音识别本身(只接收转写文本)**、
云端记忆同步、执行器扩展(内外循环/AC开关/座椅加热的实际控制)、多模态热舒适感知。
