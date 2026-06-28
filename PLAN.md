# 座舱智慧空调 Agent · 方案与实现(as-built)

> 工作空间权威方案文档,随实现更新。硬约束见 [CLAUDE.md](CLAUDE.md),使用见 [README.md](README.md)。
> 当前状态:PoC 完成,97 项测试全绿,Python 3.14。

## 1. 目标

基于 LangGraph + LangChain 的座舱热感 Agent:输入环境/人员/车辆/天气,经专业热舒适知识
约束的 LLM 实时推理,为每位乘员独立给出风量/温度/出风模式;用户手动或语音修改被记忆,
相似场景**逐步逼近**用户偏好。核心价值 = 专业约束 + 记忆闭环 + 实时推理。

## 2. 关键决策(全部已确认)

| 决策点 | 选择 |
|--------|------|
| 目标 | PoC(PC/Python),Mock 数据,模块化预留产品化接口 |
| 触发 | 30s 周期 + 输入显著变化事件 |
| LLM | provider 可配置,默认 DeepSeek;直接输出 JSON 自解析(兼容思考/普通模型);无 key 离线 Mock;超时降级 + 结果缓存 |
| 多乘员 | 每乘员独立预测 → 多温区;按 用户×座位 独立记忆 |
| 学习 | 逐步逼近:一次修正后,冷静期外后续推理沿专业推荐→偏好游标式迭代收敛(差值越大首步越大,2-3 步到位) |
| 偏好优先级 | 安全 > 用户偏好逐步逼近 > 专业舒适锚点 > LLM |
| 设定域 | 温度 15.5–31.5℃/0.5;风量 1–7;7 出风模式 = 3 基础 + 除霜叠加;纯除霜仅最大除霜开关 |
| 输入 | 车外/车内温度、湿度、主副驾两路日照 W/m²、车速、SOC、天气、季节、四门四窗、最大除霜;乘员 年龄/性别/身高/体重/BMI/衣着/情绪/活动 |
| 专业知识 | 计算工具 + RAG(sklearn TF-IDF + chromadb 向量库);66 条知识(舒适 56 + 除雾 10) |
| 语音 | 接收文本(语音转写在外部完成),每座位一个对话入口,NLU 理解 → 修正并记忆 |
| 鲁棒性 | 降级链 + 缓存 + 防抖 + 冷静期 + 脏输入兜底 |

## 3. PMV 输入来源(诚实对待,写入 DecisionTrace.assumptions)

ta=车内温度(实测,缺失则车外+日照热模型估算)、RH=实测、风速=当前风量档查表、
clo=衣着映射、met=活动映射、MRT=车内温+日照估算(唯一估算项)。
PMV/PPD/EQT 为基于上述输入的模型值,UI/CLI 展示估算项与计算过程,不当作实测真值。

## 4. 出风模式模型 + 多 Agent(智能除雾 Agent)

3 舒适基础模式(face 夏/制冷、feet 冬/制热、face_feet 春秋)由舒适推理给出;
除霜叠加由**独立「智能除雾 Agent」**判定,安全层消费其 `DefogDecision` 叠加(face_defrost 等);
纯 defrost 仅显式开启最大除霜功能时强制最大风量。LLM/Mock 只产 3 基础;7 态由 `safety.apply_safety` 落定。

**智能除雾 Agent(`defog/`,车厢级独立 LangGraph)**:
- 职责与舒适 Agent 解耦,仅凭**玻璃表面温度 / 玻璃附近空气温度 / 玻璃附近湿度 / 雨量信号
  `rain_level`(无/小雨/中雨/大雨)** 判定是否除雾及紧急度(none/mild/strong)。
- 判定 = 前馈(雨量/高湿)+ 反馈(玻璃温度−露点裕度;≤2℃强、≤4℃轻度,阈值在 `config.DEFOG`)。
- 图序列 `sense → knowledge → decide`;专属工具 `defog/tools.py`、专属知识库 `defog/docs/defog_kb.json`、
  专属决策器 `defog/decider.py`(规则 + LLM,**安全取严**:取规则与 LLM 更紧急者)。
- 接入:`engine.defog_for(cabin)` 按车厢指纹缓存,注入各座位图 `state['defog']`;
  strong 级叠加时保证风量 ≥ `DEFOG.fan_floor_strong`(=4)。

## 4b. 瞬态控制(负荷 → 稳态)

`tools/thermal_comfort.transient_setpoint_fan(cabin_temp, target)`:车内离目标越远(负荷越大)
→ 设定越激进(制冷更低/制热更高)+ 风量越大,加快收敛;趋近 → 设定回归目标 + 降风量;
稳态 → 最舒适目标 + 低档(常规 **2~3 档**,`fan_steady_min=2`;个性化可学到 1 档)。
comfort 节点算出 `transient` 注入 payload `transient_recommendation`,LLM 提示词**强制同向遵循**,
MockDecider 直接采用,口径统一。

## 5. 逐步逼近(游标式 + LLM 实时推理 + 确定性兜底)

- 一次修正建立偏好(写入 user×seat 记忆),并重置逼近游标、进入冷静期。
- 冷静期(默认 600s)外的每次推理:`approach` 节点从游标(首次=专业基准)按
  `approach_step` 比例(APPROACH_RATE=0.5)朝偏好迈一步、贴齐网格(温度 0.5℃/风量 1 档),
  2-3 步收敛。游标在 `engine.approach_cursor[(user,seat)]` 跨推理持久化,仅非冷静期推进。
- LLM 实时推理:payload 注入用户偏好 + 逐步逼近提示,知识库含 `gradual_preference_approach`
  等条目(has_preference 时检索),approach 节点为确定性兜底保证精确收敛。
- 防抖放宽(风量任意整档、温度 0.5℃ 步进均生效),避免吃掉逼近中间步。

## 6. 架构 / LangGraph

```
触发(周期+事件) → SceneInput{cabin, occupants[]}
 → engine.infer:每位在座乘员各跑一次单座位图(多温区独立)
   featurize(列全部特征) → recall(user×seat) → comfort(PMV/PPD/EQT+知识RAG)
   → llm_infer(专业基准,已注入偏好与逼近规则) → approach(游标式逐步逼近)
   → safety(消费除雾 Agent 的 DefogDecision 叠加除霜/边界)
 → 智能除雾 Agent(车厢级,每场景判定一次,sense→knowledge→decide)结论注入各座位 state
 → 防抖/冷静期决定生效;DecisionTrace 留痕
用户覆盖/语音 → apply_correction / apply_command:写记忆 + 锁定 + 重置游标
实时推理链:engine.infer(capture_chain=True) 一次推理同时产出"生效设定"与"推理链步骤",
            总览与推理链数值完全一致(逐节点含计算过程);stream_seat 保留供单座位流式/测试
降级链:云LLM(超时)→ MockDecider → 沿用上次+舒适锚点
```

> **一致性要点**:总览卡片与推理链来自**同一次 infer**;LLM payload **不含 seat_id**(热舒适与座位无关),
> 相同物理输入 → 相同结果。左右差异只来自实际不同输入(按座位日照、各自记忆)。

## 7. 预装能力清单(knowledge / tool / skill)

- **Knowledge(舒适 56 条 + 除雾 10 条)**:季节策略、暴晒辐射补偿、快速降温分阶段、高湿除湿、起雾/除霜、
  内外循环、低电量节能、老人/小孩/睡眠/兴奋/衣着/性别 BMI、气流与吹风感、局部不适、
  多温区分区、PMV/PPD 解读、适应性舒适、门窗影响、安全优先,以及记忆学习(逐步逼近、
  一致性/矛盾/时效、冷静期)。位于 `knowledge/docs/*.json`,放入新文件即被索引。
- **Tool(确定性,@tool 注册)**:compute_pmv_ppd、equivalent_temperature、
  target_comfort_temp、comfort_temp_band、fog_risk、dew_point、recirculation_hint、
  energy_hint、met_from_activity、clo_from_clothing、estimate_cabin_temp、approach_step。
- **Skill**:ThermalComfort、Knowledge、Strategy(场景→高层策略)、Energy(内外循环/节能)、
  Weather、Vehicle、Memory。
- **除雾 Agent 专属能力**:Tool = dew_point_margin / rain_fog_pressure / defog_urgency(`defog/tools.py`);
  Knowledge = `defog/docs/defog_kb.json`(10 条:起雾原理/前馈反馈/雨量分级/裕度策略/气流 AC 除湿/内外循环/
  冬季冷启动/安全优先/湿源);决策器 = 规则 + LLM(`defog/decider.py`)。

## 8. 验证

- 97 项测试:特征/记忆/热舒适(对照 ISO 7730)/知识 RAG/Skills/provider/safety/**除雾 Agent**/触发/
  NLU 指令/端到端闭环(逐步逼近 7→5→4→3、多温区独立、季节/用户隔离、降级、防抖、工况回归)。
- 运行:`pytest -q`;`streamlit run tms_agent/app_web.py`;`app_cli teach 0`。

## 更新记录

- **GitHub 公开仓库**:https://github.com/suzike/Vehicle-Thermal-LLM-MultiAgent;双击脚本 `setup.bat`(装依赖)/
  `run_web.bat`(启 Web)/ `run_cli.bat`(启 CLI 终端,`tms` 快捷命令);`.gitignore` 排除
  `.env/.venv/记忆`,仅 `.env.example` 入库;**`.gitattributes` 强制 `.bat` 以 CRLF 检出**(修复
  Windows 双击一闪而退)。CLI 与 Web **同引擎、功能对齐**(infer/chain/say/correct/teach/memory/reset)。
- **总览↔推理链同源 + 主副驾一致(修复)**:`engine.infer(capture_chain=True)` 一次推理同时产出
  生效设定与推理链步骤,Web 两处数值完全一致;LLM payload 去掉 `seat_id`,相同物理输入 → 相同结果。
- **瞬态控制接入 LLM**:`transient_setpoint_fan` 设定/风量随负荷回调,注入 payload 并由提示词强制遵循;
  稳态风量落脚 2~3 档(`fan_steady_min=2`,个性化可学到 1 档)。
- **完整记忆链条**:`CorrectionRecord` 存 人员/车辆/环境 完整快照 → 推理 → 用户修正;语音与手动调节都写入。
- **PMV 可视化模块 + HUD 渲染修复**:座舱图下方 PMV 逻辑/标尺/变化曲线;富 SVG 改用
  `st.components.v1.html`(iframe)渲染避免 Markdown 截断;座椅按内温着色 + 动态气流;冷静期倒计时。
- **知识必依据**:LLM 提示词强制以注入的 knowledge 规则为权威依据。
- **多 Agent:智能除雾 Agent 独立**(`defog/`)——起雾判定从 safety 拆出为车厢级独立 LangGraph,
  新增 `rain_level` 雨量输入与 `DefogDecision`;前馈+反馈判定,安全取严,舒适侧叠加除霜。
- 学习策略:渐进置信 → 持续逼近 → **游标式逐步逼近**(按推理次数迭代,7→5→4→3);防抖放宽避免吃步。
- 设定域:温度 15.5–31.5/0.5、风量 1–7、出风 7 模式(3 基础+除霜叠加,纯除霜仅最大档)。
- 输入:太阳两路 W/m²、车内温度实测输入(缺失估算)、衣着、人体特征/情绪/活动、门窗。
- PMV 风速改为当前风量档查表;MRT/车内温估算透明化。
- 引入 scikit-learn + chromadb 知识检索;新增露点/起雾/舒适温区/内外循环/节能工具与 Strategy/Energy 技能。
- 语音/对话文本输入(每座位入口,NLU 理解并记忆)。
- HMI 改造:暗色座舱主题、车厢俯视图、按座位实时推理链 + 计算过程可视化、修改自动记忆。
