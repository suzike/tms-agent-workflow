# 座舱智慧空调 Agent · PoC

基于 **LangGraph + LangChain** 的座舱热感 Agent。输入环境/人员/车辆/天气,经
**专业热舒适知识(ISO 7730 / ASHRAE 55)约束的 LLM** 推理,为**每位乘员独立**给出
**风量 / 温度 / 出风模式**;用户手动调整或语音指令会被**记忆并逐步逼近**,越用越懂你。

## 核心能力

- **细粒度输入**:车外/车内温度、相对湿度、**按主副驾两路太阳辐照(W/m²)**、车速、SOC、
  天气、季节、四门四窗状态、最大除霜开关;每位乘员含年龄/性别/身高/体重/BMI、**衣着**、
  情绪(喜怒哀乐愁)、活动"热状态"(兴奋/平静/睡眠),自动识别小孩/大人/老人。
- **专业热舒适层**:确定性计算工具(PMV/PPD、当量温度 EQT、目标温度、露点/起雾风险、
  舒适温区、内外循环/节能建议)+ scikit-learn TF-IDF + chromadb 向量库的知识 RAG。
- **多温区独立预测**:每位乘员按"用户×座位"独立推理与记忆。
- **多 Agent · 智能除雾 Agent**:独立于舒适 Agent 的车厢级 LangGraph(感知→检索→判定),仅凭
  **玻璃表面温度 / 玻璃附近空气温度 / 玻璃附近湿度 / 雨量信号(无/小雨/中雨/大雨)** 判定起雾,
  前馈(雨量/高湿)+反馈(玻璃温度 vs 露点裕度),结论叠加到舒适出风模式(吹面除霜/吹脚除霜等);
  自带专属工具/知识库/决策器。
- **逐步逼近学习**:手动/语音修改后,冷静期外的后续推理沿"专业推荐→用户偏好"**迭代收敛**
  (差值越大首步越大,2-3 步到位,例:风量 7→5→4→3),而非一次跳变。
- **语音/对话输入**:每座位一个对话入口,你把语音转成文本(如"太冷了""天太热了"),
  系统理解诉求即时调整并写入记忆。
- **设定域**:温度 15.5–31.5℃/0.5;风量 1–7;出风 7 模式 = 3 舒适基础(吹面/吹面吹脚/吹脚)
  + 由智能除雾 Agent 判定后叠加除霜;纯除霜仅在开启最大除霜功能时进入。
- **决策优先级**:安全(除雾 Agent/强制除霜)> 用户偏好逐步逼近 > 专业舒适锚点 > LLM。
- **鲁棒性**:LLM 超时/失败降级到规则引擎、结果缓存、防抖迟滞、手动冷静期、脏输入兜底。
- **可观测**:每座位 DecisionTrace 留痕、PMV 输入透明化(assumptions)、会话修正率。
- **科技感 HMI**:暗色座舱主题 + 车厢俯视图(座椅按**内温**着色、按出风模式喷射**动态气流**)
  + 按座位实时推理链(含车内温度/PMV/PPD 计算过程可视化)。
- **PMV 可视化模块**:座舱图下方独立模块,展示 PMV 计算逻辑 + 多温区 PMV 标尺(舒适带)
  + 各区域 PMV 变化曲线。
- **完整记忆链条**:每条记忆存「人员/车辆/环境输入 → 系统推理 → 用户修正」完整快照,
  「学习记忆」页可逐条展开查看;手动调节与语音指令都即时写入并显示学习状态与冷静期倒计时。

## PMV 六输入来源(透明)

| 输入 | 来源 |
|---|---|
| 空气温度 ta | **车内温度(实测)**,缺失时由车外温+日照热模型估算 |
| 相对湿度 | 实测 |
| 风速 | **当前空调风量档位查表** |
| clo 服装热阻 | **衣着输入映射** |
| met 代谢率 | **活动状态映射** |
| 平均辐射温度 MRT | 由车内温度 + 太阳辐照估算(唯一估算项,已标注) |

## 快速开始(Windows 双击)

1. 安装 [Python 3.11+](https://www.python.org/downloads/),安装时务必勾选 **Add Python to PATH**。
2. 双击 **`setup.bat`** —— 自动创建虚拟环境 `.venv` 并安装全部依赖(首次约几分钟)。
3. 双击 **`run_web.bat`** —— 启动座舱 Web 界面,浏览器访问 http://127.0.0.1:8501。

> 无需任何云端 Key 即可**离线运行**(内置规则引擎兜底)。接入 DeepSeek 见下方「接入云端 LLM」。

## 手动命令(可选)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows;已在 Python 3.14 验证

.venv/Scripts/python -m pytest -q                          # 97 项测试(含独立除雾 Agent)
.venv/Scripts/python -m streamlit run tms_agent/app_web.py # Web HMI(访问 http://127.0.0.1:8501)
.venv/Scripts/python -m tms_agent.app_cli teach 0          # CLI 闭环演示
```

## 接入云端 LLM

复制 `.env.example` 为 `.env` 填入 Key(默认 DeepSeek):
```
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_MODEL=deepseek-v4-flash   # 注意小写;思考版 deepseek-v4-pro 较慢
```
未配置 Key 自动退回内置规则引擎离线运行。Web 侧栏「🔌 测试连接」可自检。

## 目录结构

```
tms_agent/
  config.py            阈值/枚举/查表集中收口
  schemas.py           Pydantic 模型 + 输入兜底(CabinContext/OccupantState/...)
  features.py          每乘员分桶加权特征(季节隔离)
  memory/store.py      记忆引擎(user×seat,逼近证据,时间衰减)
  tools/thermal_comfort.py  PMV/PPD/EQT/目标温度/露点/起雾/舒适温区/内外循环/节能/逼近步
  knowledge/           sklearn TF-IDF + chromadb 向量库 + docs/(56 条:策略/人群画像/ISO·ASHRAE)
  skills/              ThermalComfort/Knowledge/Strategy/Energy/Weather/Vehicle/Memory
  llm/provider.py      provider 工厂(默认 DeepSeek,JSON 解析)+ Mock 规则引擎 + NLU interpret
  nlu.py               自然语言/语音指令理解(LLM + 关键词兜底)
  graph/               舒适 LangGraph:featurize→recall→comfort→llm_infer→approach→safety
  defog/               独立智能除雾 Agent:agent.py(sense→knowledge→decide)/tools.py/decider.py/docs/defog_kb.json
  safety.py            边界 + 消费 DefogDecision 叠加除霜 + 防抖 + 冷静期
  runtime/triggers.py  周期 + 事件触发
  engine.py            infer / apply_correction / apply_command / stream_seat(实时链)
  observability.py     日志 + 修正率
  app_cli.py / app_web.py   CLI / Streamlit 座舱 HMI
data/  mock_scenes.json 演示场景  scenario_set.json 回归工况  memory.json 运行时生成
```

## 文档

- [CLAUDE.md](CLAUDE.md) — 开发硬约束(自动加载)
- [PLAN.md](PLAN.md) — as-built 方案与架构

## 非目标(已用接口隔离,本期不做)

真实车辆信号(CAN/SOA)、域控嵌入式部署、本地小模型、语音识别本身(只接收文本)、
云端记忆同步、座椅加热/通风等执行器、多模态热舒适感知。
