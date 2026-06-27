# 开发指南

## 环境要求

- Python 3.14
- Windows / macOS / Linux
- Git

## 快速开始

```bash
# 克隆仓库
git clone http://192.168.110.104/develop/vehicle-thermal-agent.git
cd vehicle-thermal-agent

# 安装依赖 (Windows 双击 setup.bat)
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt

# 配置 LLM（可选，不配置则使用 Mock 模式）
cp .env.example .env
# 编辑 .env 填写 DEEPSEEK_API_KEY

# 运行测试
.venv/Scripts/python -m pytest -q

# 启动 Web UI
.venv/Scripts/python -m streamlit run tms_agent/app_web.py

# 启动 CLI
.venv/Scripts/python -m tms_agent.app_cli
```

## 项目结构

```
├── CLAUDE.md          # 开发约束（必读）
├── PLAN.md            # 详细方案
├── config.py          # 所有阈值（调参改这里）
├── engine.py          # 核心引擎
├── graph/             # LangGraph 推理链
│   ├── build.py       # 图构建
│   ├── nodes.py       # 图节点
│   └── state.py       # 状态定义
├── defog/             # 智能除雾 Agent（独立）
│   ├── agent.py       # 除雾图
│   ├── decider.py     # 确定性决策器
│   ├── tools.py       # 除雾工具
│   └── docs/          # 除雾知识库
├── knowledge/         # 舒适侧知识库
├── memory/            # 记忆存储
├── skills/            # 技能系统
├── tools/             # 计算工具
├── llm/               # LLM Provider
├── runtime/           # 触发/调度
├── safety.py          # 安全约束
├── schemas.py         # 数据模型
├── nlu.py             # 自然语言理解
├── app_web.py         # Streamlit Web UI
├── app_cli.py         # Rich CLI 终端
└── tests/             # 测试（97 个用例）
```

## 开发约定

1. **代码注释**：使用中文
2. **不可变风格**：节点返回 state 增量，不原地修改
3. **阈值集中**：所有配置在 `config.py` 的 `Thresholds`/`DefogThresholds` 中
4. **测试优先**：修改功能前先写测试，确保 `pytest -q` 全绿

## 常见任务

### 添加新知识条目

编辑 `knowledge/docs/*.json`，重新运行推理即可生效。

### 调配舒适参数

修改 `config.py` 中的 `ComfortDefaults` 和 `Thresholds`。

### 接入新 LLM Provider

参考 `llm/provider.py`，实现 `CloudDecider` 接口。
