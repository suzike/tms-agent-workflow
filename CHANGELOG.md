# 更新日志

> 本文档记录座舱智慧空调 Agent 的所有重要变更。
> 格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/),
> 版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

---

## [Unreleased]

### 计划
- 真实 CAN/SOA 信号接入
- 域控嵌入式部署适配
- 多乘员三维舒适度可视化
- 云端记忆跨车同步

---

## [0.2.1] - 2026-06-23

### 文档
- 同步 README/PLAN/CLAUDE 中的脚本说明（run_cli/CLI 对齐）
- 记录 `.gitattributes` CRLF 修复方案

### 修复
- `.bat` 脚本强制 CRLF 换行，解决双击一闪而退问题

---

## [0.2.0] - 2026-06-22

### 新增
- CLI 终端一键启动脚本 `run_cli.bat`
- CLI 命令对齐 Web（`say`/`chain`/`memory` + 除雾决策显示）
- 多 Agent 推理链可解释性展示
- 学习记忆链条完整可视化

### 文档
- README 新增 CLI 终端界面展示（rich 终端图）
- 开发文档同步至最新现状

---

## [0.1.2] - 2026-06-21

### 修复
- 总览页与推理链统一为同一次推理调用，消除温度不一致问题
- LLM payload 移除 `seat_id`，同物理输入保证主/副驾一致输出

### 文档
- README 图文并茂重写（产品架构、多 Agent 架构 Mermaid 图、学习闭环图、实拍界面）

---

## [0.1.1] - 2026-06-20

### 文档
- 同步开发文档：总览↔推理链同源、payload 去 seat_id、瞬态注入说明
- 记录完整记忆链条、PMV 可视化、HUD 渲染方案

---

## [0.1.0] - 2026-06-19

### 新增
- 多温区独立预测（每乘员独立 LangGraph 推理链）
- 独立智能除雾 Agent（车厢级，sense→knowledge→decide）
- 舒适 Agent 按座位输出温度/风量/出风模式
- ISO 7730 / ASHRAE 55 专业热舒适计算（PMV/PPD/EQT）
- 逐步逼近学习策略（游标式，非硬切换）
- 瞬态控制（负荷驱动温度/风量动态调整）
- 语音/对话指令入口（NLU 解释 + 记忆）
- RAG 知识检索（sklearn TF-IDF + chromadb，66 条知识）
- Streamlit Web UI（座舱 HUD + PMV 可视化 + 推理链）
- CLI 终端（同一引擎、功能对齐）
- 完整记忆链条（CorrectionRecord 快照）
- 安全层：除雾叠加 + 设定域硬约束 + 防抖迟滞
- LLM 降级链（云→Mock→沿用上次+舒适锚点）
