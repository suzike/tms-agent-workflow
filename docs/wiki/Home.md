# 座舱智慧空调 Agent Wiki

欢迎来到 TMS Cockpit AI 知识库。

## 快速导航

- [架构总览](Architecture) — 多 Agent 架构、推理链、数据流
- [开发指南](Development-Guide) — 环境搭建、本地运行、扩展开发
- [API 参考](API-Reference) — Engine/Graph/Memory/Tools 接口说明
- [更新日志](CHANGELOG) — 版本变更记录

## 项目简介

座舱智慧空调 Agent 是一套面向汽车座舱的「热感知 → 专业推理 → 多温区下发 → 学习用户偏好」智能体。

核心特性：

| 特性 | 说明 |
|------|------|
| 多 Agent | 舒适 Agent + 智能除雾 Agent |
| 专业约束 | ISO 7730 / ASHRAE 55 PMV/PPD |
| 多温区 | 每乘员独立预测与记忆 |
| 学习 | 逐步逼近用户真实偏好 |
| 知识库 | TfidfVectorizer + chromadb, 66 条 |

## 仓库

- GitLab: `http://192.168.110.104/develop/vehicle-thermal-agent`
- GitHub: `https://github.com/suzike/tms-agent-workflow`
