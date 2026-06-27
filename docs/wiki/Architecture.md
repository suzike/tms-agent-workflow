# 架构总览

## 多 Agent 架构

```
┌─────────────────────────────────────────┐
│              Engine (engine.py)          │
│  ┌───────────────────────────────────┐  │
│  │      除雾 Agent (defog/)          │  │
│  │  sense → knowledge → decide       │  │
│  │  输入: 玻璃温湿度/雨量            │  │
│  │  输出: DefogDecision              │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │      舒适 Agent (per seat)        │  │
│  │  featurize → recall → comfort     │  │
│  │  → llm_infer → approach → safety  │  │
│  │  输出: 温度/风量/出风模式          │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │    安全合成 (safety.py)           │  │
│  │  除雾叠加 + 设定域硬约束          │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

## 数据流

1. 输入采集（环境 / 人员 / 车辆 / 天气）
2. 除雾 Agent 判定（车厢级，优先）
3. 舒适 Agent 推理（每座位独立）
4. 安全合成（叠加除霜 + 约束检查）
5. 输出写入 + 记忆更新

## 知识库

- `knowledge/docs/thermal_comfort_kb.json` - 56 条热舒适知识
- `knowledge/docs/comfort_strategies.json` - 舒适策略
- `knowledge/docs/occupant_profiles.json` - 乘员画像
- `defog/docs/defog_kb.json` - 10 条除雾知识

使用 sklearn TfidfVectorizer + chromadb 进行检索。

## 记忆系统

- 按 `用户 × 座位` 独立存储
- 每条记忆包含完整输入 → 推理 → 修正链条
- 逐步逼近策略（游标式，非硬切换）
- 冷静期 600s + 半衰期老化
