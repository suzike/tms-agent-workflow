"""座舱智慧空调 Agent —— PoC。

基于 LangGraph + LangChain 的座舱热感 Agent:输入环境/人员/车辆/天气,经专业
热舒适知识约束的 LLM 推理出每座位的风量/温度/出风模式,并通过记忆闭环学习用户修正。
"""

__version__ = "0.1.0"
