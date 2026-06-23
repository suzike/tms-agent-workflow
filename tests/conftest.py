"""测试夹具:强制使用离线 Mock 决策器。

即使 .env 配了真实云端 Key,测试也必须可复现、零网络调用。
本文件在任何 tms_agent 导入前设置环境变量,使 config.LLM_CONFIG.provider == "mock"。
"""
import os

os.environ["LLM_PROVIDER"] = "mock"
