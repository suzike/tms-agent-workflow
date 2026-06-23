@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [提示] 尚未安装环境,请先双击 setup.bat 完成安装。
  echo.
  pause
  exit /b 1
)
REM 打开持久交互终端:激活虚拟环境 + 设置 tms 快捷命令 + 打印用法
cmd /k "call .venv\Scripts\activate.bat & doskey tms=python -m tms_agent.app_cli $* & cls & echo ================ 座舱智慧空调 Agent · CLI 已就绪 ================ & echo  tms list                              列出演示场景 & echo  tms infer 0                           推理场景0(含除雾Agent决策) & echo  tms chain 0                           多Agent实时推理链 & echo  tms say 0 driver 太冷了               语音/对话指令 & echo  tms correct 0 driver 19 6 face_feet   手动修正(写入记忆) & echo  tms teach 0                           学习闭环演示(逐步逼近) & echo  tms memory                            查看学习记忆完整链条 & echo  tms reset                             清空所有记忆 & echo ================================================================ & echo (输入 exit 关闭窗口)"
