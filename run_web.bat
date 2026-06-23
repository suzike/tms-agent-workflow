@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [提示] 尚未安装环境，请先双击 setup.bat 完成安装。
  echo.
  pause
  exit /b 1
)
echo 启动座舱智慧空调 Agent Web 界面 ...
echo 浏览器请访问： http://127.0.0.1:8501   （关闭本窗口即停止）
echo.
".venv\Scripts\python.exe" -m streamlit run tms_agent/app_web.py --server.address 127.0.0.1
pause
