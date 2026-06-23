@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo   座舱智慧空调 Agent  -  环境一键安装
echo ============================================================
echo.

REM ---- 检查 Python ----
where python >nul 2>nul
if errorlevel 1 (
  echo [错误] 未检测到 Python。
  echo        请先安装 Python 3.11+：https://www.python.org/downloads/
  echo        安装时务必勾选 "Add Python to PATH"，然后重新双击本脚本。
  echo.
  pause
  exit /b 1
)
echo [信息] 已检测到 Python：
python --version
echo.

REM ---- 创建虚拟环境 ----
if not exist ".venv\Scripts\python.exe" (
  echo [1/3] 创建虚拟环境 .venv ...
  python -m venv .venv
  if errorlevel 1 (
    echo [错误] 虚拟环境创建失败。
    pause
    exit /b 1
  )
) else (
  echo [1/3] 已存在虚拟环境 .venv，跳过创建。
)

REM ---- 升级 pip ----
echo [2/3] 升级 pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip

REM ---- 安装依赖 ----
echo [3/3] 安装依赖（首次约几分钟，请耐心等待）...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo [错误] 依赖安装失败，请检查网络后重试。
  echo        如在国内可尝试：".venv\Scripts\python.exe" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  pause
  exit /b 1
)

echo.
echo ============================================================
echo   安装完成！
echo   - 双击 run_web.bat  启动座舱 Web 界面（浏览器 http://127.0.0.1:8501）
echo   - 接入云端大模型：复制 .env.example 为 .env 并填入 DEEPSEEK_API_KEY
echo     （不填也能离线运行，使用内置规则引擎）
echo ============================================================
echo.
pause
