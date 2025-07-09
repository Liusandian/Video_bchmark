@echo off
REM VBench Camera Motion 评测服务启动脚本 (Windows)

setlocal enabledelayedexpansion

echo ========================================
echo  VBench Camera Motion 评测服务启动器
echo ========================================

REM 默认配置
set DEFAULT_HOST=0.0.0.0
set DEFAULT_PORT=5000
set DEFAULT_DEVICE=cuda:0
set DEFAULT_ENV=development

REM 解析命令行参数
set HOST=%DEFAULT_HOST%
set PORT=%DEFAULT_PORT%
set DEVICE=%DEFAULT_DEVICE%
set ENVIRONMENT=%DEFAULT_ENV%
set LOCAL_MODE=false

:parse
if "%~1"=="" goto :config
if "%~1"=="--host" (
    set HOST=%~2
    shift
    shift
    goto :parse
)
if "%~1"=="--port" (
    set PORT=%~2
    shift
    shift
    goto :parse
)
if "%~1"=="--device" (
    set DEVICE=%~2
    shift
    shift
    goto :parse
)
if "%~1"=="--env" (
    set ENVIRONMENT=%~2
    shift
    shift
    goto :parse
)
if "%~1"=="--local" (
    set LOCAL_MODE=true
    shift
    goto :parse
)
if "%~1"=="--help" (
    echo 使用方法: %0 [选项]
    echo.
    echo 选项:
    echo   --host HOST        服务器地址 (默认: %DEFAULT_HOST%^)
    echo   --port PORT        服务器端口 (默认: %DEFAULT_PORT%^)
    echo   --device DEVICE    GPU设备 (默认: %DEFAULT_DEVICE%^)
    echo   --env ENV          环境配置 (development/production, 默认: %DEFAULT_ENV%^)
    echo   --local            本地模式，跳过模型下载
    echo   --help             显示帮助信息
    echo.
    echo 示例:
    echo   %0                                          # 使用默认配置启动
    echo   %0 --port 8080 --device cuda:1             # 指定端口和GPU设备
    echo   %0 --env production --local                 # 生产环境本地模式
    goto :end
)
echo 未知参数: %~1
echo 使用 --help 查看帮助信息
goto :end

:config
echo 配置信息:
echo   服务器地址: %HOST%:%PORT%
echo   GPU设备: %DEVICE%
echo   环境: %ENVIRONMENT%
echo   本地模式: %LOCAL_MODE%
echo.

REM 检查 Python 环境
echo 检查 Python 环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到 python
    goto :end
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python 版本: %PYTHON_VERSION%

REM 检查虚拟环境
if defined VIRTUAL_ENV (
    echo 虚拟环境: %VIRTUAL_ENV%
) else (
    echo 警告: 未检测到虚拟环境，建议使用虚拟环境
)

REM 检查 CUDA
if "%DEVICE:~0,4%"=="cuda" (
    echo 检查 CUDA 环境...
    nvidia-smi >nul 2>&1
    if errorlevel 1 (
        echo 警告: 未检测到 CUDA 环境，将回退到 CPU
        set DEVICE=cpu
    ) else (
        echo CUDA 环境可用
    )
)

REM 检查依赖
echo 检查依赖包...
if exist requirements_service.txt (
    echo 发现依赖文件: requirements_service.txt
    echo 如需安装依赖，请运行: pip install -r requirements_service.txt
) else (
    echo 警告: 未找到 requirements_service.txt
)

REM 检查服务文件
if not exist camera_motion_service.py (
    echo 错误: 未找到服务文件 camera_motion_service.py
    goto :end
)

REM 设置环境变量
set HOST=%HOST%
set PORT=%PORT%
set DEFAULT_DEVICE=%DEVICE%
set FLASK_ENV=%ENVIRONMENT%

if "%LOCAL_MODE%"=="true" (
    set LOCAL_MODE=true
)

REM 创建必要的目录
if not exist uploads mkdir uploads
if not exist logs mkdir logs

echo.
echo 启动服务...
echo 访问地址: http://%HOST%:%PORT%
echo 健康检查: http://%HOST%:%PORT%/health
echo.
echo 按 Ctrl+C 停止服务
echo.

REM 启动服务
python camera_motion_service.py

:end
pause 