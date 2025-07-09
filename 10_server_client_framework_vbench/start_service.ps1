# VBench Camera Motion 评测服务启动脚本 (PowerShell)

param(
    [string]$ServerHost = "0.0.0.0",
    [int]$Port = 5000,
    [string]$Device = "cuda:0", 
    [string]$Environment = "development",
    [switch]$Local,
    [switch]$Help
)

function Write-ColorText {
    param([string]$Text, [string]$Color = "White")
    
    $colors = @{
        "Red" = "Red"
        "Green" = "Green"
        "Yellow" = "Yellow"
        "Blue" = "Blue"
        "White" = "White"
    }
    
    Write-Host $Text -ForegroundColor $colors[$Color]
}

if ($Help) {
    Write-Host "VBench Camera Motion 评测服务启动器"
    Write-Host ""
    Write-Host "使用方法: .\start_service.ps1 [参数]"
    Write-Host ""
    Write-Host "参数:"
    Write-Host "  -ServerHost <地址>  服务器地址 (默认: 0.0.0.0)"
    Write-Host "  -Port <端口>        服务器端口 (默认: 5000)"
    Write-Host "  -Device <设备>      GPU设备 (默认: cuda:0)"
    Write-Host "  -Environment <环境> 环境配置 (默认: development)"
    Write-Host "  -Local              本地模式，跳过模型下载"
    Write-Host "  -Help               显示帮助信息"
    Write-Host ""
    Write-Host "示例:"
    Write-Host "  .\start_service.ps1"
    Write-Host "  .\start_service.ps1 -Port 8080 -Device cuda:1"
    Write-Host "  .\start_service.ps1 -Environment production -Local"
    exit 0
}

Write-ColorText "========================================" "Blue"
Write-ColorText " VBench Camera Motion 评测服务启动器" "Blue"
Write-ColorText "========================================" "Blue"

Write-ColorText "配置信息:" "Green"
Write-Host "  服务器地址: $ServerHost`:$Port" -ForegroundColor Yellow
Write-Host "  GPU设备: $Device" -ForegroundColor Yellow
Write-Host "  环境: $Environment" -ForegroundColor Yellow
Write-Host "  本地模式: $Local" -ForegroundColor Yellow
Write-Host ""

# 检查 Python 环境
Write-ColorText "检查 Python 环境..." "Blue"
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-ColorText "Python 版本: $pythonVersion" "Green"
    } else {
        Write-ColorText "错误: 未找到 python" "Red"
        exit 1
    }
} catch {
    Write-ColorText "错误: 未找到 python" "Red"
    exit 1
}

# 检查虚拟环境
if ($env:VIRTUAL_ENV) {
    Write-ColorText "虚拟环境: $env:VIRTUAL_ENV" "Green"
} else {
    Write-ColorText "警告: 未检测到虚拟环境，建议使用虚拟环境" "Yellow"
}

# 检查 CUDA
if ($Device.StartsWith("cuda")) {
    Write-ColorText "检查 CUDA 环境..." "Blue"
    try {
        nvidia-smi | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-ColorText "CUDA 环境可用" "Green"
        } else {
            Write-ColorText "警告: 未检测到 CUDA 环境，将回退到 CPU" "Yellow"
            $Device = "cpu"
        }
    } catch {
        Write-ColorText "警告: 未检测到 CUDA 环境，将回退到 CPU" "Yellow"
        $Device = "cpu"
    }
}

# 检查依赖
Write-ColorText "检查依赖包..." "Blue"
if (Test-Path "requirements_service.txt") {
    Write-ColorText "发现依赖文件: requirements_service.txt" "Green"
    Write-ColorText "如需安装依赖，请运行: pip install -r requirements_service.txt" "Yellow"
} else {
    Write-ColorText "警告: 未找到 requirements_service.txt" "Yellow"
}

# 检查服务文件
if (!(Test-Path "camera_motion_service.py")) {
    Write-ColorText "错误: 未找到服务文件 camera_motion_service.py" "Red"
    exit 1
}

# 设置环境变量
$env:HOST = $ServerHost
$env:PORT = $Port.ToString()
$env:DEFAULT_DEVICE = $Device
$env:FLASK_ENV = $Environment

if ($Local) {
    $env:LOCAL_MODE = "true"
}

# 创建必要的目录
if (!(Test-Path "uploads")) { New-Item -ItemType Directory -Name "uploads" | Out-Null }
if (!(Test-Path "logs")) { New-Item -ItemType Directory -Name "logs" | Out-Null }

Write-Host ""
Write-ColorText "启动服务..." "Green"
Write-ColorText "访问地址: http://$ServerHost`:$Port" "Blue"
Write-ColorText "健康检查: http://$ServerHost`:$Port/health" "Blue"
Write-Host ""
Write-ColorText "按 Ctrl+C 停止服务" "Yellow"
Write-Host ""

# 启动服务
try {
    python camera_motion_service.py
} catch {
    Write-ColorText "服务启动失败: $_" "Red"
    exit 1
} 