#!/bin/bash

# VBench Camera Motion 评测服务启动脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE} VBench Camera Motion 评测服务启动器${NC}"
echo -e "${BLUE}========================================${NC}"

# 默认配置
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="5000"
DEFAULT_DEVICE="cuda:0"
DEFAULT_ENV="development"

# 参数解析
while [[ $# -gt 0 ]]; do
  case $1 in
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --env)
      ENVIRONMENT="$2"
      shift 2
      ;;
    --local)
      LOCAL_MODE="true"
      shift
      ;;
    --help)
      echo "使用方法: $0 [选项]"
      echo ""
      echo "选项:"
      echo "  --host HOST        服务器地址 (默认: $DEFAULT_HOST)"
      echo "  --port PORT        服务器端口 (默认: $DEFAULT_PORT)"
      echo "  --device DEVICE    GPU设备 (默认: $DEFAULT_DEVICE)"
      echo "  --env ENV          环境配置 (development/production, 默认: $DEFAULT_ENV)"
      echo "  --local            本地模式，跳过模型下载"
      echo "  --help             显示帮助信息"
      echo ""
      echo "示例:"
      echo "  $0                                # 使用默认配置启动"
      echo "  $0 --port 8080 --device cuda:1   # 指定端口和GPU设备"
      echo "  $0 --env production --local       # 生产环境本地模式"
      exit 0
      ;;
    *)
      echo -e "${RED}未知参数: $1${NC}"
      echo "使用 --help 查看帮助信息"
      exit 1
      ;;
  esac
done

# 设置默认值
HOST=${HOST:-$DEFAULT_HOST}
PORT=${PORT:-$DEFAULT_PORT}
DEVICE=${DEVICE:-$DEFAULT_DEVICE}
ENVIRONMENT=${ENVIRONMENT:-$DEFAULT_ENV}
LOCAL_MODE=${LOCAL_MODE:-"false"}

echo -e "${GREEN}配置信息:${NC}"
echo -e "  服务器地址: ${YELLOW}$HOST:$PORT${NC}"
echo -e "  GPU设备: ${YELLOW}$DEVICE${NC}"
echo -e "  环境: ${YELLOW}$ENVIRONMENT${NC}"
echo -e "  本地模式: ${YELLOW}$LOCAL_MODE${NC}"
echo ""

# 检查 Python 环境
echo -e "${BLUE}检查 Python 环境...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到 python3${NC}"
    exit 1
fi

python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}Python 版本: $python_version${NC}"

# 检查虚拟环境
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo -e "${GREEN}虚拟环境: $VIRTUAL_ENV${NC}"
else
    echo -e "${YELLOW}警告: 未检测到虚拟环境，建议使用虚拟环境${NC}"
fi

# 检查 CUDA
if [[ "$DEVICE" == cuda* ]]; then
    echo -e "${BLUE}检查 CUDA 环境...${NC}"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
        echo -e "${GREEN}CUDA 环境可用${NC}"
    else
        echo -e "${YELLOW}警告: 未检测到 CUDA 环境，将回退到 CPU${NC}"
        DEVICE="cpu"
    fi
fi

# 检查依赖
echo -e "${BLUE}检查依赖包...${NC}"
if [[ -f "requirements_service.txt" ]]; then
    echo -e "${GREEN}发现依赖文件: requirements_service.txt${NC}"
    echo -e "${YELLOW}如需安装依赖，请运行: pip install -r requirements_service.txt${NC}"
else
    echo -e "${YELLOW}警告: 未找到 requirements_service.txt${NC}"
fi

# 检查服务文件
if [[ ! -f "camera_motion_service.py" ]]; then
    echo -e "${RED}错误: 未找到服务文件 camera_motion_service.py${NC}"
    exit 1
fi

# 设置环境变量
export HOST="$HOST"
export PORT="$PORT"
export DEFAULT_DEVICE="$DEVICE"
export FLASK_ENV="$ENVIRONMENT"

if [[ "$LOCAL_MODE" == "true" ]]; then
    export LOCAL_MODE="true"
fi

# 创建必要的目录
mkdir -p uploads
mkdir -p logs

echo ""
echo -e "${GREEN}启动服务...${NC}"
echo -e "${BLUE}访问地址: http://$HOST:$PORT${NC}"
echo -e "${BLUE}健康检查: http://$HOST:$PORT/health${NC}"
echo ""
echo -e "${YELLOW}按 Ctrl+C 停止服务${NC}"
echo ""

# 启动服务
exec python3 camera_motion_service.py 