# VBench Camera Motion 评测服务

基于 VBench Camera Motion 评测功能的 Flask HTTP 服务框架，提供简化的 API 接口供其他同事远程调用。

## 功能特性

- 🎬 **运镜预测**: 分析视频中的运镜类型（pan, tilt, zoom, orbit 等）
- 📊 **运镜评测**: 根据期望运镜类型给出评测分数
- 🔄 **批量处理**: 支持多个视频同时评测
- 🌐 **HTTP接口**: 简单的 REST API，易于集成
- 🚀 **GPU加速**: 支持 CUDA 加速，提升评测速度
- 📋 **详细日志**: 完整的评测过程日志记录

## 支持的运镜类型

- `pan_left` - 左移
- `pan_right` - 右移  
- `tilt_up` - 上仰
- `tilt_down` - 下俯
- `zoom_in` - 推近
- `zoom_out` - 拉远
- `orbits` - 环绕
- `oblique` - 斜角
- `static` - 静止

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境（推荐）
python3 -m venv vbench_env
source vbench_env/bin/activate  # Linux/Mac
# 或 vbench_env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements_service.txt
```

### 2. 启动服务

```bash
# Linux/Mac 使用启动脚本（推荐）
chmod +x start_service.sh
./start_service.sh

# Windows 使用 PowerShell 脚本
.\start_service.ps1

# Windows 使用批处理文件
.\start_service.bat

# 或直接运行
python3 camera_motion_service.py

# 指定端口和GPU设备 (Linux/Mac)
./start_service.sh --port 8080 --device cuda:1

# 指定端口和GPU设备 (Windows PowerShell)
.\start_service.ps1 -Port 8080 -Device cuda:1

# 本地模式（跳过模型下载）
./start_service.sh --local          # Linux/Mac
.\start_service.ps1 -Local           # Windows PowerShell
.\start_service.bat --local          # Windows 批处理
```

### 3. 初始化服务

首次使用需要初始化服务（下载模型）：

```bash
# 使用客户端工具
python3 client_example.py --action init

# 或使用 curl
curl -X POST http://localhost:5000/initialize \
  -H "Content-Type: application/json" \
  -d '{"device": "cuda:0", "local_mode": false}'
```

### 4. 测试服务

使用测试脚本验证服务是否正常：

```bash
# 运行 API 测试
python3 test_api.py

# 跳过初始化测试（如果已经初始化过）
python3 test_api.py --skip-init

# 测试远程服务
python3 test_api.py --url http://192.168.1.100:5000
```

## API 文档

### 健康检查

**GET** `/health`

检查服务运行状态。

**响应示例:**
```json
{
  "status": "healthy",
  "service": "Camera Motion Evaluation", 
  "initialized": true
}
```

### 初始化服务

**POST** `/initialize`

初始化评测模型（首次使用必须调用）。

**请求参数:**
```json
{
  "device": "cuda:0",     // GPU设备，可选
  "local_mode": false     // 本地模式，可选
}
```

### 运镜预测

**POST** `/predict_motion`

分析视频运镜类型，无需期望运镜类型。

**请求参数:**
```json
{
  "video_path": "/path/to/video.mp4"
}
```

**响应示例:**
```json
{
  "status": "success",
  "video_path": "/path/to/video.mp4",
  "predicted_motions": ["pan_left", "tilt_up"],
  "motion_types": {
    "pan_left": true,
    "pan_right": false,
    "tilt_up": true,
    // ...
  }
}
```

### 单视频评测

**POST** `/evaluate_single`

评测单个视频的运镜效果。

**请求参数:**
```json
{
  "video_path": "/path/to/video.mp4",
  "expected_motion": "pan_left"     // 期望的运镜类型
}
```

**响应示例:**
```json
{
  "status": "success",
  "video_path": "/path/to/video.mp4",
  "predicted_motions": ["pan_left", "tilt_up"],
  "expected_motion": "pan_left",
  "score": 1.0,
  "match": true
}
```

### 批量评测

**POST** `/evaluate`

批量评测多个视频。

**请求参数:**
```json
{
  "video_paths": [
    "/path/to/video1.mp4",
    "/path/to/video2.mp4"
  ],
  "expected_motion": "pan_left"     // 可选，所有视频使用相同期望运镜
}
```

## 客户端使用示例

### 命令行客户端

```bash
# 健康检查
python3 client_example.py --action health

# 初始化服务
python3 client_example.py --action init --device cuda:0

# 预测运镜
python3 client_example.py --action predict --video_path /path/to/video.mp4

# 评测运镜
python3 client_example.py --action evaluate \
  --video_path /path/to/video.mp4 \
  --expected_motion pan_left

# 批量评测
python3 client_example.py --action batch_evaluate \
  --video_paths /path/to/video1.mp4 /path/to/video2.mp4 \
  --expected_motion pan_left

# 连接远程服务
python3 client_example.py --url http://192.168.1.100:5000 --action health
```

### Python 代码示例

```python
import requests

# 初始化服务
response = requests.post('http://localhost:5000/initialize', 
                        json={'device': 'cuda:0'})
print(response.json())

# 预测运镜
response = requests.post('http://localhost:5000/predict_motion',
                        json={'video_path': '/path/to/video.mp4'})
result = response.json()
print(f"预测运镜: {result['predicted_motions']}")

# 评测运镜
response = requests.post('http://localhost:5000/evaluate_single',
                        json={
                            'video_path': '/path/to/video.mp4',
                            'expected_motion': 'pan_left'
                        })
result = response.json()
print(f"评测分数: {result['score']}")
```

## 部署建议

### 生产环境

```bash
# 使用生产配置
./start_service.sh --env production --port 8080

# 使用 gunicorn (推荐)
pip install gunicorn
gunicorn -w 1 -b 0.0.0.0:8080 camera_motion_service:app

# 使用 supervisor 进程管理
sudo apt-get install supervisor
# 配置 supervisor 配置文件
```

### Docker 部署

```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements_service.txt .
RUN pip install -r requirements_service.txt

COPY . .
EXPOSE 5000

CMD ["python3", "camera_motion_service.py"]
```

### 反向代理 (Nginx)

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 性能优化

1. **GPU 内存管理**: 使用合适的批处理大小
2. **模型缓存**: 初始化后模型保持在内存中
3. **并发限制**: 建议单实例避免 GPU 内存冲突
4. **视频预处理**: 大视频文件建议预先压缩

## 故障排除

### 常见问题

1. **CUDA 内存不足**
   ```bash
   # 使用 CPU 模式
   ./start_service.sh --device cpu
   ```

2. **模型下载失败**
   ```bash
   # 使用本地模式
   ./start_service.sh --local
   ```

3. **端口被占用**
   ```bash
   # 更换端口
   ./start_service.sh --port 8080
   ```

### 日志查看

服务日志会输出到控制台，包含详细的评测过程信息。

## 开发指南

### 添加新接口

在 `camera_motion_service.py` 中添加新的路由：

```python
@app.route('/new_endpoint', methods=['POST'])
def new_endpoint():
    # 实现新功能
    pass
```

### 扩展功能

- 添加其他 VBench 评测维度
- 支持视频上传
- 添加结果缓存
- 集成数据库存储

## 许可证

本项目基于 VBench 项目，请遵循相应的开源许可证。

## 联系方式

如有问题或建议，请联系开发团队。 