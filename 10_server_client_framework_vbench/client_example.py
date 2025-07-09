#!/usr/bin/env python3
"""
VBench Camera Motion 评测服务客户端示例

使用示例：
1. 健康检查: python client_example.py --action health
2. 初始化服务: python client_example.py --action init
3. 预测运镜: python client_example.py --action predict --video_path /path/to/video.mp4
4. 评测运镜: python client_example.py --action evaluate --video_path /path/to/video.mp4 --expected_motion pan_left
"""

import argparse
import json
import requests
import os
import time
from typing import List, Optional

class CameraMotionClient:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
    def health_check(self) -> dict:
        """健康检查"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": str(e)}
    
    def initialize_service(self, device: str = "cuda:0", local_mode: bool = False) -> dict:
        """初始化服务"""
        try:
            data = {
                "device": device,
                "local_mode": local_mode
            }
            response = self.session.post(
                f"{self.base_url}/initialize", 
                json=data, 
                timeout=300  # 初始化可能需要较长时间
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": str(e)}
    
    def predict_motion(self, video_path: str) -> dict:
        """预测视频运镜类型"""
        try:
            data = {"video_path": video_path}
            response = self.session.post(
                f"{self.base_url}/predict_motion", 
                json=data, 
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": str(e)}
    
    def evaluate_single(self, video_path: str, expected_motion: str = "unknown") -> dict:
        """评测单个视频"""
        try:
            data = {
                "video_path": video_path,
                "expected_motion": expected_motion
            }
            response = self.session.post(
                f"{self.base_url}/evaluate_single", 
                json=data, 
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": str(e)}
    
    def evaluate_batch(self, video_paths: List[str], expected_motion: Optional[str] = None) -> dict:
        """批量评测视频"""
        try:
            data = {"video_paths": video_paths}
            if expected_motion:
                data["expected_motion"] = expected_motion
                
            response = self.session.post(
                f"{self.base_url}/evaluate", 
                json=data, 
                timeout=300
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"status": "error", "message": str(e)}

def print_result(result: dict, action: str):
    """格式化打印结果"""
    print(f"\n=== {action.upper()} 结果 ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    if result.get("status") == "success":
        print(f"\n✅ {action} 成功")
        
        if action == "predict" and "predicted_motions" in result:
            motions = result["predicted_motions"]
            print(f"🎬 检测到的运镜类型: {', '.join(motions)}")
            
        elif action == "evaluate" and "score" in result:
            score = result["score"]
            match = result.get("match", False)
            print(f"📊 评测分数: {score}")
            print(f"🎯 运镜匹配: {'是' if match else '否'}")
            
        elif action == "batch_evaluate" and "overall_score" in result:
            score = result["overall_score"]
            count = result.get("video_count", 0)
            print(f"📊 总体评分: {score}")
            print(f"📹 视频数量: {count}")
            
    else:
        print(f"\n❌ {action} 失败: {result.get('message', '未知错误')}")

def main():
    parser = argparse.ArgumentParser(description="VBench Camera Motion 评测客户端")
    parser.add_argument("--url", default="http://localhost:5000", help="服务端地址")
    parser.add_argument("--action", required=True, choices=["health", "init", "predict", "evaluate", "batch_evaluate"], 
                       help="操作类型")
    parser.add_argument("--video_path", help="视频文件路径")
    parser.add_argument("--video_paths", nargs="+", help="多个视频文件路径（批量评测）")
    parser.add_argument("--expected_motion", help="期望的运镜类型")
    parser.add_argument("--device", default="cuda:0", help="设备类型")
    parser.add_argument("--local_mode", action="store_true", help="本地模式")
    
    args = parser.parse_args()
    
    # 创建客户端
    client = CameraMotionClient(args.url)
    
    # 根据操作类型执行相应功能
    if args.action == "health":
        result = client.health_check()
        print_result(result, "health_check")
        
    elif args.action == "init":
        print("🔄 正在初始化服务，请稍等...")
        result = client.initialize_service(args.device, args.local_mode)
        print_result(result, "initialize")
        
    elif args.action == "predict":
        if not args.video_path:
            print("❌ 错误: --video_path 参数是必需的")
            return
        if not os.path.exists(args.video_path):
            print(f"❌ 错误: 视频文件不存在: {args.video_path}")
            return
            
        print(f"🔍 正在预测视频运镜: {args.video_path}")
        result = client.predict_motion(args.video_path)
        print_result(result, "predict")
        
    elif args.action == "evaluate":
        if not args.video_path:
            print("❌ 错误: --video_path 参数是必需的")
            return
        if not os.path.exists(args.video_path):
            print(f"❌ 错误: 视频文件不存在: {args.video_path}")
            return
            
        expected_motion = args.expected_motion or "unknown"
        print(f"📊 正在评测视频: {args.video_path}")
        print(f"🎯 期望运镜: {expected_motion}")
        result = client.evaluate_single(args.video_path, expected_motion)
        print_result(result, "evaluate")
        
    elif args.action == "batch_evaluate":
        if not args.video_paths:
            print("❌ 错误: --video_paths 参数是必需的")
            return
            
        # 检查所有视频文件是否存在
        missing_files = [path for path in args.video_paths if not os.path.exists(path)]
        if missing_files:
            print(f"❌ 错误: 以下视频文件不存在:")
            for file in missing_files:
                print(f"  - {file}")
            return
            
        print(f"📊 正在批量评测 {len(args.video_paths)} 个视频...")
        result = client.evaluate_batch(args.video_paths, args.expected_motion)
        print_result(result, "batch_evaluate")

if __name__ == "__main__":
    main() 