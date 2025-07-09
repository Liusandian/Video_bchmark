#!/usr/bin/env python3
"""
VBench Camera Motion 服务 API 测试脚本

用于验证服务的各个接口是否正常工作
"""

import requests
import json
import time
import argparse


class APITester:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
    def test_health_check(self):
        """测试健康检查接口"""
        print("🔍 测试健康检查接口...")
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()
            result = response.json()
            print(f"✅ 健康检查成功: {result}")
            return True
        except Exception as e:
            print(f"❌ 健康检查失败: {e}")
            return False
    
    def test_initialize(self, device="cuda:0", local_mode=False):
        """测试初始化接口"""
        print(f"🚀 测试初始化接口 (设备: {device}, 本地模式: {local_mode})...")
        try:
            data = {
                "device": device,
                "local_mode": local_mode
            }
            response = self.session.post(
                f"{self.base_url}/initialize",
                json=data,
                timeout=600  # 初始化可能需要很长时间
            )
            response.raise_for_status()
            result = response.json()
            print(f"✅ 初始化成功: {result}")
            return True
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            return False
    
    def test_predict_motion_mock(self):
        """测试运镜预测接口（模拟）"""
        print("🎬 测试运镜预测接口...")
        # 使用一个不存在的视频路径来测试接口逻辑
        test_video_path = "/test/fake_video.mp4"
        
        try:
            data = {"video_path": test_video_path}
            response = self.session.post(
                f"{self.base_url}/predict_motion",
                json=data,
                timeout=120
            )
            
            # 我们期望这里返回错误，因为文件不存在
            if response.status_code == 400:
                result = response.json()
                if "Video file not found" in result.get("message", ""):
                    print("✅ 运镜预测接口响应正确（文件不存在错误）")
                    return True
            
            print(f"❌ 运镜预测接口响应异常: {response.status_code}")
            return False
            
        except Exception as e:
            print(f"❌ 运镜预测接口测试失败: {e}")
            return False
    
    def test_evaluate_single_mock(self):
        """测试单视频评测接口（模拟）"""
        print("📊 测试单视频评测接口...")
        test_video_path = "/test/fake_video.mp4"
        
        try:
            data = {
                "video_path": test_video_path,
                "expected_motion": "pan_left"
            }
            response = self.session.post(
                f"{self.base_url}/evaluate_single",
                json=data,
                timeout=120
            )
            
            # 我们期望这里返回错误，因为文件不存在
            if response.status_code == 400:
                result = response.json()
                if "Video file not found" in result.get("message", ""):
                    print("✅ 单视频评测接口响应正确（文件不存在错误）")
                    return True
            
            print(f"❌ 单视频评测接口响应异常: {response.status_code}")
            return False
            
        except Exception as e:
            print(f"❌ 单视频评测接口测试失败: {e}")
            return False
    
    def test_evaluate_batch_mock(self):
        """测试批量评测接口（模拟）"""
        print("📹 测试批量评测接口...")
        test_video_paths = ["/test/fake_video1.mp4", "/test/fake_video2.mp4"]
        
        try:
            data = {
                "video_paths": test_video_paths,
                "expected_motion": "pan_left"
            }
            response = self.session.post(
                f"{self.base_url}/evaluate",
                json=data,
                timeout=300
            )
            
            # 我们期望这里返回错误，因为文件不存在
            if response.status_code == 400:
                result = response.json()
                if "Video file not found" in result.get("message", ""):
                    print("✅ 批量评测接口响应正确（文件不存在错误）")
                    return True
            
            print(f"❌ 批量评测接口响应异常: {response.status_code}")
            return False
            
        except Exception as e:
            print(f"❌ 批量评测接口测试失败: {e}")
            return False
    
    def test_invalid_endpoints(self):
        """测试无效端点"""
        print("🚫 测试无效端点...")
        try:
            response = self.session.get(f"{self.base_url}/invalid_endpoint")
            if response.status_code == 404:
                print("✅ 无效端点正确返回 404")
                return True
            else:
                print(f"❌ 无效端点返回异常状态码: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ 无效端点测试失败: {e}")
            return False
    
    def run_all_tests(self, device="cuda:0", local_mode=False, skip_init=False):
        """运行所有测试"""
        print("=" * 50)
        print("🧪 VBench Camera Motion 服务 API 测试")
        print("=" * 50)
        
        tests = []
        
        # 健康检查
        tests.append(("健康检查", self.test_health_check))
        
        # 初始化（如果不跳过）
        if not skip_init:
            tests.append(("初始化服务", lambda: self.test_initialize(device, local_mode)))
        
        # 接口测试
        tests.append(("运镜预测", self.test_predict_motion_mock))
        tests.append(("单视频评测", self.test_evaluate_single_mock))
        tests.append(("批量评测", self.test_evaluate_batch_mock))
        tests.append(("无效端点", self.test_invalid_endpoints))
        
        # 运行测试
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            print(f"\n--- 测试: {test_name} ---")
            try:
                if test_func():
                    passed += 1
                    print(f"✅ {test_name} 通过")
                else:
                    print(f"❌ {test_name} 失败")
            except Exception as e:
                print(f"❌ {test_name} 异常: {e}")
            
            time.sleep(1)  # 短暂延迟
        
        # 总结
        print("\n" + "=" * 50)
        print(f"🎯 测试总结: {passed}/{total} 通过")
        
        if passed == total:
            print("🎉 所有测试通过！服务运行正常")
            return True
        else:
            print(f"⚠️  有 {total - passed} 个测试失败")
            return False


def main():
    parser = argparse.ArgumentParser(description="VBench Camera Motion API 测试工具")
    parser.add_argument("--url", default="http://localhost:5000", help="服务地址")
    parser.add_argument("--device", default="cuda:0", help="GPU 设备")
    parser.add_argument("--local", action="store_true", help="本地模式")
    parser.add_argument("--skip-init", action="store_true", help="跳过初始化测试")
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    
    success = tester.run_all_tests(
        device=args.device,
        local_mode=args.local,
        skip_init=args.skip_init
    )
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main() 