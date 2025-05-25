#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wan2GP LoRA模型自动下载脚本
基于原版download_loras()函数实现
"""

import os
import shutil
import glob
import time
from pathlib import Path
from datetime import datetime

def download_loras_standalone(target_dir="loras_i2v", repo_id="DeepBeepMeep/Wan2.1"):
    """
    独立的LoRA下载函数
    
    Args:
        target_dir: 目标目录 ("loras" 或 "loras_i2v")
        repo_id: HuggingFace仓库ID
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ 需要安装 huggingface_hub: pip install huggingface_hub")
        return False
    
    # 确保目标目录存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"📁 创建目录: {target_dir}")
    
    # 检查是否已经下载过
    log_path = os.path.join(target_dir, "log.txt")
    if os.path.isfile(log_path):
        print(f"✅ LoRA模型已存在于 {target_dir}")
        print("如需重新下载，请删除 log.txt 文件")
        return True
    
    print(f"🚀 开始下载LoRA模型到 {target_dir}...")
    print(f"📦 从仓库下载: {repo_id}")
    
    # 临时下载目录
    tmp_path = os.path.join(target_dir, "tmp_lora_download")
    
    try:
        # 下载LoRA文件
        print("⬇️  正在下载，请稍候...")
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=f"{target_dir}/*",  # 只下载对应目录的文件
            local_dir=tmp_path
        )
        
        # 移动文件到目标目录
        source_path = os.path.join(tmp_path, target_dir)
        if os.path.exists(source_path):
            downloaded_files = 0
            for file_path in glob.glob(os.path.join(source_path, "*.*")):
                filename = Path(file_path).name
                target_file = os.path.join(target_dir, filename)
                
                if os.path.isfile(target_file):
                    print(f"⚠️  文件已存在，跳过: {filename}")
                    os.remove(file_path)
                else:
                    shutil.move(file_path, target_dir)
                    print(f"✅ 下载完成: {filename}")
                    downloaded_files += 1
            
            print(f"🎉 成功下载 {downloaded_files} 个LoRA文件")
        else:
            print(f"❌ 未找到 {target_dir} 目录在下载的文件中")
            return False
        
        # 清理临时目录
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)
        
        # 创建下载日志
        dt = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        with open(log_path, "w", encoding="utf-8") as writer:
            writer.write(f"LoRA模型下载于 {dt}")
        
        print(f"📝 下载日志已保存: {log_path}")
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        # 清理临时目录
        if os.path.exists(tmp_path):
            shutil.rmtree(tmp_path)
        return False

def download_all_loras():
    """下载所有类型的LoRA模型"""
    print("🎯 开始下载所有LoRA模型...")
    
    # 下载I2V LoRA
    success_i2v = download_loras_standalone("loras_i2v")
    
    # 下载T2V LoRA (如果仓库中有的话)
    success_t2v = download_loras_standalone("loras")
    
    if success_i2v or success_t2v:
        print("\n🎉 LoRA下载完成！")
        print("📍 文件位置:")
        if success_i2v:
            print(f"   - 图生视频LoRA: ./loras_i2v/")
        if success_t2v:
            print(f"   - 文生视频LoRA: ./loras/")
        print("\n💡 使用方法:")
        print("   1. 启动Wan2GP: python wgp.py")
        print("   2. 在界面中选择要使用的LoRA")
        print("   3. 设置权重并生成视频")
    else:
        print("❌ LoRA下载失败")

def list_downloaded_loras(directory="loras_i2v"):
    """列出已下载的LoRA模型"""
    if not os.path.exists(directory):
        print(f"❌ 目录不存在: {directory}")
        return
    
    lora_files = glob.glob(os.path.join(directory, "*.safetensors")) + \
                glob.glob(os.path.join(directory, "*.sft"))
    
    if lora_files:
        print(f"\n📋 {directory} 中的LoRA模型:")
        for i, file_path in enumerate(lora_files, 1):
            filename = Path(file_path).name
            file_size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"   {i:2d}. {filename} ({file_size:.1f} MB)")
    else:
        print(f"📭 {directory} 中没有找到LoRA文件")

def clean_download_cache():
    """清理下载缓存，强制重新下载"""
    directories = ["loras", "loras_i2v"]
    
    for directory in directories:
        log_path = os.path.join(directory, "log.txt")
        if os.path.exists(log_path):
            os.remove(log_path)
            print(f"🗑️  已删除下载日志: {log_path}")
    
    print("✅ 缓存清理完成，下次运行将重新下载")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Wan2GP LoRA自动下载工具")
    parser.add_argument("--download", action="store_true", help="下载LoRA模型")
    parser.add_argument("--list", action="store_true", help="列出已下载的LoRA")
    parser.add_argument("--clean", action="store_true", help="清理下载缓存")
    parser.add_argument("--dir", default="loras_i2v", help="指定目录 (loras 或 loras_i2v)")
    
    args = parser.parse_args()
    
    if args.download:
        if args.dir in ["loras", "loras_i2v"]:
            download_loras_standalone(args.dir)
        else:
            download_all_loras()
    elif args.list:
        list_downloaded_loras(args.dir)
    elif args.clean:
        clean_download_cache()
    else:
        print("🎯 Wan2GP LoRA下载工具")
        print("\n使用方法:")
        print("  python auto_download_loras.py --download     # 下载所有LoRA")
        print("  python auto_download_loras.py --download --dir loras_i2v  # 下载I2V LoRA")
        print("  python auto_download_loras.py --list         # 列出已下载的LoRA")
        print("  python auto_download_loras.py --clean        # 清理缓存")

if __name__ == "__main__":
    main() 