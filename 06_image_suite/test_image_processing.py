#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VBench图像处理套件测试脚本
"""

import os
import json
import tempfile
from PIL import Image
import sys

# 添加脚本目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from image_process_suite import ImageProcessor

def create_test_image(width, height, filename):
    """创建测试图像"""
    # 创建一个简单的测试图像
    img = Image.new('RGB', (width, height), color='blue')
    img.save(filename)
    return filename

def test_image_classification():
    """测试图像分类功能"""
    print("=== 测试图像分类功能 ===")
    processor = ImageProcessor()
    
    test_cases = [
        ("a beautiful mountain landscape with trees", "scenery"),
        ("a young woman smiling", "single_person"),
        ("group of people at a party", "multiple_person"),
        ("modern skyscrapers in the city", "architecture"),
        ("cute cat playing in the garden", "animals"),
        ("delicious pizza on a plate", "food"),
        ("colorful flowers in bloom", "plant"),
        ("abstract art with geometric patterns", "abstract")
    ]
    
    for caption, expected in test_cases:
        result = processor.classify_image_type(caption)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{caption}' -> {result} (期望: {expected})")

def test_crop_calculations():
    """测试裁剪计算功能"""
    print("\n=== 测试裁剪计算功能 ===")
    processor = ImageProcessor()
    
    # 测试风景图像 (4000x3000)
    print("测试风景图像 (4000x3000):")
    first_crop = processor.calculate_first_crop(4000, 3000)
    print(f"第一次裁剪: {first_crop}")
    
    second_crop = processor.calculate_second_crop(first_crop, "scenery")
    print(f"第二次裁剪: {second_crop}")
    
    first_offset = (first_crop["first_bbox"][0], first_crop["first_bbox"][1])
    diff_crops = processor.calculate_different_ratio_crops(second_crop, first_offset)
    print(f"不同比例裁剪: {diff_crops}")
    
    # 测试人像图像 (3000x4000)
    print("\n测试人像图像 (3000x4000):")
    first_crop = processor.calculate_first_crop(3000, 4000)
    print(f"第一次裁剪: {first_crop}")
    
    second_crop = processor.calculate_second_crop(first_crop, "single_person")
    print(f"第二次裁剪: {second_crop}")
    
    first_offset = (first_crop["first_bbox"][0], first_crop["first_bbox"][1])
    diff_crops = processor.calculate_different_ratio_crops(second_crop, first_offset)
    print(f"不同比例裁剪: {diff_crops}")

def test_full_processing():
    """测试完整处理流程"""
    print("\n=== 测试完整处理流程 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        processor = ImageProcessor()
        
        # 创建测试图像
        test_images = [
            (4000, 3000, "landscape.jpg", "beautiful mountain landscape"),
            (3000, 4000, "portrait.jpg", "young woman smiling"),
            (2000, 2000, "square.jpg", "modern architecture building")
        ]
        
        for width, height, filename, caption in test_images:
            image_path = os.path.join(temp_dir, filename)
            create_test_image(width, height, image_path)
            
            print(f"\n处理图像: {filename} ({width}x{height})")
            result = processor.process_single_image(image_path, caption)
            
            if result:
                print(f"  类型: {result['type']}")
                print(f"  第一次裁剪: {result['first_crop']['first_bbox']}")
                print(f"  第二次裁剪: {result['second_crop']['second_bbox']}")
                print(f"  1:1 比例: {result['diff_ratio_crop']['1-1']}")
                print(f"  16:9 比例: {result['diff_ratio_crop']['16-9']}")
                
                # 测试实际裁剪
                crop_info = result['diff_ratio_crop']['1-1']
                output_path = os.path.join(temp_dir, f"cropped_{filename}")
                success = processor.crop_image_to_ratio(image_path, crop_info, output_path)
                print(f"  裁剪测试: {'成功' if success else '失败'}")
                
                if success and os.path.exists(output_path):
                    with Image.open(output_path) as cropped_img:
                        print(f"  裁剪后尺寸: {cropped_img.size}")
            else:
                print("  处理失败")

def test_batch_processing():
    """测试批量处理功能"""
    print("\n=== 测试批量处理功能 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        processor = ImageProcessor()
        
        # 创建测试图像目录
        image_dir = os.path.join(temp_dir, "images")
        os.makedirs(image_dir)
        
        # 创建测试图像
        test_images = [
            (4000, 3000, "landscape1.jpg"),
            (3000, 4000, "portrait1.jpg"),
            (2000, 1500, "landscape2.jpg")
        ]
        
        for width, height, filename in test_images:
            image_path = os.path.join(image_dir, filename)
            create_test_image(width, height, image_path)
        
        # 创建元数据文件
        metadata = {
            "landscape1.jpg": {
                "caption": "beautiful mountain landscape with trees",
                "url": "www.example.com/landscape1"
            },
            "portrait1.jpg": {
                "caption": "young woman smiling in the garden",
                "url": "www.example.com/portrait1"
            },
            "landscape2.jpg": {
                "caption": "sunset over the ocean",
                "url": "www.example.com/landscape2"
            }
        }
        
        metadata_file = os.path.join(temp_dir, "metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 批量处理
        output_json = os.path.join(temp_dir, "image_info.json")
        processor.process_image_batch(image_dir, metadata_file, output_json)
        
        # 检查结果
        if os.path.exists(output_json):
            with open(output_json, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"批量处理成功，处理了 {len(results)} 张图像")
            
            for result in results:
                print(f"  {result['file_name']}: {result['type']}")
        else:
            print("批量处理失败")

def main():
    """主测试函数"""
    print("VBench图像处理套件测试")
    print("=" * 50)
    
    try:
        test_image_classification()
        test_crop_calculations()
        test_full_processing()
        test_batch_processing()
        
        print("\n" + "=" * 50)
        print("所有测试完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 