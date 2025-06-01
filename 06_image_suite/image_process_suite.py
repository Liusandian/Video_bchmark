# 图像评测前预处理流程

import json
import os
import random
from PIL import Image
import argparse
from typing import Dict, List, Tuple, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    VBench图像处理套件
    实现基于论文的图像裁剪流水线，支持风景和人像图像的不同处理逻辑
    """
    
    def __init__(self):
        # 支持的宽高比
        self.supported_ratios = {
            "1-1": (1, 1),      # 1:1 正方形
            "8-5": (8, 5),      # 8:5 
            "7-4": (7, 4),      # 7:4
            "16-9": (16, 9)     # 16:9 宽屏
        }
        
        # 图像类型分类
        self.image_types = {
            "scenery": ["scenery", "landscape", "nature", "outdoor", "mountain", "lake", "forest", "beach", "ocean", "sky"],
            "architecture": ["architecture", "building", "city", "urban", "house", "tower", "bridge", "street", "village", "cliff"],
            "single_person": ["person", "portrait", "man", "woman", "child", "face"],
            "multiple_person": ["people", "group", "crowd", "family", "team"],
            "animals": ["animal", "pet", "wildlife", "cat", "dog", "bird", "elephant", "squirrel"],
            "food": ["food", "meal", "cooking", "restaurant", "bread", "dish"],
            "plant": ["plant", "flower", "tree", "garden", "grass", "leaves"],
            "abstract": ["abstract", "art", "pattern"]
        }
    
    def determine_image_orientation(self, width: int, height: int) -> str:
        """
        判断图像方向：风景(landscape)或人像(portrait)
        
        Args:
            width: 图像宽度
            height: 图像高度
            
        Returns:
            str: "landscape" 或 "portrait"
        """
        if width > height:
            return "landscape"
        else:
            return "portrait"
    
    def classify_image_type(self, caption: str, url: str = "") -> str:
        """
        基于描述和URL自动分类图像类型
        
        Args:
            caption: 图像描述
            url: 图像来源URL
            
        Returns:
            str: 图像类型
        """
        caption_lower = caption.lower()
        url_lower = url.lower()
        
        # 检查人物相关关键词
        person_keywords = ["person", "people", "man", "woman", "child", "human", "face"]
        person_count = sum(1 for keyword in person_keywords if keyword in caption_lower)
        
        if person_count > 0:
            # 判断单人还是多人
            multiple_keywords = ["people", "group", "crowd", "family", "team"]
            if any(keyword in caption_lower for keyword in multiple_keywords):
                return "multiple_person"
            else:
                return "single_person"
        
        # 检查其他类型
        for img_type, keywords in self.image_types.items():
            if img_type in ["single_person", "multiple_person"]:
                continue
            if any(keyword in caption_lower for keyword in keywords):
                return img_type
        
        # 默认返回scenery
        return "scenery"
    
    def calculate_first_crop(self, width: int, height: int) -> Dict:
        """
        第一次裁剪：裁剪到16:9比例
        
        Args:
            width: 原始图像宽度
            height: 原始图像高度
            
        Returns:
            Dict: 包含裁剪信息的字典
        """
        target_ratio = 16 / 9
        
        if width / height > target_ratio:
            # 图像太宽，需要裁剪宽度
            new_width = int(height * target_ratio)
            new_height = height
            x = (width - new_width) // 2
            y = 0
        else:
            # 图像太高，需要裁剪高度
            new_width = width
            new_height = int(width / target_ratio)
            x = 0
            y = (height - new_height) // 2
        
        return {
            "width": width,
            "height": height,
            "first_bbox": [x, y, new_width, new_height]
        }
    
    def calculate_second_crop(self, first_crop_info: Dict, image_type: str) -> Dict:
        """
        第二次裁剪：基于图像类型裁剪到1:1比例
        
        Args:
            first_crop_info: 第一次裁剪的信息
            image_type: 图像类型
            
        Returns:
            Dict: 包含第二次裁剪信息的字典
        """
        x, y, crop_w, crop_h = first_crop_info["first_bbox"]
        
        # 确定正方形的边长（取较小值）
        square_size = min(crop_w, crop_h)
        
        if image_type in ["scenery", "architecture", "plant", "abstract"]:
            # 风景类图像：居中裁剪
            new_x = (crop_w - square_size) // 2
            new_y = (crop_h - square_size) // 2
        else:
            # 人物、动物、食物类图像：智能裁剪（偏向上方或中心）
            new_x = (crop_w - square_size) // 2
            # 对于人物图像，稍微偏向上方
            if image_type in ["single_person", "multiple_person"]:
                new_y = max(0, (crop_h - square_size) // 3)
            else:
                new_y = (crop_h - square_size) // 2
        
        return {
            "width": crop_w,
            "height": crop_h,
            "second_bbox": [new_x, new_y, square_size, square_size]
        }
    
    def calculate_different_ratio_crops(self, second_crop_info: Dict, 
                                      first_crop_offset: Tuple[int, int]) -> Dict:
        """
        计算不同宽高比的裁剪区域
        
        Args:
            second_crop_info: 第二次裁剪信息
            first_crop_offset: 第一次裁剪的偏移量 (offset_x, offset_y)
            
        Returns:
            Dict: 包含不同比例裁剪信息的字典
        """
        random.seed(123)  # 确保结果可重现
        
        width, height = second_crop_info['width'], second_crop_info['height']
        x, y, crop_w, crop_h = second_crop_info['second_bbox']
        offset_x, offset_y = first_crop_offset
        
        diff_ratio_crops = {}
        
        for ratio_name, (ratio_w, ratio_h) in self.supported_ratios.items():
            if ratio_name == "1-1":
                # 1:1 比例直接使用第二次裁剪结果
                diff_ratio_crops[ratio_name] = [
                    x + offset_x, y + offset_y, crop_w, crop_h
                ]
            else:
                # 计算其他比例
                crop_info = self._calculate_ratio_crop(
                    second_crop_info, ratio_w, ratio_h, offset_x, offset_y
                )
                diff_ratio_crops[ratio_name] = crop_info
        
        return diff_ratio_crops
    
    def _calculate_ratio_crop(self, second_crop_info: Dict, ratio_w: int, ratio_h: int,
                            offset_x: int, offset_y: int) -> List[int]:
        """
        计算特定比例的裁剪区域
        
        Args:
            second_crop_info: 第二次裁剪信息
            ratio_w: 目标宽度比例
            ratio_h: 目标高度比例
            offset_x: X轴偏移
            offset_y: Y轴偏移
            
        Returns:
            List[int]: [x, y, width, height]
        """
        width, height = second_crop_info['width'], second_crop_info['height']
        x, y, crop_w, crop_h = second_crop_info['second_bbox']
        
        # 计算目标尺寸
        if width == height:  # 正方形图像
            target_w = int(width / ratio_w) * ratio_w
            target_h = int(width / ratio_w) * ratio_h
            
            if target_h <= crop_h:
                target_x = 0
                y_min = max(y - (target_h - crop_h), 0)
                y_max = min(y + target_h, height) - target_h
                target_y = random.randint(max(0, y_min), max(0, y_max)) if y_max >= y_min else 0
            else:
                # 如果目标高度超出，调整到最大可能
                target_h = height
                target_w = int(target_h * ratio_w / ratio_h)
                target_x = (width - target_w) // 2
                target_y = 0
        else:  # 矩形图像
            target_w = int(height / ratio_h) * ratio_w
            target_h = int(height / ratio_h) * ratio_h
            
            if target_w <= crop_w:
                target_y = 0
                x_min = max(x - (target_w - crop_w), 0)
                x_max = min(x + target_w, width) - target_w
                target_x = random.randint(max(0, x_min), max(0, x_max)) if x_max >= x_min else 0
            else:
                # 如果目标宽度超出，调整到最大可能
                target_w = width
                target_h = int(target_w * ratio_h / ratio_w)
                target_x = 0
                target_y = (height - target_h) // 2
        
        return [target_x + offset_x, target_y + offset_y, target_w, target_h]
    
    def process_single_image(self, image_path: str, caption: str, url: str = "") -> Dict:
        """
        处理单张图像，生成完整的裁剪信息
        
        Args:
            image_path: 图像文件路径
            caption: 图像描述
            url: 图像来源URL
            
        Returns:
            Dict: 完整的图像处理信息，格式与截图一致
        """
        try:
            # 打开图像获取尺寸
            with Image.open(image_path) as img:
                width, height = img.size
            
            # 分类图像类型
            image_type = self.classify_image_type(caption, url)
            
            # 第一次裁剪（16:9）
            first_crop = self.calculate_first_crop(width, height)
            
            # 第二次裁剪（1:1）
            second_crop = self.calculate_second_crop(first_crop, image_type)
            
            # 计算不同比例的裁剪
            first_offset = (first_crop["first_bbox"][0], first_crop["first_bbox"][1])
            diff_ratio_crops = self.calculate_different_ratio_crops(second_crop, first_offset)
            
            # 构建完整信息，格式与截图完全一致
            result = {
                "file_name": os.path.basename(image_path),
                "url": url,
                "type": image_type,
                "origin_width": width,
                "origin_height": height,
                "first_crop": {
                    "width": width,
                    "height": height,
                    "first_bbox": first_crop["first_bbox"]
                },
                "second_crop": {
                    "width": first_crop["first_bbox"][2],  # 第一次裁剪后的宽度
                    "height": first_crop["first_bbox"][3], # 第一次裁剪后的高度
                    "second_bbox": second_crop["second_bbox"]
                },
                "diff_ratio_crop": diff_ratio_crops,
                "caption": caption
            }
            
            logger.info(f"处理完成: {os.path.basename(image_path)} - 类型: {image_type}")
            return result
            
        except Exception as e:
            logger.error(f"处理图像失败 {image_path}: {str(e)}")
            return None
    
    def crop_image_to_ratio(self, image_path: str, crop_info: List[int], 
                          output_path: str) -> bool:
        """
        根据裁剪信息裁剪图像
        
        Args:
            image_path: 原始图像路径
            crop_info: 裁剪信息 [x, y, width, height]
            output_path: 输出路径
            
        Returns:
            bool: 是否成功
        """
        try:
            with Image.open(image_path) as img:
                x, y, width, height = crop_info
                cropped = img.crop((x, y, x + width, y + height))
                cropped.save(output_path)
                return True
        except Exception as e:
            logger.error(f"裁剪图像失败: {str(e)}")
            return False
    
    def process_image_batch(self, image_dir: str, metadata_file: str, 
                          output_json: str) -> None:
        """
        批量处理图像
        
        Args:
            image_dir: 图像目录
            metadata_file: 元数据文件（包含caption等信息）
            output_json: 输出JSON文件路径
        """
        # 读取元数据
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        results = []
        
        # 遍历图像目录
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(image_dir, filename)
                
                # 获取元数据
                file_metadata = metadata.get(filename, {})
                caption = file_metadata.get('caption', filename)
                url = file_metadata.get('url', '')
                
                # 处理图像
                result = self.process_single_image(image_path, caption, url)
                if result:
                    results.append(result)
        
        # 保存结果
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"批量处理完成，共处理 {len(results)} 张图像")
    
    def crop_images_by_ratio(self, info_json: str, image_dir: str, 
                           output_dir: str, target_ratio: str) -> None:
        """
        根据指定比例批量裁剪图像
        
        Args:
            info_json: 图像信息JSON文件
            image_dir: 原始图像目录
            output_dir: 输出目录
            target_ratio: 目标比例 (如 "16-9", "1-1")
        """
        with open(info_json, 'r', encoding='utf-8') as f:
            image_infos = json.load(f)
        
        os.makedirs(output_dir, exist_ok=True)
        
        success_count = 0
        for info in image_infos:
            filename = info['file_name']
            image_path = os.path.join(image_dir, filename)
            
            if not os.path.exists(image_path):
                logger.warning(f"图像文件不存在: {image_path}")
                continue
            
            # 获取裁剪信息
            if target_ratio in info['diff_ratio_crop']:
                crop_info = info['diff_ratio_crop'][target_ratio]
                
                # 生成输出文件名
                name, ext = os.path.splitext(filename)
                output_filename = f"{name}_{target_ratio}{ext}"
                output_path = os.path.join(output_dir, output_filename)
                
                # 裁剪图像
                if self.crop_image_to_ratio(image_path, crop_info, output_path):
                    success_count += 1
                    logger.info(f"裁剪成功: {output_filename}")
        
        logger.info(f"批量裁剪完成，成功处理 {success_count} 张图像")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='VBench图像处理套件')
    parser.add_argument('--mode', choices=['process', 'crop'], required=True,
                       help='运行模式：process(处理图像生成信息) 或 crop(根据信息裁剪图像)')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='图像目录路径')
    parser.add_argument('--metadata_file', type=str,
                       help='元数据文件路径（process模式）')
    parser.add_argument('--info_json', type=str,
                       help='图像信息JSON文件路径（crop模式）')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录路径')
    parser.add_argument('--target_ratio', type=str, choices=['1-1', '8-5', '7-4', '16-9'],
                       help='目标裁剪比例（crop模式）')
    
    args = parser.parse_args()
    
    processor = ImageProcessor()
    
    if args.mode == 'process':
        # 处理图像生成信息
        metadata_file = args.metadata_file or 'metadata.json'
        output_json = os.path.join(args.output_dir, 'image_info.json')
        os.makedirs(args.output_dir, exist_ok=True)
        processor.process_image_batch(args.image_dir, metadata_file, output_json)
        
    elif args.mode == 'crop':
        # 根据信息裁剪图像
        if not args.info_json or not args.target_ratio:
            print("crop模式需要指定 --info_json 和 --target_ratio 参数")
            return
        processor.crop_images_by_ratio(args.info_json, args.image_dir, 
                                     args.output_dir, args.target_ratio)


if __name__ == '__main__':
    main()
