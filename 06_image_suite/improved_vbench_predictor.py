#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进版VBench Dimension和Image Type预测器
基于真实数据分析结果优化预测准确性
"""

import json
import re
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter

class ImprovedVBenchPredictor:
    """改进版VBench维度和图像类型预测器"""
    
    def __init__(self):
        # 基于真实数据分析的关键词映射
        self.load_real_data_patterns()
        
        # Camera motion关键词
        self.camera_motion_keywords = [
            'camera pans left', 'camera pans right', 
            'camera tilts up', 'camera tilts down',
            'camera zooms in', 'camera zooms out', 
            'camera static'
        ]
        
        # 维度分配规则
        self.background_types = ['abstract', 'architecture', 'indoor', 'scenery']
        self.subject_types = ['animal', 'transportation', 'food', 'other', 'plant', 'single-human', 'multiple-human']
        
        # 基础维度集合
        self.base_background_dimensions = [
            'i2v_background', 'background_consistency', 
            'aesthetic_quality', 'imaging_quality', 'temporal_flickering'
        ]
        
        self.base_subject_dimensions = [
            'i2v_subject', 'subject_consistency', 'motion_smoothness',
            'dynamic_degree', 'aesthetic_quality', 'imaging_quality', 'temporal_flickering'
        ]
        
        self.camera_motion_dimension = ['camera_motion']
    
    def load_real_data_patterns(self):
        """从真实数据中学习模式"""
        try:
            with open('vbench2_beta_i2v/vbench2_i2v_full_info.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 分析每个类型的真实关键词
            self.type_patterns = defaultdict(lambda: {'keywords': Counter(), 'phrases': Counter()})
            
            for item in data:
                if 'camera' not in item['prompt_en']:  # 只分析基础prompt
                    prompt = item['prompt_en'].lower()
                    img_type = item['image_type']
                    
                    # 提取单词
                    words = re.findall(r'\b\w+\b', prompt)
                    self.type_patterns[img_type]['keywords'].update(words)
                    
                    # 提取短语（2-3个词的组合）
                    for i in range(len(words) - 1):
                        phrase = ' '.join(words[i:i+2])
                        self.type_patterns[img_type]['phrases'][phrase] += 1
                        if i < len(words) - 2:
                            phrase3 = ' '.join(words[i:i+3])
                            self.type_patterns[img_type]['phrases'][phrase3] += 1
            
            # 构建优化的关键词映射
            self.build_optimized_keywords()
            
        except FileNotFoundError:
            # 如果文件不存在，使用默认关键词
            self.use_default_keywords()
    
    def build_optimized_keywords(self):
        """构建优化的关键词映射"""
        self.image_type_keywords = {}
        self.image_type_phrases = {}
        
        for img_type, patterns in self.type_patterns.items():
            # 取高频关键词（出现次数>=2）
            keywords = [word for word, count in patterns['keywords'].most_common(50) if count >= 2]
            phrases = [phrase for phrase, count in patterns['phrases'].most_common(20) if count >= 2]
            
            self.image_type_keywords[img_type] = keywords
            self.image_type_phrases[img_type] = phrases
        
        # 特殊规则优化
        self.optimize_special_rules()
    
    def optimize_special_rules(self):
        """优化特殊规则"""
        # 人物检测关键词
        self.human_keywords = {
            'single': ['man', 'woman', 'person', 'individual', 'he', 'she', 'boy', 'girl', 'child'],
            'multiple': ['people', 'group', 'crowd', 'family', 'team', 'couple', 'friends', 'children', 'two', 'three', 'several']
        }
        
        # 动物关键词
        self.animal_keywords = [
            'cat', 'dog', 'bird', 'elephant', 'horse', 'cow', 'lion', 'tiger', 
            'bear', 'deer', 'rabbit', 'fish', 'whale', 'dolphin', 'butterfly', 
            'bee', 'spider', 'snake', 'animal', 'pet', 'wildlife'
        ]
        
        # 交通工具关键词
        self.transportation_keywords = [
            'car', 'truck', 'bus', 'train', 'plane', 'boat', 'ship', 'bicycle', 
            'motorcycle', 'vehicle', 'driving', 'flying', 'sailing'
        ]
        
        # 食物关键词
        self.food_keywords = [
            'food', 'cooking', 'meal', 'dish', 'plate', 'kitchen', 'restaurant', 
            'eating', 'bread', 'cake', 'fruit', 'vegetable', 'pizza', 'burger'
        ]
        
        # 植物关键词
        self.plant_keywords = [
            'flower', 'tree', 'plant', 'garden', 'grass', 'leaf', 'rose', 
            'tulip', 'forest', 'botanical', 'flora', 'bloom'
        ]
        
        # 建筑关键词
        self.architecture_keywords = [
            'building', 'house', 'castle', 'tower', 'bridge', 'city', 'skyline', 
            'street', 'village', 'architecture', 'urban', 'skyscraper'
        ]
        
        # 室内关键词
        self.indoor_keywords = [
            'room', 'living', 'kitchen', 'bedroom', 'office', 'indoor', 'interior', 
            'furniture', 'table', 'chair', 'sofa', 'bed'
        ]
        
        # 风景关键词
        self.scenery_keywords = [
            'landscape', 'mountain', 'lake', 'forest', 'beach', 'ocean', 'sunset', 
            'sunrise', 'valley', 'hill', 'river', 'field', 'nature', 'outdoor', 'scenic'
        ]
        
        # 抽象关键词
        self.abstract_keywords = [
            'abstract', 'liquid', 'smoke', 'bubbles', 'pattern', 'swirly', 
            'mesmerizing', 'geometric', 'texture', 'close', 'up'
        ]
    
    def use_default_keywords(self):
        """使用默认关键词（当无法加载真实数据时）"""
        self.image_type_keywords = {
            'abstract': ['abstract', 'liquid', 'smoke', 'bubbles', 'pattern', 'swirly', 'mesmerizing'],
            'architecture': ['building', 'house', 'castle', 'tower', 'bridge', 'city', 'skyline'],
            'indoor': ['room', 'living', 'kitchen', 'bedroom', 'office', 'indoor', 'interior'],
            'scenery': ['landscape', 'mountain', 'lake', 'forest', 'beach', 'ocean', 'sunset'],
            'single-human': ['man', 'woman', 'person', 'individual', 'he', 'she', 'boy', 'girl'],
            'multiple-human': ['people', 'group', 'crowd', 'family', 'team', 'couple'],
            'animal': ['animal', 'dog', 'cat', 'bird', 'elephant', 'horse', 'cow'],
            'transportation': ['car', 'truck', 'bus', 'train', 'plane', 'boat', 'ship'],
            'food': ['food', 'cooking', 'meal', 'dish', 'plate', 'kitchen'],
            'plant': ['flower', 'tree', 'plant', 'garden', 'grass', 'leaf'],
            'other': ['object', 'item', 'thing', 'tool', 'equipment']
        }
        self.image_type_phrases = {k: [] for k in self.image_type_keywords.keys()}
        
        # 确保特殊关键词属性也被初始化
        self.optimize_special_rules()
    
    def calculate_type_score(self, prompt: str, img_type: str) -> float:
        """计算类型匹配分数"""
        prompt_lower = prompt.lower()
        words = re.findall(r'\b\w+\b', prompt_lower)
        
        score = 0.0
        
        # 关键词匹配
        if img_type in self.image_type_keywords:
            for keyword in self.image_type_keywords[img_type]:
                if keyword in words:
                    score += 2.0
                elif keyword in prompt_lower:
                    score += 1.0
        
        # 短语匹配
        if img_type in self.image_type_phrases:
            for phrase in self.image_type_phrases[img_type]:
                if phrase in prompt_lower:
                    score += 3.0
        
        # 特殊规则加分
        score += self.apply_special_rules(prompt_lower, img_type)
        
        return score
    
    def apply_special_rules(self, prompt_lower: str, img_type: str) -> float:
        """应用特殊规则"""
        score = 0.0
        
        # 人物检测
        if img_type == 'single-human':
            if any(word in prompt_lower for word in self.human_keywords['single']):
                score += 5.0
            # 如果包含多人关键词，减分
            if any(word in prompt_lower for word in self.human_keywords['multiple']):
                score -= 3.0
        
        elif img_type == 'multiple-human':
            if any(word in prompt_lower for word in self.human_keywords['multiple']):
                score += 5.0
        
        # 动物检测
        elif img_type == 'animal':
            if any(word in prompt_lower for word in self.animal_keywords):
                score += 4.0
        
        # 交通工具检测
        elif img_type == 'transportation':
            if any(word in prompt_lower for word in self.transportation_keywords):
                score += 4.0
        
        # 食物检测
        elif img_type == 'food':
            if any(word in prompt_lower for word in self.food_keywords):
                score += 4.0
        
        # 植物检测
        elif img_type == 'plant':
            if any(word in prompt_lower for word in self.plant_keywords):
                score += 4.0
        
        # 建筑检测
        elif img_type == 'architecture':
            if any(word in prompt_lower for word in self.architecture_keywords):
                score += 3.0
        
        # 室内检测
        elif img_type == 'indoor':
            if any(word in prompt_lower for word in self.indoor_keywords):
                score += 3.0
        
        # 风景检测
        elif img_type == 'scenery':
            if any(word in prompt_lower for word in self.scenery_keywords):
                score += 3.0
        
        # 抽象检测
        elif img_type == 'abstract':
            if any(word in prompt_lower for word in self.abstract_keywords):
                score += 3.0
        
        return score
    
    def predict_image_type(self, prompt: str) -> str:
        """预测图像类型"""
        # 计算所有类型的分数
        type_scores = {}
        all_types = ['abstract', 'architecture', 'indoor', 'scenery', 'single-human', 
                    'multiple-human', 'animal', 'transportation', 'food', 'plant', 'other']
        
        for img_type in all_types:
            type_scores[img_type] = self.calculate_type_score(prompt, img_type)
        
        # 返回得分最高的类型
        best_type = max(type_scores, key=type_scores.get)
        
        # 如果最高分数太低，返回other
        if type_scores[best_type] < 1.0:
            return 'other'
        
        return best_type
    
    def predict_dimensions(self, prompt: str, image_type: str) -> List[str]:
        """预测维度"""
        prompt_lower = prompt.lower()
        
        # 检查是否包含camera motion
        is_camera_motion = any(camera_keyword in prompt_lower for camera_keyword in self.camera_motion_keywords)
        
        if is_camera_motion:
            return self.camera_motion_dimension
        else:
            # 根据图像类型分配基础维度
            if image_type in self.background_types:
                return self.base_background_dimensions
            elif image_type in self.subject_types:
                return self.base_subject_dimensions
            else:
                # 默认使用background维度
                return self.base_background_dimensions
    
    def predict(self, prompt: str) -> Dict[str, any]:
        """预测prompt对应的image_type和dimensions"""
        # 预测图像类型
        image_type = self.predict_image_type(prompt)
        
        # 预测维度
        dimensions = self.predict_dimensions(prompt, image_type)
        
        return {
            'prompt_en': prompt,
            'predicted_image_type': image_type,
            'predicted_dimensions': dimensions
        }
    
    def batch_predict(self, prompts: List[str]) -> List[Dict]:
        """批量预测"""
        results = []
        for prompt in prompts:
            result = self.predict(prompt)
            results.append(result)
        return results
    
    def evaluate_predictions(self, test_data: List[Dict]) -> Dict:
        """评估预测准确性"""
        correct_type = 0
        correct_dimensions = 0
        total = 0
        
        type_confusion = defaultdict(lambda: defaultdict(int))
        
        for item in test_data:
            prompt = item['prompt_en']
            true_type = item['image_type']
            true_dimensions = set(item['dimension'])
            
            prediction = self.predict(prompt)
            pred_type = prediction['predicted_image_type']
            pred_dimensions = set(prediction['predicted_dimensions'])
            
            # 统计准确性
            if pred_type == true_type:
                correct_type += 1
            
            if pred_dimensions == true_dimensions:
                correct_dimensions += 1
            
            # 混淆矩阵
            type_confusion[true_type][pred_type] += 1
            
            total += 1
        
        return {
            'type_accuracy': correct_type / total,
            'dimension_accuracy': correct_dimensions / total,
            'total_samples': total,
            'type_confusion_matrix': dict(type_confusion)
        }

def test_improved_predictor():
    """Test improved predictor"""
    predictor = ImprovedVBenchPredictor()
    
    # Test samples
    test_prompts = [
        "a close up of a blue and orange liquid",
        "a castle on top of a hill covered in snow", 
        "a man walking in the park",
        "a group of people at a party",
        "a cat sitting on a chair",
        "a car driving down the street",
        "a beautiful flower in the garden",
        "a delicious cake on the table",
        "a living room with modern furniture",
        "a mountain landscape at sunset",
        "a close up of a blue and orange liquid, camera pans left"
    ]
    
    print("=== Improved Predictor Test ===")
    for prompt in test_prompts:
        result = predictor.predict(prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Predicted Type: {result['predicted_image_type']}")
        print(f"Predicted Dimensions: {result['predicted_dimensions']}")

def compare_predictors():
    """Compare original and improved predictors"""
    try:
        with open('vbench2_beta_i2v/vbench2_i2v_full_info.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Only evaluate base prompts
        base_data = [item for item in data if 'camera' not in item['prompt_en']]
        
        # Improved predictor
        improved_predictor = ImprovedVBenchPredictor()
        improved_eval = improved_predictor.evaluate_predictions(base_data)
        
        print(f"\n=== Predictor Performance Comparison ===")
        print(f"Evaluation samples: {len(base_data)}")
        print(f"Improved image type prediction accuracy: {improved_eval['type_accuracy']:.3f}")
        print(f"Improved dimension prediction accuracy: {improved_eval['dimension_accuracy']:.3f}")
        
    except FileNotFoundError:
        print("Cannot find data file, skipping performance comparison")

def main():
    """Main function"""
    # Test improved predictor
    test_improved_predictor()
    
    # Compare predictor performance
    compare_predictors()

if __name__ == '__main__':
    main() 