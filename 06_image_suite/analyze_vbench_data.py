#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze the relationship between dimension and image_type in VBench data and prompt_en
"""

import json
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set
import pandas as pd

def load_vbench_data(file_path: str) -> List[Dict]:
    """Load VBench data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_data_structure(data: List[Dict]) -> None:
    """Analyze data structure"""
    print("=== Data Structure Analysis ===")
    print(f"Total data count: {len(data)}")
    
    # Count image_type distribution
    image_types = Counter([item['image_type'] for item in data])
    print(f"\nimage_type distribution:")
    for img_type, count in image_types.most_common():
        print(f"  {img_type}: {count}")
    
    # Count dimension distribution
    all_dimensions = []
    for item in data:
        all_dimensions.extend(item['dimension'])
    dimension_counts = Counter(all_dimensions)
    print(f"\ndimension distribution:")
    for dim, count in dimension_counts.most_common():
        print(f"  {dim}: {count}")
    
    # Count prompt patterns
    camera_motion_prompts = [item for item in data if 'camera' in item['prompt_en']]
    base_prompts = [item for item in data if 'camera' not in item['prompt_en']]
    print(f"\nPrompts with camera motion: {len(camera_motion_prompts)}")
    print(f"Base prompts: {len(base_prompts)}")

def extract_prompt_patterns(data: List[Dict]) -> Dict:
    """Extract prompt patterns"""
    print("\n=== Prompt Pattern Analysis ===")
    
    # Analyze camera motion patterns
    camera_patterns = {
        'pans left': [],
        'pans right': [],
        'tilts up': [],
        'tilts down': [],
        'zooms in': [],
        'zooms out': [],
        'static': []
    }
    
    base_prompts = {}  # base prompt -> corresponding dimension and image_type
    
    for item in data:
        prompt = item['prompt_en']
        
        if 'camera' in prompt:
            # Extract camera motion type
            for pattern in camera_patterns.keys():
                if pattern in prompt:
                    camera_patterns[pattern].append(item)
                    break
        else:
            # Base prompt
            base_prompts[prompt] = {
                'dimension': item['dimension'],
                'image_type': item['image_type'],
                'image_name': item['image_name']
            }
    
    print(f"Base prompt count: {len(base_prompts)}")
    for pattern, items in camera_patterns.items():
        print(f"Camera {pattern}: {len(items)}")
    
    return {
        'base_prompts': base_prompts,
        'camera_patterns': camera_patterns
    }

def analyze_dimension_rules(data: List[Dict]) -> Dict:
    """Analyze dimension assignment rules"""
    print("\n=== Dimension Assignment Rules Analysis ===")
    
    # Base prompt dimensions
    base_dimensions = set()
    camera_dimensions = set()
    
    for item in data:
        if 'camera' in item['prompt_en']:
            camera_dimensions.update(item['dimension'])
        else:
            base_dimensions.update(item['dimension'])
    
    print("Base prompt dimensions:")
    for dim in sorted(base_dimensions):
        print(f"  {dim}")
    
    print("\nCamera motion prompt dimensions:")
    for dim in sorted(camera_dimensions):
        print(f"  {dim}")
    
    # Analyze image_type and dimension relationship
    type_dimension_map = defaultdict(set)
    for item in data:
        if 'camera' not in item['prompt_en']:  # Only analyze base prompts
            type_dimension_map[item['image_type']].update(item['dimension'])
    
    print("\nimage_type and dimension relationship:")
    for img_type, dimensions in type_dimension_map.items():
        print(f"  {img_type}: {sorted(dimensions)}")
    
    return {
        'base_dimensions': base_dimensions,
        'camera_dimensions': camera_dimensions,
        'type_dimension_map': dict(type_dimension_map)
    }

def analyze_image_type_keywords(data: List[Dict]) -> Dict:
    """Analyze image_type and prompt keyword relationship"""
    print("\n=== Image Type Keyword Analysis ===")
    
    type_keywords = defaultdict(set)
    
    for item in data:
        if 'camera' not in item['prompt_en']:  # Only analyze base prompts
            prompt = item['prompt_en'].lower()
            words = re.findall(r'\b\w+\b', prompt)
            type_keywords[item['image_type']].update(words)
    
    # Find characteristic words for each type
    type_specific_keywords = {}
    all_words = set()
    for words in type_keywords.values():
        all_words.update(words)
    
    for img_type, words in type_keywords.items():
        # Calculate word frequency
        type_prompts = [item['prompt_en'].lower() for item in data 
                       if item['image_type'] == img_type and 'camera' not in item['prompt_en']]
        
        word_counts = Counter()
        for prompt in type_prompts:
            prompt_words = re.findall(r'\b\w+\b', prompt)
            word_counts.update(prompt_words)
        
        # Take top 10 high-frequency words
        top_words = [word for word, count in word_counts.most_common(10)]
        type_specific_keywords[img_type] = top_words
        
        print(f"{img_type} type high-frequency keywords: {top_words}")
    
    return type_specific_keywords

def main():
    """Main function"""
    # Load data
    data = load_vbench_data('vbench2_beta_i2v/vbench2_i2v_full_info.json')
    
    # Analyze data structure
    analyze_data_structure(data)
    
    # Extract prompt patterns
    patterns = extract_prompt_patterns(data)
    
    # Analyze dimension rules
    dimension_rules = analyze_dimension_rules(data)
    
    # Analyze image_type keywords
    type_keywords = analyze_image_type_keywords(data)
    
    # Save analysis results
    analysis_result = {
        'patterns': patterns,
        'dimension_rules': dimension_rules,
        'type_keywords': type_keywords
    }
    
    # Save key information separately due to non-serializable objects like set
    summary = {
        'base_dimensions': list(dimension_rules['base_dimensions']),
        'camera_dimensions': list(dimension_rules['camera_dimensions']),
        'type_dimension_map': {k: list(v) for k, v in dimension_rules['type_dimension_map'].items()},
        'type_keywords': type_keywords
    }
    
    with open('vbench_analysis_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nAnalysis results saved to: vbench_analysis_summary.json")

if __name__ == '__main__':
    main() 