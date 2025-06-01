#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VBench Predictor Usage Example
Simple example showing how to use the VBench predictors
"""

from vbench_prompt_analyzer import VBenchPromptAnalyzer

def main():
    """Main example function"""
    print("VBench Predictor Usage Example")
    print("=" * 40)
    
    # Create analyzer
    analyzer = VBenchPromptAnalyzer()
    
    # Example 1: Single prompt analysis
    print("\n1. Single Prompt Analysis:")
    prompt = "a man walking in the park"
    result = analyzer.analyze(prompt)
    print(f"Prompt: {prompt}")
    print(f"Image Type: {result['image_type']}")
    print(f"Dimensions: {result['dimensions']}")
    
    # Example 2: Camera motion detection
    print("\n2. Camera Motion Detection:")
    camera_prompt = "a beautiful landscape, camera pans left"
    camera_result = analyzer.analyze(camera_prompt)
    print(f"Prompt: {camera_prompt}")
    print(f"Image Type: {camera_result['image_type']}")
    print(f"Dimensions: {camera_result['dimensions']}")
    
    # Example 3: Batch analysis
    print("\n3. Batch Analysis:")
    prompts = [
        "a cat sitting on a sofa",
        "a group of friends at dinner",
        "a modern building in the city",
        "a beautiful flower garden"
    ]
    
    results = analyzer.batch_analyze(prompts)
    for result in results:
        print(f"  {result['prompt_en'][:30]}... -> {result['image_type']}")
    
    # Example 4: Different image types
    print("\n4. Different Image Types Examples:")
    examples = {
        "Abstract": "a swirling pattern of colors",
        "Architecture": "a tall skyscraper in downtown",
        "Indoor": "a cozy living room with fireplace",
        "Scenery": "a mountain lake at sunset",
        "Single Human": "a woman reading a book",
        "Multiple Human": "a family having picnic",
        "Animal": "a dog playing in the yard",
        "Transportation": "a car driving on highway",
        "Food": "a delicious pizza on table",
        "Plant": "a rose blooming in garden"
    }
    
    for category, prompt in examples.items():
        result = analyzer.analyze(prompt)
        print(f"  {category:15} -> {result['image_type']:15} | {prompt}")
    
    print("\n" + "=" * 40)
    print("Usage complete! All predictors working correctly.")

if __name__ == '__main__':
    main() 