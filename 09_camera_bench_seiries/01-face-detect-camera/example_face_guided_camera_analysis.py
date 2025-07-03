#!/usr/bin/env python3
"""
Face-Guided Camera Motion Analysis Example

This script demonstrates how to use the integrated DBFace face detection 
with camera motion analysis to get better zoom in/out detection.

Requirements:
- Make sure DBFace model weights are available at: vbench2/third_party/DBFace-master/model/dbface.pth
- Video file for analysis
- CUDA-capable GPU (recommended)

Usage:
    python example_face_guided_camera_analysis.py --video path/to/video.mp4
"""

import argparse
import sys
import os

# Add VBench path
sys.path.append(os.path.join(os.path.dirname(__file__), 'vbench2'))

from vbench2.camera_motion import analyze_single_video_with_face_guidance, demo_face_guided_camera_analysis
import torch

def main():
    parser = argparse.ArgumentParser(description='Face-guided camera motion analysis')
    parser.add_argument('--video', type=str, help='Path to video file for analysis')
    parser.add_argument('--demo', action='store_true', help='Run demo with example configuration')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], 
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # CoTracker model configuration
    submodules_dict = {"repo": "facebook/co-tracker", "model": "cotracker2"}
    
    if args.demo:
        print("Running demo...")
        demo_face_guided_camera_analysis()
        return
    
    if not args.video:
        print("Please provide a video path with --video or use --demo")
        return
    
    if not os.path.exists(args.video):
        print(f"Video file not found: {args.video}")
        return
    
    print(f"Analyzing video: {args.video}")
    print("This may take a few moments...")
    
    try:
        # Perform comprehensive analysis
        results = analyze_single_video_with_face_guidance(args.video, device, submodules_dict)
        
        if "error" in results:
            print(f"Analysis failed: {results['error']}")
            return
        
        print("\n" + "="*60)
        print("FACE-GUIDED CAMERA MOTION ANALYSIS RESULTS")
        print("="*60)
        
        print(f"\n📹 Video: {os.path.basename(results.get('video_path', 'Unknown'))}")
        
        # Traditional analysis results
        traditional = results.get('traditional_motion_analysis', [])
        print(f"\n🎯 Traditional Motion Analysis: {', '.join(traditional) if traditional else 'None detected'}")
        
        # Face analysis results
        if results.get('face_analysis') and results['face_analysis'].get('face_motion'):
            face_info = results['face_analysis']
            print(f"\n👤 Face-Based Analysis:")
            print(f"   Motion Type: {face_info.get('face_motion', 'None')}")
            print(f"   Confidence: {face_info.get('confidence', 0):.2f}")
            print(f"   Face Area Change: {face_info.get('ratio_change_percent', 0):+.1f}%")
            print(f"   Faces Detected: {face_info.get('start_faces_count', 0)} → {face_info.get('end_faces_count', 0)}")
        else:
            print(f"\n👤 Face-Based Analysis: No significant face-based motion detected")
        
        # Final recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   {results['user_guidance']['message']}")
        
        for i, suggestion in enumerate(results['user_guidance']['suggestions'], 1):
            print(f"   {i}. {suggestion}")
        
        # Detailed breakdown
        if results.get('final_recommendations'):
            print(f"\n📊 Detailed Analysis:")
            for rec in results['final_recommendations']:
                confidence_emoji = "🟢" if rec['confidence'] == "High" else "🟡" if rec['confidence'] == "Medium" else "🔴"
                print(f"   {confidence_emoji} {rec['motion_type'].upper()}")
                print(f"      Source: {rec['source']}")
                print(f"      Reason: {rec['description']}")
                print()
        
        # Usage suggestions
        print("💭 USAGE SUGGESTIONS:")
        final_recs = results.get('final_recommendations', [])
        if any(rec['source'] == 'Face Analysis' for rec in final_recs):
            print("   • Face detection provided enhanced zoom analysis")
            print("   • Consider the face-based recommendations for zoom motions")
        
        if len(final_recs) > 1:
            print("   • Multiple motions detected - video may have complex camera movement")
            print("   • Consider segmenting the video for more precise analysis")
        
        print(f"\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 