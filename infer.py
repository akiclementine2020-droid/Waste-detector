#!/usr/bin/env python
"""
Simple inference script for waste detection
Use this script to run inference on images without the web interface
"""
import os
import sys
from pathlib import Path
from PIL import Image
import argparse
from app import WasteDetectionSystem
import json
from datetime import datetime


def run_inference_on_image(
    image_path: str,
    model_path: str = "models/waste_detector/weights/best.pt",
    confidence: float = 0.5,
    output_dir: str = "inference_results",
    save_annotated: bool = True,
    save_report: bool = True
):
    """
    Run inference on a single image.
    
    Args:
        image_path: Path to input image
        model_path: Path to trained model
        confidence: Confidence threshold
        output_dir: Directory to save results
        save_annotated: Save annotated image
        save_report: Save detection report
    """
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using: python train.py")
        return None
    
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    print(f"Image size: {image.size}")
    
    print(f"Loading model: {model_path}")
    detector = WasteDetectionSystem(model_path)
    
    if detector.model is None:
        print("Error: Could not load model")
        return None
    
    print(f"Running inference with confidence threshold: {confidence}")
    annotated_image, detections = detector.detect(image, confidence=confidence)
    
    # Generate output filename
    image_name = Path(image_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save annotated image
    if save_annotated:
        output_image_path = os.path.join(
            output_dir,
            f"{image_name}_detected_{timestamp}.jpg"
        )
        annotated_image.save(output_image_path)
        print(f"\n✓ Annotated image saved: {output_image_path}")
    
    # Save report
    if save_report:
        output_report_path = os.path.join(
            output_dir,
            f"{image_name}_report_{timestamp}.json"
        )
        with open(output_report_path, 'w') as f:
            json.dump(detections, f, indent=2)
        print(f"✓ Report saved: {output_report_path}")
    
    # Print results
    print("\n" + "="*60)
    print("📊 DETECTION RESULTS")
    print("="*60)
    print(f"Total objects detected: {detections['total_objects']}")
    
    if detections['total_objects'] > 0:
        print("\nBy class:")
        for class_name, count in sorted(detections['by_class'].items()):
            if count > 0:
                percentage = (count / detections['total_objects']) * 100
                print(f"  - {class_name.capitalize()}: {count} ({percentage:.1f}%)")
        
        print("\nDetailed detections:")
        for i, det in enumerate(detections['details'], 1):
            print(f"  {i}. {det['class'].capitalize()}: {det['confidence']}")
    else:
        print("No waste objects detected in this image.")
    
    print("="*60 + "\n")
    
    return {
        'image_path': image_path,
        'detections': detections,
        'confidence': confidence,
        'output_image': output_image_path if save_annotated else None,
        'output_report': output_report_path if save_report else None
    }


def run_inference_on_folder(
    folder_path: str,
    model_path: str = "models/waste_detector/weights/best.pt",
    confidence: float = 0.5,
    output_dir: str = "inference_results"
):
    """
    Run inference on all images in a folder.
    
    Args:
        folder_path: Path to folder containing images
        model_path: Path to trained model
        confidence: Confidence threshold
        output_dir: Directory to save results
    """
    
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    image_files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} images to process")
    print("="*60 + "\n")
    
    results = []
    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(folder_path, image_file)
        print(f"[{i}/{len(image_files)}] Processing: {image_file}")
        
        result = run_inference_on_image(
            image_path=image_path,
            model_path=model_path,
            confidence=confidence,
            output_dir=output_dir,
            save_annotated=True,
            save_report=True
        )
        
        if result:
            results.append(result)
        
        print()
    
    # Summary report
    print("="*60)
    print("📈 BATCH PROCESSING SUMMARY")
    print("="*60)
    print(f"Processed: {len(results)}/{len(image_files)} images")
    
    if results:
        total_detections = sum(r['detections']['total_objects'] for r in results)
        print(f"Total objects detected: {total_detections}")
        
        # Aggregate class counts
        class_totals = {}
        for result in results:
            for cls, count in result['detections']['by_class'].items():
                class_totals[cls] = class_totals.get(cls, 0) + count
        
        if class_totals:
            print("\nAggregate by class:")
            for cls, count in sorted(class_totals.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    print(f"  - {cls.capitalize()}: {count}")
    
    print(f"\nResults saved to: {output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Waste Detection System - Inference Script"
    )
    
    parser.add_argument(
        "input",
        help="Path to image file or folder containing images"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="models/waste_detector/weights/best.pt",
        help="Path to trained model"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold (0.0-1.0)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="inference_results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--no-image",
        action="store_true",
        help="Don't save annotated images"
    )
    
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Don't save JSON reports"
    )
    
    args = parser.parse_args()
    
    # Check if input is a file or folder
    if os.path.isfile(args.input):
        print("🗑️  Waste Detection System - Single Image Inference\n")
        run_inference_on_image(
            image_path=args.input,
            model_path=args.model,
            confidence=args.confidence,
            output_dir=args.output,
            save_annotated=not args.no_image,
            save_report=not args.no_report
        )
    elif os.path.isdir(args.input):
        print("🗑️  Waste Detection System - Batch Inference\n")
        run_inference_on_folder(
            folder_path=args.input,
            model_path=args.model,
            confidence=args.confidence,
            output_dir=args.output
        )
    else:
        print(f"Error: {args.input} is neither a file nor a directory")
        sys.exit(1)
