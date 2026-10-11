"""
Utility functions for Waste Detection System
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import os


def test_model_on_folder(model_path: str, image_folder: str, output_folder: str = "test_results"):
    """
    Test model on all images in a folder and save annotated results.
    
    Args:
        model_path: Path to trained model
        image_folder: Folder containing test images
        output_folder: Folder to save results
    """
    Path(output_folder).mkdir(exist_ok=True)
    
    model = YOLO(model_path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    image_files = [
        f for f in os.listdir(image_folder)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]
    
    print(f"Found {len(image_files)} images to process")
    
    for i, img_file in enumerate(image_files, 1):
        img_path = os.path.join(image_folder, img_file)
        
        # Run detection
        results = model(img_path, verbose=False)
        
        # Save annotated image
        annotated = results[0].plot()
        output_path = os.path.join(output_folder, f"detected_{img_file}")
        cv2.imwrite(output_path, annotated)
        
        print(f"[{i}/{len(image_files)}] Processed: {img_file}")
    
    print(f"\nResults saved to: {output_folder}")


def estimate_model_size(model_name: str) -> dict:
    """
    Get information about YOLOv8 model sizes.
    
    Args:
        model_name: Model size (nano, small, medium, large, xlarge)
        
    Returns:
        Dictionary with model info
    """
    model_info = {
        'nano': {'params': 3.2, 'size_mb': 11.2, 'speed': 'Fastest'},
        'small': {'params': 11.2, 'size_mb': 40.6, 'speed': 'Fast'},
        'medium': {'params': 25.9, 'size_mb': 100.7, 'speed': 'Balanced'},
        'large': {'params': 43.7, 'size_mb': 165.2, 'speed': 'Slower'},
        'xlarge': {'params': 68.2, 'size_mb': 257.7, 'speed': 'Slowest'},
    }
    return model_info.get(model_name, {})


def get_model_recommendations():
    """Print model selection recommendations"""
    recommendations = {
        'Real-time inference': 'yolov8n or yolov8s',
        'Balanced (recommended)': 'yolov8m',
        'High accuracy': 'yolov8l or yolov8x',
        'Edge devices': 'yolov8n',
        'Cloud/GPU': 'yolov8x'
    }
    print("\n=== Model Selection Guide ===")
    for use_case, model in recommendations.items():
        print(f"{use_case}: {model}")


def setup_directories():
    """Create necessary project directories"""
    directories = [
        'models',
        'test_results',
        'inference_results',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")


def validate_dataset(data_yaml: str):
    """
    Validate dataset configuration and paths.
    
    Args:
        data_yaml: Path to data.yaml file
        
    Returns:
        Boolean indicating if dataset is valid
    """
    import yaml
    
    if not os.path.exists(data_yaml):
        print(f"Error: {data_yaml} not found")
        return False
    
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print("\n=== Dataset Validation ===")
    print(f"Classes: {data_config.get('names', [])}")
    print(f"Number of classes: {data_config.get('nc', 0)}")
    
    # Check paths
    for split in ['train', 'val', 'test']:
        path = data_config.get(split)
        if path:
            full_path = path if os.path.isabs(path) else os.path.join(
                os.path.dirname(data_yaml), path
            )
            exists = os.path.exists(full_path)
            status = "✓" if exists else "✗"
            print(f"{status} {split}: {full_path}")
    
    return True


if __name__ == "__main__":
    print("Waste Detection System - Utility Functions")
    get_model_recommendations()
    setup_directories()
