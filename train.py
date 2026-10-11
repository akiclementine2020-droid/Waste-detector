"""
Training script for YOLOv8 Waste Detection Model
"""
import os
from ultralytics import YOLO
import argparse
from pathlib import Path


def train_waste_detector(
    data_yaml: str = "data.yaml",
    model_name: str = "yolov8n.pt",
    epochs: int = 55,
    imgsz: int = 416,
    batch_size: int = 8,
    device: int = -1,
    patience: int = 20,
    output_dir: str = "models"
):
    """
    Train YOLOv8 model for waste detection.
    
    Args:
        data_yaml: Path to dataset configuration file
        model_name: YOLOv8 model size (nano, small, medium, large, xlarge)
        epochs: Number of training epochs
        imgsz: Input image size
        batch_size: Batch size for training
        device: GPU device ID (0 for first GPU, -1 for CPU)
        patience: Early stopping patience
        output_dir: Directory to save trained models
    """
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"Loading YOLOv8 model: {model_name}")
    model = YOLO(model_name)
    
    print(f"Starting training with dataset: {data_yaml}")
    print(f"Configuration:")
    print(f"  - Model: {model_name}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Image size: {imgsz}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Device: {'GPU' if device >= 0 else 'CPU'}")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        patience=patience,
        save=True,
        project=output_dir,
        name="waste_detector",
        exist_ok=True,
        verbose=True,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
    )
    
    print("\nTraining completed!")
    print(f"Best model saved to: {output_dir}/waste_detector/weights/best.pt")
    
    return results


def validate_model(
    model_path: str,
    data_yaml: str = "data.yaml",
    imgsz: int = 416,
    batch_size: int = 8,
    device: int = 0
):
    """
    Validate trained model on validation set.
    
    Args:
        model_path: Path to trained model
        data_yaml: Path to dataset configuration file
        imgsz: Input image size
        batch_size: Batch size for validation
        device: GPU device ID
    """
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print("Running validation...")
    results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        verbose=True
    )
    
    print("\nValidation Results:")
    print(f"  - mAP50: {results.box.map50:.4f}")
    print(f"  - mAP50-95: {results.box.map:.4f}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 Waste Detection Model"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data.yaml",
        help="Path to dataset configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="YOLOv8 model size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=55,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=416,
        help="Input image size"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="GPU device ID (0 for first GPU, -1 for CPU)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models",
        help="Output directory for trained models"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation on existing model"
    )
    parser.add_argument(
        "--validate-model",
        type=str,
        default=None,
        help="Path to model for validation"
    )
    
    args = parser.parse_args()
    
    if args.validate_only and args.validate_model:
        
        validate_model(
            model_path=args.validate_model,
            data_yaml=args.data,
            device=args.device
        )
    else:
        train_waste_detector(
            data_yaml=args.data,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            output_dir=args.output
        )
