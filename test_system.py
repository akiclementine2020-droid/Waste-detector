#!/usr/bin/env python
"""
Test script to verify the Waste Detection System is working correctly
"""
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np


def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV: {e}")
        return False
    
    try:
        import gradio
        print(f"✓ Gradio {gradio.__version__}")
    except ImportError as e:
        print(f"✗ Gradio: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print(f"✓ Ultralytics YOLOv8")
    except ImportError as e:
        print(f"✗ Ultralytics: {e}")
        return False
    
    return True


def test_dataset():
    """Test if dataset is properly structured"""
    print("\n🧪 Testing dataset structure...")
    
    required_paths = [
        ('data.yaml', 'File'),
        ('train/images', 'Directory'),
        ('train/labels', 'Directory'),
        ('valid/images', 'Directory'),
        ('valid/labels', 'Directory'),
        ('test/images', 'Directory'),
        ('test/labels', 'Directory'),
    ]
    
    all_exist = True
    for path, path_type in required_paths:
        if os.path.exists(path):
            if path_type == 'Directory':
                count = len(os.listdir(path))
                print(f"✓ {path} ({count} items)")
            else:
                size = os.path.getsize(path) / 1024  # KB
                print(f"✓ {path} ({size:.1f} KB)")
        else:
            print(f"✗ {path} - NOT FOUND")
            all_exist = False
    
    return all_exist


def test_device():
    """Test GPU/CPU availability"""
    print("\n🧪 Testing device availability...")
    
    import torch
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU available: {device_name}")
        return True
    else:
        print("⚠️  GPU not available (using CPU - training will be slower)")
        return False


def test_model_loading():
    """Test if we can load a model (even if untrained)"""
    print("\n🧪 Testing model loading...")
    
    from ultralytics import YOLO
    
    try:
        # Try to load pretrained model
        model = YOLO('yolov8n.pt')
        print("✓ Successfully loaded YOLOv8 model")
        return True
    except Exception as e:
        print(f"✗ Could not load model: {e}")
        return False


def test_inference_on_dummy():
    """Test inference on a dummy image"""
    print("\n🧪 Testing inference on dummy image...")
    
    try:
        from ultralytics import YOLO
        import numpy as np
        
        # Load model
        model = YOLO('yolov8n.pt')
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Run inference
        results = model(dummy_image, verbose=False)
        
        print(f"✓ Inference successful")
        print(f"  - Image size: {dummy_image.shape}")
        print(f"  - Detections: {len(results[0].boxes) if results[0].boxes is not None else 0}")
        return True
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        return False


def test_gradio_interface():
    """Test if Gradio interface can be created"""
    print("\n🧪 Testing Gradio interface creation...")
    
    try:
        import gradio as gr
        
        # Simple test interface
        def echo(text):
            return text
        
        demo = gr.Interface(fn=echo, inputs="text", outputs="text")
        print("✓ Gradio interface created successfully")
        return True
    except Exception as e:
        print(f"✗ Gradio interface creation failed: {e}")
        return False


def test_trained_model():
    """Test if trained model exists and can be loaded"""
    print("\n🧪 Testing trained model...")
    
    model_path = "models/waste_detector/weights/best.pt"
    
    if os.path.exists(model_path):
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            print(f"✓ Trained model found and loaded")
            print(f"  - Path: {model_path}")
            return True
        except Exception as e:
            print(f"✗ Could not load trained model: {e}")
            return False
    else:
        print(f"⚠️  Trained model not found at {model_path}")
        print(f"   Run 'python train.py' to train the model first")
        return False


def print_recommendations():
    """Print recommendations based on test results"""
    print("\n" + "="*60)
    print("📋 RECOMMENDATIONS")
    print("="*60)
    
    print("\n1. If all tests passed:")
    print("   → Run: python train.py")
    print("   → Then: python app.py")
    
    print("\n2. If some tests failed:")
    print("   → Install missing dependencies: pip install -r requirements.txt")
    print("   → Check dataset paths are correct")
    print("   → Verify GPU drivers (if using GPU)")
    
    print("\n3. For training:")
    print("   → GPU recommended for faster training")
    print("   → Start with: python train.py --data data.yaml --epochs 50")
    print("   → Monitor with: tensorboard --logdir models")
    
    print("\n4. For inference:")
    print("   → After training: python app.py")
    print("   → Or use: python -c 'from app import WasteDetectionSystem'")
    
    print("\n" + "="*60)


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🗑️  WASTE DETECTION SYSTEM - TEST SUITE")
    print("="*60 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Dataset", test_dataset),
        ("Device", test_device),
        ("Model Loading", test_model_loading),
        ("Dummy Inference", test_inference_on_dummy),
        ("Gradio Interface", test_gradio_interface),
        ("Trained Model", test_trained_model),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} test error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓" if result else "✗"
        print(f"{status} {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("✅ All tests passed! System is ready to use.\n")
    elif passed >= total - 1:
        print("⚠️  Most tests passed. Check the trained model warning above.\n")
    else:
        print("⚠️  Some tests failed. Please fix the issues and try again.\n")
    
    print_recommendations()


if __name__ == "__main__":
    main()
