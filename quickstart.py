#!/usr/bin/env python
"""
Quick Start Script for Waste Detection System
Run this script to set up and verify your environment
"""
import os
import sys
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor} (Compatible)")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor} (Requires 3.8+)")
        return False


def check_dependencies():
    """Check if all required packages are installed"""
    print("\n🔍 Checking dependencies...")
    
    required_packages = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'ultralytics': 'Ultralytics YOLOv8',
        'gradio': 'Gradio',
        'numpy': 'NumPy'
    }
    
    missing_packages = []
    
    for package_name, display_name in required_packages.items():
        try:
            __import__(package_name)
            print(f"✓ {display_name}")
        except ImportError:
            print(f"✗ {display_name}")
            missing_packages.append(package_name)
    
    return len(missing_packages) == 0, missing_packages


def create_directories():
    """Create necessary project directories"""
    print("\n📁 Creating project directories...")
    
    directories = [
        'models',
        'test_results',
        'inference_results',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ {directory}/")


def verify_dataset():
    """Verify dataset structure and configuration"""
    print("\n📊 Verifying dataset...")
    
    # Check data.yaml
    if not os.path.exists('data.yaml'):
        print("✗ data.yaml not found")
        return False
    print("✓ data.yaml found")
    
    # Check train directory
    if not os.path.exists('train/images'):
        print("✗ train/images/ not found")
        return False
    print(f"✓ train/images/ ({len(os.listdir('train/images'))} images)")
    
    # Check valid directory
    if not os.path.exists('valid/images'):
        print("✗ valid/images/ not found")
        return False
    print(f"✓ valid/images/ ({len(os.listdir('valid/images'))} images)")
    
    # Check test directory
    if not os.path.exists('test/images'):
        print("✗ test/images/ not found")
        return False
    print(f"✓ test/images/ ({len(os.listdir('test/images'))} images)")
    
    return True


def check_gpu():
    """Check if GPU is available"""
    print("\n🖥️  Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("⚠ No GPU detected - CPU will be used (training will be slower)")
            return False
    except Exception as e:
        print(f"⚠ Could not check GPU: {e}")
        return False


def print_next_steps():
    """Print recommended next steps"""
    print("\n" + "="*60)
    print("🎉 Setup Complete!")
    print("="*60)
    
    print("\n📝 Next Steps:\n")
    
    print("1️⃣  TRAIN THE MODEL (This is required first!)")
    print("   Command:")
    print("   $ python train.py --data data.yaml --epochs 100")
    print()
    
    print("2️⃣  RUN THE WEB INTERFACE")
    print("   Command:")
    print("   $ python app.py")
    print("   Then open: http://localhost:7860")
    print()
    
    print("3️⃣  TEST ON FOLDER")
    print("   Command:")
    print("   $ python -c \"from utils import test_model_on_folder;\"")
    print("            test_model_on_folder('models/waste_detector/weights/best.pt', 'test/images')\"")
    print()
    
    print("📚 For more details, see README.md")
    print("="*60)


def main():
    """Run all checks"""
    print("\n" + "="*60)
    print("🗑️  WASTE DETECTION SYSTEM - SETUP VERIFICATION")
    print("="*60 + "\n")
    
    checks_passed = 0
    total_checks = 5
    
    # Check Python version
    if check_python_version():
        checks_passed += 1
    
    # Check dependencies
    deps_ok, missing = check_dependencies()
    if deps_ok:
        checks_passed += 1
    else:
        print("\n⚠️  Missing packages! Install with:")
        print(f"   pip install {' '.join(missing)}")
    
    # Create directories
    try:
        create_directories()
        checks_passed += 1
    except Exception as e:
        print(f"✗ Error creating directories: {e}")
    
    # Verify dataset
    if verify_dataset():
        checks_passed += 1
    else:
        print("\n⚠️  Dataset issues detected. Please verify your data structure.")
    
    # Check GPU
    check_gpu()
    checks_passed += 1
    
    # Summary
    print("\n" + "="*60)
    if checks_passed >= 4:
        print(f"✓ Setup verification passed! ({checks_passed}/{total_checks} checks)")
        print_next_steps()
    else:
        print(f"⚠️  Some checks failed ({checks_passed}/{total_checks})")
        print("Please fix the issues above and try again.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
