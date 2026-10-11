@echo off
REM Waste Detection System - Training Script
REM This script auto-detects available hardware and runs training

echo.
echo =====================================
echo 🗑️  Waste Detection System
echo Training Initialization Script
echo =====================================
echo.

REM Check Python availability
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo ✓ Python is available
echo.

REM Run training with CPU (since GPU is not available)
echo Starting training on CPU...
echo This will take 30-60 minutes depending on your processor
echo.

python train.py --data data.yaml --model yolov8n.pt --epochs 5 --batch 8 --device -1

if errorlevel 1 (
    echo.
    echo ERROR: Training failed
    pause
    exit /b 1
)

echo.
echo =====================================
echo ✓ Training completed successfully!
echo Best model saved to: models/waste_detector/weights/best.pt
echo.
echo Next steps:
echo 1. Run: python app.py
echo 2. Open: http://localhost:7860
echo =====================================
echo.

pause
