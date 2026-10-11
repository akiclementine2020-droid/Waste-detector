# Waste Detection System - Training Script
# This script auto-detects available hardware and runs training on CPU

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "🗑️  Waste Detection System" -ForegroundColor Yellow
Write-Host "Training Initialization Script" -ForegroundColor Yellow
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Check Python availability
try {
    python --version | Out-Null
    Write-Host "✓ Python is available" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Exit 1
}

Write-Host ""
Write-Host "Starting training on CPU..." -ForegroundColor Yellow
Write-Host "This will take 30-60 minutes depending on your processor" -ForegroundColor Yellow
Write-Host ""

# Run training with CPU
python train.py --data data.yaml --model yolov8n.pt --epochs 5 --batch 8 --device -1

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Training failed" -ForegroundColor Red
    Exit 1
}

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "✓ Training completed successfully!" -ForegroundColor Green
Write-Host "Best model saved to: models/waste_detector/weights/best.pt" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Run: python app.py" -ForegroundColor White
Write-Host "2. Open: http://localhost:7860" -ForegroundColor White
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
