# 🗑️ Waste Detection System

A comprehensive waste detection and classification system built with **YOLOv8** and **Gradio**, designed to support proper waste management and recycling through AI-powered computer vision.

## 🎯 Features

- **Multi-class Waste Detection**: Automatically detects and classifies 6 waste categories
  - Bottle
  - Cardboard
  - Metal
  - Organic
  - Paper
  - Plastic

- **Web-Based Interface**: User-friendly Gradio interface for easy image uploads and real-time detection
- **Bounding Box Visualization**: Precise object localization with confidence scores
- **Confidence Threshold Control**: Adjustable detection sensitivity
- **Detailed Statistics**: Comprehensive breakdown of detected waste items
- **Training Scripts**: Fully configurable training pipeline with YOLOv8

## 📋 Requirements

- Python 3.8+
- GPU (NVIDIA CUDA 11.8+) - recommended for faster training/inference
- 8GB RAM minimum (16GB recommended)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd detection

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Your dataset is already organized in the required format:
```
detection/
├── data.yaml              # Dataset configuration
├── train/                 # Training images and labels
│   ├── images/
│   └── labels/
├── valid/                 # Validation images and labels
│   ├── images/
│   └── labels/
└── test/                  # Test images and labels
    ├── images/
    └── labels/
```

Verify the dataset configuration:
```bash
python utils.py
```

### 3. Train the Model

**Basic training** (recommended for first-time users):
```bash
python train.py \
    --data data.yaml \
    --model yolov8m.pt \
    --epochs 100 \
    --batch 16
```

**Advanced training options**:
```bash
python train.py \
    --data data.yaml \
    --model yolov8l.pt \
    --epochs 150 \
    --batch 32 \
    --imgsz 800 \
    --device 0 \
    --output models
```

**Available model sizes**:
- `yolov8n.pt` - Nano (Fastest, least accurate)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium (Recommended)
- `yolov8l.pt` - Large
- `yolov8x.pt` - XLarge (Slowest, most accurate)

### 4. Validate Model

```bash
python train.py \
    --validate-only \
    --validate-model models/waste_detector/weights/best.pt \
    --data data.yaml
```

### 5. Run Web Interface

```bash
python app.py
```

The interface will start at `http://localhost:7860` and display a shareable link.

## 📊 Usage Guide

### Web Interface

1. **Upload Image**: Click on the image area or drag-and-drop an image
2. **Adjust Confidence**: Use the slider to set detection confidence threshold (0.0-1.0)
3. **View Results**: 
   - Annotated image with bounding boxes
   - Statistics showing waste composition
   - Detailed detection report

### Command-Line Inference

```python
from app import WasteDetectionSystem
from PIL import Image

detector = WasteDetectionSystem("models/waste_detector/weights/best.pt")
image = Image.open("path/to/image.jpg")
annotated_image, detections = detector.detect(image, confidence=0.5)
```

## 📁 Project Structure

```
detection/
├── data.yaml                 # Dataset configuration (6 waste classes)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── train.py                  # Training script with full configuration
├── app.py                    # Gradio web interface for inference
├── utils.py                  # Utility functions
├── models/                   # Trained models (created after training)
│   └── waste_detector/
│       └── weights/
│           ├── best.pt       # Best model checkpoint
│           └── last.pt       # Last epoch checkpoint
├── test_results/             # Test inference results (created on demand)
├── train/                    # Training dataset
│   ├── images/               # Training images
│   └── labels/               # YOLO format labels
├── valid/                    # Validation dataset
│   ├── images/               # Validation images
│   └── labels/               # YOLO format labels
└── test/                     # Test dataset
    ├── images/               # Test images
    └── labels/               # YOLO format labels
```

## 🎓 Training Details

### Data Augmentation
The training script applies advanced augmentation strategies:
- Horizontal and vertical flips (50% probability)
- Rotation (±10 degrees)
- HSV color space adjustments
- Scaling and translation
- Mosaic augmentation

### Optimization
- Optimizer: SGD with momentum
- Learning rate: 0.01 (initial), 0.01 (final)
- Momentum: 0.937
- Weight decay: 0.0005
- Early stopping: 20 epochs patience

### Expected Performance
Training with the recommended configuration on a good GPU:
- **Training time**: 2-6 hours (depends on model size)
- **Convergence**: Usually within 50-80 epochs
- **Expected mAP50**: 85-95% (depending on model size)

## 🔧 Configuration

### Modify Training Parameters

Edit the `train_waste_detector()` function in `train.py`:

```python
results = model.train(
    data=data_yaml,
    epochs=100,              # Increase for better accuracy
    imgsz=640,               # Larger = better accuracy, slower
    batch=16,                # Increase if GPU memory allows
    device=0,                # GPU device ID
    patience=20,             # Early stopping patience
    lr0=0.01,                # Initial learning rate
    momentum=0.937,          # Optimizer momentum
    weight_decay=0.0005,     # L2 regularization
)
```

### Inference Confidence Threshold

The web interface allows real-time adjustment. Lower values = more detections (higher false positives). Higher values = fewer detections (higher confidence).

## 🖥️ System Requirements

### Minimum
- CPU: Intel i5 / AMD Ryzen 5
- RAM: 8GB
- Storage: 5GB free space

### Recommended
- CPU: Intel i7 / AMD Ryzen 7
- GPU: NVIDIA RTX 2060 or better (with 6GB+ VRAM)
- RAM: 16GB
- Storage: 10GB free space (SSD recommended)

### For GPU Training
Install CUDA toolkit and cuDNN:
- CUDA 11.8+
- cuDNN 8.6+
- PyTorch with CUDA support (already in requirements.txt)

## 🧪 Testing and Evaluation

### Batch Inference

Process all images in a folder:

```python
from utils import test_model_on_folder

test_model_on_folder(
    model_path="models/waste_detector/weights/best.pt",
    image_folder="test/images",
    output_folder="test_results"
)
```

### Metrics

After training, check the results:
```
models/waste_detector/results/
├── confusion_matrix.png
├── results.csv
├── F1_curve.png
├── P_curve.png
├── R_curve.png
└── PR_curve.png
```

## 🐛 Troubleshooting

### Model not loading
```
Error: Model not found at models/waste_detector/weights/best.pt
```
**Solution**: Train the model first using `python train.py`

### CUDA/GPU issues
```
Error: CUDA out of memory
```
**Solutions**:
- Reduce batch size: `--batch 8`
- Reduce image size: `--imgsz 416`
- Use smaller model: `--model yolov8s.pt`
- Use CPU: `--device -1`

### Poor detection results
- Increase training epochs: `--epochs 150`
- Use larger model: `--model yolov8l.pt`
- Verify dataset labels are correct
- Check data augmentation settings

## 📈 Performance Tips

1. **GPU Memory**: Monitor with `nvidia-smi`
2. **Batch Processing**: Larger batches = faster training but more memory
3. **Model Selection**: Medium (yolov8m) balances speed and accuracy
4. **Early Stopping**: Prevents overfitting, saves training time

## 🌍 Use Cases

- **Waste Sorting Facilities**: Automate waste classification
- **Recycling Centers**: Improve sorting efficiency
- **Environmental Monitoring**: Track waste composition
- **Smart Bins**: Real-time waste identification
- **Educational Projects**: Learn computer vision and ML

## 📚 References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Gradio Documentation](https://www.gradio.app/)
- [YOLO Object Detection](https://github.com/ultralytics/ultralytics)

## 📄 License

This project uses datasets from Roboflow. Refer to the dataset-specific licenses:
- Dataset License: CC BY 4.0
- See: [Roboflow Smart Waste Detection](https://universe.roboflow.com/aryas-workspace-yc2uz/smart-waste-detection-mvy71/)

## 🤝 Contributing

Feel free to extend this project with:
- Additional waste categories
- Real-time webcam support
- Mobile app integration
- Edge device deployment
- Performance optimizations

## ✨ Future Enhancements

- [ ] Real-time webcam streaming
- [ ] Batch image processing
- [ ] Export results to CSV/JSON
- [ ] Model quantization for edge devices
- [ ] TensorFlow Lite conversion
- [ ] Docker containerization
- [ ] API endpoint for integration
- [ ] Mobile app wrapper

---

**Happy waste detecting! 🗑️ Remember to recycle! ♻️**
