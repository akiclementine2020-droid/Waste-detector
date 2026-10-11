"""
Waste Detection System with Gradio Interface
Uses YOLOv8 for object detection and classification
"""
import gradio as gr
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import os
from pathlib import Path
import json
from datetime import datetime


class WasteDetectionSystem:
    """Waste Detection System using YOLOv8"""
    
    # Waste class colors for visualization
    CLASS_COLORS = {
        'bottle': (255, 0, 0),        # Blue
        'cardboard': (0, 165, 255),   # Orange
        'metal': (192, 192, 192),     # Gray
        'organic': (34, 139, 34),     # Green
        'paper': (255, 255, 0),       # Cyan
        'plastic': (128, 0, 128)      # Purple
    }
    
    CONFIDENCE_THRESHOLD = 0.5
    
    def __init__(self, model_path: str = None):
        """
        Initialize the waste detection system.
        
        Args:
            model_path: Path to trained YOLOv8 model
        """
        # Try multiple model locations
        if model_path is None:
            possible_paths = [
                "runs/detect/models/waste_detector/weights/best.pt",  # Training output
                "models/waste_detector/weights/best.pt",  # Alternative location
            ]
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
        
        self.model_path = model_path
        self.model = None
        self.class_names = ['bottle', 'cardboard', 'metal', 'organic', 'paper', 'plastic']
        
        if model_path and os.path.exists(model_path):
            self.load_model()
        else:
            print(f"Warning: Model not found at {model_path}")
            print("Please train the model first using train.py")
    
    def load_model(self):
        """Load YOLOv8 model"""
        try:
            print(f"Loading model from {self.model_path}")
            self.model = YOLO(self.model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def detect(self, image: Image.Image, confidence: float = 0.5) -> tuple:
        """
        Detect waste objects in image.
        
        Args:
            image: PIL Image
            confidence: Confidence threshold
            
        Returns:
            Tuple of (annotated_image, detections_dict)
        """
        if self.model is None:
            return image, {"error": "Model not loaded"}
        
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Run inference
        results = self.model(img_array, conf=confidence, verbose=False)
        
        # Process results
        detections = self._process_results(results[0])
        
        # Draw annotations
        annotated_image = self._draw_annotations(image, results[0])
        
        return annotated_image, detections
    
    def _process_results(self, result) -> dict:
        """Process YOLOv8 results into structured format"""
        detections = {
            "total_objects": 0,
            "by_class": {},
            "details": []
        }
        
        if result.boxes is not None and len(result.boxes) > 0:
            detections["total_objects"] = len(result.boxes)
            
            # Initialize class counts
            for cls_name in self.class_names:
                detections["by_class"][cls_name] = 0
            
            # Process each detection
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else "unknown"
                confidence = float(box.conf[0])
                
                detections["by_class"][class_name] += 1
                detections["details"].append({
                    "class": class_name,
                    "confidence": f"{confidence:.2%}",
                    "coordinates": {
                        "x1": float(box.xyxy[0][0]),
                        "y1": float(box.xyxy[0][1]),
                        "x2": float(box.xyxy[0][2]),
                        "y2": float(box.xyxy[0][3])
                    }
                })
        
        return detections
    
    def _draw_annotations(self, image: Image.Image, result) -> Image.Image:
        """Draw bounding boxes and labels on image"""
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        
        if result.boxes is None or len(result.boxes) == 0:
            return img_copy
        
        # Try to load a default font, fall back to default if not available
        try:
            font_size = int(image.width / 80)
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else "unknown"
            confidence = float(box.conf[0])
            
            # Get color for this class
            color = self.CLASS_COLORS.get(class_name, (255, 255, 255))
            # Convert BGR to RGB if needed
            color = color[::-1] if len(color) == 3 else color
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw bounding box
            box_width = 3
            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline=color,
                width=box_width
            )
            
            # Draw label with background
            label = f"{class_name}: {confidence:.2%}"
            bbox = draw.textbbox((x1, y1 - 25), label, font=font)
            
            # Draw label background
            draw.rectangle(
                [bbox[0] - 5, bbox[1] - 5, bbox[2] + 5, bbox[3] + 5],
                fill=color
            )
            
            # Draw label text
            draw.text((x1, y1 - 20), label, fill=(255, 255, 255), font=font)
        
        return img_copy
    
    def get_statistics(self, detections: dict) -> str:
        """Generate statistics summary from detections"""
        if "error" in detections:
            return "⚠️ Error: Model not available for detection"
        
        total = detections["total_objects"]
        stats = f"### Detection Summary\n\n"
        stats += f"**Total Objects Detected:** {total}\n\n"
        
        if total == 0:
            stats += "📊 **No waste objects detected in this image.**\n\n"
            stats += "*This could mean:*\n"
            stats += "- No waste items are present in the image\n"
            stats += "- Objects are too small or unclear\n"
            stats += "- Try adjusting the confidence threshold\n"
        else:
            stats += "📦 **Breakdown by Waste Type:**\n\n"
            by_class = detections["by_class"]
            for cls_name, count in sorted(by_class.items()):
                if count > 0:
                    percentage = (count / total) * 100
                    stats += f"- **{cls_name.capitalize()}**: {count} ({percentage:.1f}%)\n"
        
        return stats


# Initialize the detection system
detector = WasteDetectionSystem()


def process_image(image: Image.Image, confidence_slider: float):
    """Process image for waste detection"""
    if image is None:
        return None, "Please upload an image", "No detections yet", "No data", ""
    
    if detector.model is None:
        return image, "❌ Model not loaded. Please train the model first using train.py", "Model Error", "Error", ""
    
    # Run detection
    annotated_image, detections = detector.detect(image, confidence=confidence_slider)
    
    # Generate statistics
    stats = detector.get_statistics(detections)
    
    # Generate statistical cards HTML
    stats_cards = create_statistics_cards(detections)
    
    # Generate detailed report
    if detections["total_objects"] > 0:
        report = "### Detection Details\n\n"
        report += f"📍 **Total Objects Found:** {detections['total_objects']}\n\n"
        report += "**Classification Breakdown:**\n\n"
        for cls_name, count in detections['by_class'].items():
            if count > 0:
                report += f"- **{cls_name.capitalize()}**: {count} object(s)\n"
        report += "\n**Individual Detections:**\n\n"
        for i, det in enumerate(detections['details'], 1):
            report += f"**{i}. {det['class'].capitalize()}**\n"
            report += f"   - Confidence: {det['confidence']}\n"
            report += f"   - Position: ({det['coordinates']['x1']:.0f}, {det['coordinates']['y1']:.0f})\n\n"
    else:
        report = "### Detection Details\n\n"
        report += "✓ **Analysis Complete**\n\n"
        report += "📊 **Result:** No waste objects detected in this image.\n\n"
        report += "*The detection system scanned the image and found no recognizable waste items.*\n"
        report += "*Try uploading an image with clearly visible waste items like bottles, cardboard, plastic, etc.*"
    
    # Generate unclassified waste section
    unclassified_info = get_unclassified_info(detections)
    
    return annotated_image, stats_cards, report, stats, unclassified_info


def create_statistics_cards(detections: dict) -> str:
    """Create HTML statistics cards for detection summary"""
    if "error" in detections:
        return "<div class='stat-card-error'>⚠️ Model Error</div>"
    
    total = detections["total_objects"]
    by_class = detections["by_class"]
    
    # Count classified objects
    classified_count = sum(count for count in by_class.values())
    
    html = "<div class='statistics-container'>"
    
    # Total objects card
    html += f"""
    <div class='stat-card stat-card-primary'>
        <div class='stat-value'>{total}</div>
        <div class='stat-label'>Total Objects</div>
        <div class='stat-icon'>📦</div>
    </div>
    """
    
    # Classified objects card
    html += f"""
    <div class='stat-card stat-card-success'>
        <div class='stat-value'>{classified_count}</div>
        <div class='stat-label'>Classified</div>
        <div class='stat-icon'>✓</div>
    </div>
    """
    
    # Most common class card
    if classified_count > 0:
        most_common_class = max(by_class.items(), key=lambda x: x[1])
        html += f"""
        <div class='stat-card stat-card-info'>
            <div class='stat-value'>{most_common_class[1]}</div>
            <div class='stat-label'>Most Common: {most_common_class[0].capitalize()}</div>
            <div class='stat-icon'>🔝</div>
        </div>
        """
    
    # Class breakdown mini cards
    if classified_count > 0:
        html += "<div class='class-breakdown'>"
        html += "<div class='breakdown-title'>Class Breakdown</div>"
        for cls_name, count in sorted(by_class.items()):
            if count > 0:
                percentage = (count / total) * 100 if total > 0 else 0
                html += f"""
                <div class='breakdown-item'>
                    <span class='breakdown-name'>{cls_name.capitalize()}</span>
                    <span class='breakdown-count'>{count}</span>
                    <div class='breakdown-bar'>
                        <div class='breakdown-fill' style='width: {percentage}%'></div>
                    </div>
                </div>
                """
        html += "</div>"
    
    html += "</div>"
    return html


def get_unclassified_info(detections: dict) -> str:
    """Generate information about unclassified or uncertain detections"""
    if "error" in detections:
        return ""
    
    total = detections["total_objects"]
    by_class = detections["by_class"]
    classified_count = sum(count for count in by_class.values())
    unclassified_count = total - classified_count
    
    html = "<div class='unclassified-section'>"
    html += "<div class='section-header'>📊 Detection Analysis</div>"
    
    if unclassified_count == 0 and total == 0:
        html += "<div class='analysis-message'>✓ Image scanned - No waste objects detected</div>"
    elif unclassified_count == 0:
        html += f"<div class='analysis-message'>✓ All {total} detected object(s) successfully classified</div>"
    else:
        html += f"""
        <div class='analysis-warning'>
            ⚠️ {unclassified_count} object(s) could not be classified
        </div>
        <div class='analysis-details'>
            <p><strong>Reasons for unclassification:</strong></p>
            <ul>
                <li>Objects may be partially obscured or at unusual angles</li>
                <li>Low image quality or poor lighting conditions</li>
                <li>Objects appear similar to multiple waste types</li>
                <li>Model confidence is below the selected threshold</li>
            </ul>
            <p><strong>Suggestion:</strong> Try adjusting the confidence threshold slider or providing a clearer image.</p>
        </div>
        """
    
    # Add confidence note
    if total > 0:
        avg_confidence = np.mean([float(det['confidence'].rstrip('%')) 
                                  for det in detections['details']]) if detections['details'] else 0
        html += f"""
        <div class='confidence-note'>
            📈 Average Detection Confidence: {avg_confidence:.1f}%
        </div>
        """
    
    html += "</div>"
    return html


# Custom CSS styling
CUSTOM_CSS = """
/* Orange theme colors */
:root {
    --orange-primary: #FF8C00;
    --orange-hover: #E67E00;
    --orange-light: #FFB84D;
}

.statistics-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 15px;
    margin-bottom: 20px;
}

.stat-card {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.stat-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
}

.stat-card-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}

.stat-card-success {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
}

.stat-card-info {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    color: white;
}

.stat-card-error {
    background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
    font-weight: bold;
}

.stat-value {
    font-size: 32px;
    font-weight: bold;
    margin: 10px 0;
}

.stat-label {
    font-size: 14px;
    opacity: 0.9;
    font-weight: 500;
}

.stat-icon {
    font-size: 24px;
    margin-top: 8px;
}

.class-breakdown {
    grid-column: 1 / -1;
    background: #f8f9fa;
    padding: 15px;
    border-radius: 10px;
    margin-top: 10px;
}

.breakdown-title {
    font-weight: bold;
    margin-bottom: 12px;
    font-size: 14px;
    color: #333;
}

.breakdown-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
    font-size: 13px;
}

.breakdown-name {
    flex: 0 0 120px;
    font-weight: 500;
    color: #333;
}

.breakdown-count {
    flex: 0 0 40px;
    text-align: right;
    font-weight: bold;
    color: #667eea;
}

.breakdown-bar {
    flex: 1;
    height: 6px;
    background: #e0e0e0;
    border-radius: 3px;
    margin: 0 10px;
    overflow: hidden;
}

.breakdown-fill {
    height: 100%;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    border-radius: 3px;
}

.unclassified-section {
    background: #f8f9fa;
    border: 2px solid #e9ecef;
    border-radius: 12px;
    padding: 20px;
    margin-top: 20px;
}

.section-header {
    font-size: 16px;
    font-weight: bold;
    color: #333;
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 2px solid #667eea;
}

.analysis-message {
    background: #d4edda;
    color: #155724;
    padding: 12px;
    border-radius: 8px;
    border-left: 4px solid #28a745;
    margin-bottom: 10px;
    font-weight: 500;
}

.analysis-warning {
    background: #fff3cd;
    color: #856404;
    padding: 12px;
    border-radius: 8px;
    border-left: 4px solid #ffc107;
    margin-bottom: 10px;
    font-weight: 500;
}

.analysis-details {
    background: white;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 10px;
    font-size: 13px;
    line-height: 1.6;
}

.analysis-details p {
    margin: 8px 0;
}

.analysis-details ul {
    margin: 8px 0 8px 20px;
    padding: 0;
}

.analysis-details li {
    margin: 4px 0;
}

.confidence-note {
    background: #e7f3ff;
    color: #0c5aa0;
    padding: 12px;
    border-radius: 8px;
    border-left: 4px solid #0084ff;
    font-weight: 500;
    font-size: 13px;
}

/* Orange Button Styling */
button.primary {
    background: linear-gradient(135deg, #FF8C00 0%, #FF7500 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: bold !important;
}

button.primary:hover {
    background: linear-gradient(135deg, #E67E00 0%, #E67000 100%) !important;
}

button.primary:active {
    background: linear-gradient(135deg, #D97000 0%, #D96500 100%) !important;
}

/* Slider styling for orange theme */
input[type="range"] {
    accent-color: #FF8C00 !important;
}

input[type="range"]::-webkit-slider-thumb {
    background: #FF8C00 !important;
    border-color: #FF8C00 !important;
}

input[type="range"]::-moz-range-thumb {
    background: #FF8C00 !important;
    border-color: #FF8C00 !important;
}

input[type="range"]::-webkit-slider-runnable-track {
    background: linear-gradient(to right, #FF8C00 0%, #FF8C00 var(--value), #ddd var(--value), #ddd 100%) !important;
}

/* Gradio slider component styling */
.slider-item input[type="range"] {
    accent-color: #FF8C00 !important;
}

/* Orange styling for Image Upload Components */
.image-container label,
.file-input label {
    color: #FF8C00 !important;
    font-weight: 600 !important;
}

/* Target Gradio image component labels */
.block-label {
    color: #FF8C00 !important;
    font-weight: 600 !important;
}

/* Image upload button styling */
.image-upload-container button,
.upload-button {
    background: linear-gradient(135deg, #FF8C00 0%, #FF7500 100%) !important;
    border: 2px solid #FF8C00 !important;
    color: white !important;
    font-weight: bold !important;
}

.image-upload-container button:hover,
.upload-button:hover {
    background: linear-gradient(135deg, #E67E00 0%, #E67000 100%) !important;
    border-color: #E67E00 !important;
}

/* Gradio Image component label styling */
div[class*="image"] .label-wrap label {
    color: #FF8C00 !important;
    font-weight: 600 !important;
}

/* Confidence threshold label styling */
.slider .label-wrap label {
    color: #FF8C00 !important;
    font-weight: 600 !important;
}

/* General component label styling */
.form-group label,
.component-label {
    color: #FF8C00 !important;
    font-weight: 600 !important;
}

/* Gradio textbox with label */
input[type="text"] {
    border: 2px solid #FF8C00 !important;
}

input[type="text"]:focus {
    border-color: #E67E00 !important;
    box-shadow: 0 0 0 3px rgba(255, 140, 0, 0.1) !important;
}

/* Target specific Gradio component labels more broadly */
.gradio-image label:first-of-type,
.gradio-slider label:first-of-type {
    color: #FF8C00 !important;
    font-weight: 600 !important;
}

/* Responsive design */
@media (max-width: 768px) {
    .statistics-container {
        grid-template-columns: 1fr;
    }
    
    .stat-value {
        font-size: 24px;
    }
    
    .breakdown-item {
        flex-wrap: wrap;
    }
    
    .breakdown-bar {
        flex: 1 1 100%;
        margin: 5px 0;
    }
}

/* Enhanced interface styling */
.gradio-container {
    max-width: 1200px;
}

.gradio-box {
    border-radius: 10px;
    border: 1px solid #e0e0e0;
}

.gradio-textbox, 
.gradio-markdown {
    border-radius: 8px;
}

#interface {
    background: linear-gradient(to bottom, #f8f9fa 0%, #ffffff 100%);
}
"""


def create_interface():
    """Create enhanced Gradio interface with theme and custom styling"""
    
    # Use Gradio's built-in soft theme
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
    ).set(
        body_background_fill="#f8f9fa",
        body_text_color="#333333",
        block_title_text_color="#667eea",
        border_color_primary="#e0e0e0",
    )
    
    with gr.Blocks(
        title="Waste Detection System",
        theme=theme,
        css=CUSTOM_CSS
    ) as demo:
        # Header section
        gr.Markdown("# 🗑️ Waste Detection System")
        gr.Markdown(
            """
            Detect and classify waste objects using YOLOv8 and AI-powered computer vision.
            
            **Waste Categories:** Bottle • Cardboard • Metal • Organic • Paper • Plastic
            
            Upload an image to get started!
            """
        )
        
        gr.Markdown("---")
        
        # Main content area
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload & Settings")
                image_input = gr.Image(
                    type="pil",
                    label="Input Image",
                    interactive=True
                )
                
                confidence_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    label="Confidence Threshold",
                    info="Higher value = stricter detection"
                )
                
                detect_button = gr.Button(
                    "🔍 Detect Waste",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 Detection Results")
                image_output = gr.Image(
                    type="pil",
                    label="Annotated Image",
                    interactive=False
                )
        
        gr.Markdown("---")
        
        # Statistics and metrics section
        gr.Markdown("### 📊 Detection Summary & Metrics")
        stats_cards_output = gr.HTML(value="<p>Upload an image to see statistics</p>")
        
        gr.Markdown("---")
        
        # Detailed information section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📋 Classification Report")
                report_output = gr.Markdown("Detection details will appear here")
            
            with gr.Column():
                gr.Markdown("### 📈 Analysis & Unclassified Objects")
                unclassified_output = gr.HTML(
                    value="<div class='unclassified-section'><div class='section-header'>📊 Detection Analysis</div><div class='analysis-message'>✓ Image scanned - Ready to analyze waste objects</div></div>"
                )
        
        # Original statistics markdown (kept for reference)
        stats_output = gr.Markdown("Upload an image to see detailed statistics", visible=False)
        
        # Event handlers
        detect_button.click(
            fn=process_image,
            inputs=[image_input, confidence_slider],
            outputs=[image_output, stats_cards_output, report_output, stats_output, unclassified_output]
        )
        
        # Auto-detect on image upload
        image_input.change(
            fn=process_image,
            inputs=[image_input, confidence_slider],
            outputs=[image_output, stats_cards_output, report_output, stats_output, unclassified_output]
        )
        
        gr.Markdown("---")
        
        # Information section
        gr.Markdown(
            """
            ### ℹ️ About This System
            
            This waste detection system uses **YOLOv8**, a state-of-the-art object detection model, 
            trained on labeled waste images to support proper waste management and recycling initiatives.
            
            **How to use:**
            1. Upload an image containing waste items
            2. Adjust the confidence threshold if needed (higher = more confident detections only)
            3. View detected objects with bounding boxes, classifications, and confidence scores
            4. Review the analysis section for insights about detection quality
            
            **Tips for best results:**
            - Use clear, well-lit images
            - Ensure waste objects are clearly visible
            - Adjust confidence threshold if detection is too strict or loose
            """
        )
    
    return demo


if __name__ == "__main__":
    print("🗑️ Waste Detection System - Starting Gradio Interface")
    print(f"Model path: {detector.model_path}")
    
    if detector.model is None:
        print("\n⚠️  WARNING: Model not found!")
        print("Please train the model first using:")
        print("  python train.py --data data.yaml --epochs 100")
        print("\nThe interface will still start, but detection won't work until the model is trained.")
    
    demo = create_interface()
    
    # Try to find an available port
    import socket
    port = 7860
    for p in range(7860, 7875):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", p))
            sock.close()
            port = p
            break
        except OSError:
            continue
    
    print(f"\n✓ Launching at http://localhost:{port}")
    
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=port,
        show_error=True
    )
