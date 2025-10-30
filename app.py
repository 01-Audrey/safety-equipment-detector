"""
Safety Equipment Detector - Streamlit App
Real-time PPE detection for construction sites
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Safety Equipment Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    """Load YOLOv8 model (cached for performance)"""
    try:
        # Try to load your trained model
        model = YOLO('notebooks/runs/detect/safety_detector_v3_PRODUCTION/weights/best.pt')
        return model, "custom"
    except:
        # Fallback to pretrained YOLOv8s
        st.warning("Custom model not found. Using pretrained YOLOv8s.")
        model = YOLO('yolov8s.pt')
        return model, "pretrained"

# Load model
model, model_type = load_model()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🛡️ Safety Equipment Detector")
st.sidebar.markdown("---")

st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Minimum confidence for detections"
)

iou_threshold = st.sidebar.slider(
    "IoU Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.45,
    step=0.05,
    help="IoU threshold for NMS"
)

st.sidebar.markdown("---")
st.sidebar.header("📊 Model Info")
if model_type == "custom":
    st.sidebar.success("✅ Production Model (v3)")
    st.sidebar.metric("mAP@50", "75.1%")
    st.sidebar.metric("Inference Speed", "3.8ms")
else:
    st.sidebar.info("ℹ️ Pretrained Model")

st.sidebar.markdown("---")
st.sidebar.header("🎯 Detectable Classes")
classes = {
    "✅": ["Helmet", "Safety Vest", "Person"],
    "⚠️": ["No Helmet", "No Vest"]
}

for category, items in classes.items():
    st.sidebar.markdown(f"**{category}**")
    for item in items:
        st.sidebar.markdown(f"- {item}")

st.sidebar.markdown("---")
st.sidebar.markdown("**📁 [GitHub Repository](https://github.com/01000001-A/Safety-Equipment-Detector)**")
st.sidebar.markdown("**👤 Created by Audrey**")

# ============================================================
# MAIN APP
# ============================================================
st.markdown("<h1 class='main-header'>🛡️ Safety Equipment Detector</h1>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <p style='font-size: 1.2rem; color: #666;'>
        AI-powered PPE detection for construction site safety monitoring
    </p>
    <p style='font-size: 1rem; color: #888;'>
        Upload an image to detect helmets, safety vests, and compliance violations
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# FILE UPLOADER
# ============================================================
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Upload a construction site image"
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Create columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Original Image")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("🔍 Detection Results")
        
        # Run inference
        with st.spinner("Detecting safety equipment..."):
            results = model(
                img_array,
                conf=confidence_threshold,
                iou=iou_threshold,
                verbose=False
            )
        
        # Get annotated image
        annotated_img = results[0].plot()
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        
        st.image(annotated_img_rgb, use_container_width=True)
    
    # ============================================================
    # DETECTION STATISTICS
    # ============================================================
    st.markdown("---")
    st.subheader("📊 Detection Statistics")
    
    # Extract detections
    boxes = results[0].boxes
    num_detections = len(boxes)
    
    if num_detections > 0:
        # Count by class
        class_names = results[0].names
        class_counts = {}
        
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = class_names[cls_id]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        
        # Display metrics
        metrics_cols = st.columns(len(class_counts) + 1)
        
        with metrics_cols[0]:
            st.metric("Total Detections", num_detections)
        
        for idx, (cls_name, count) in enumerate(class_counts.items(), 1):
            with metrics_cols[idx]:
                st.metric(cls_name.title(), count)
        
        # Detailed detections table
        st.markdown("---")
        st.subheader("📋 Detailed Detections")
        
        detection_data = []
        for idx, box in enumerate(boxes, 1):
            cls_id = int(box.cls[0])
            cls_name = class_names[cls_id]
            confidence = float(box.conf[0])
            
            detection_data.append({
                "#": idx,
                "Class": cls_name.title(),
                "Confidence": f"{confidence:.2%}"
            })
        
        st.table(detection_data)
        
        # Safety compliance check
        st.markdown("---")
        st.subheader("🔔 Safety Compliance")
        
        has_violations = "no-helmet" in class_counts or "no-vest" in class_counts
        
        if has_violations:
            st.error("⚠️ **Safety Violations Detected!**")
            if "no-helmet" in class_counts:
                st.warning(f"- {class_counts['no-helmet']} worker(s) without helmet")
            if "no-vest" in class_counts:
                st.warning(f"- {class_counts['no-vest']} worker(s) without safety vest")
        else:
            st.success("✅ **No Safety Violations Detected**")
            st.info("All workers appear to be wearing proper safety equipment.")
    
    else:
        st.info("No detections found. Try adjusting the confidence threshold or upload a different image.")

else:
    # ============================================================
    # DEMO SECTION (when no image uploaded)
    # ============================================================
    st.markdown("---")
    st.subheader("💡 How to Use")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **1️⃣ Upload Image**
        - Click the upload button above
        - Choose a construction site photo
        - Supports JPG, JPEG, PNG
        """)
    
    with col2:
        st.markdown("""
        **2️⃣ Adjust Settings**
        - Use sidebar to tune detection
        - Lower threshold = more detections
        - Higher threshold = more confident
        """)
    
    with col3:
        st.markdown("""
        **3️⃣ Review Results**
        - See bounding boxes on image
        - Check detection statistics
        - Verify safety compliance
        """)
    
    st.markdown("---")
    st.subheader("🎯 About This Project")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Model Performance**
        - mAP@50: 75.1% ✅
        - Precision: 73.5%
        - Recall: 72.1%
        - Inference: 3.8ms per image
        
        **Training Details**
        - Model: YOLOv8s (11M parameters)
        - Dataset: 246 images (104 source)
        - Training: 100 epochs, ~35 minutes
        - Hardware: CPU (AMD Ryzen 5 5600X)
        """)
    
    with col2:
        st.markdown("""
        **Detected Equipment**
        - ✅ Safety Helmets (hard hats)
        - ✅ Safety Vests (high-visibility)
        - 👷 Workers/Persons
        - ⚠️ Non-compliance (missing PPE)
        
        **Use Cases**
        - Real-time site monitoring
        - Safety compliance audits
        - Access control verification
        - Training & education
        """)
    
    st.markdown("---")
    st.info("👆 **Upload an image above to get started!**")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Safety Equipment Detector</strong> | Built with YOLOv8 & Streamlit</p>
    <p>From 15.8% to 75.1% mAP in 2 days of focused iteration</p>
    <p>Part of ML Learning Journey (Week 2, Days 12-14)</p>
</div>
""", unsafe_allow_html=True)