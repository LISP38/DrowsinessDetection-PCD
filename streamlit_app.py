import streamlit as st
import requests
import base64
import json
import cv2
import numpy as np
from PIL import Image
import io
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import deque

# Konfigurasi halaman
st.set_page_config(
    page_title="Drowsiness Detection System",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL FastAPI
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = deque(maxlen=50)
if 'video_mode' not in st.session_state:
    st.session_state.video_mode = False

def check_api_health():
    """Check if FastAPI is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def call_detection_api(image_bytes, endpoint="detect/image"):
    """Call FastAPI detection endpoint"""
    try:
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        response = requests.post(f"{API_BASE_URL}/{endpoint}", files=files, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None

def display_detection_result(result):
    """Display detection results in a nice format"""
    if not result or not result.get('success'):
        st.error("Failed to get detection result")
        return
    
    # Main status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if result['drowsiness_detected']:
            st.error("⚠️ DROWSINESS DETECTED!")
        else:
            st.success("✅ Alert")
    
    with col2:
        st.metric("Drowsiness Score", f"{result['drowsiness_score']}")
    
    with col3:
        # Ubah dari 'confidence' ke 'status'
        st.metric("Status", result.get('status', 'Unknown'))
    
    with col4:
        # Perbaiki logika face detection
        face_detected = result.get('face_detected', result.get('status') != "No Face Detected")
        face_status = "✅ Detected" if face_detected else "❌ Not Detected"
        st.metric("Face", face_status)
    
    # Metrics - sesuaikan dengan struktur response FastAPI
    st.subheader("📊 Detection Metrics")
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        ear_value = result['metrics']['ear']
        st.metric(
            "Eye Aspect Ratio (EAR)", 
            f"{ear_value:.3f}",
            help="Lower values indicate closed/droopy eyes"
        )
        if ear_value < 0.19:
            st.warning("Eyes appear closed/droopy")
    
    with metrics_col2:
        mar_value = result['metrics']['mar']
        st.metric(
            "Mouth Aspect Ratio (MAR)", 
            f"{mar_value:.3f}",
            help="Higher values indicate yawning"
        )
        if mar_value > 0.6:
            st.warning("Yawning detected")
    
    with metrics_col3:
        head_movement = result['metrics']['head_movement']
        st.metric(
            "Head Movement", 
            f"{head_movement:.3f}",
            help="Excessive head movement may indicate drowsiness"
        )
        if head_movement > 8:
            st.warning("Excessive head movement")
    
    # Alerts - gunakan alerts dari FastAPI
    if result.get('alerts'):
        st.subheader("🚨 Alerts")
        for alert in result['alerts']:
            st.warning(f"• {alert}")
    
    # Processed image - sesuaikan dengan key yang benar
    image_key = 'processed_image' if 'processed_image' in result else 'processed_frame'
    if image_key in result:
        st.subheader("🖼️ Processed Image")
        try:
            # Decode base64 image
            image_data = base64.b64decode(result[image_key])
            image = Image.open(io.BytesIO(image_data))
            st.image(image, caption="Detection Result", use_column_width=True)
        except Exception as e:
            st.error(f"Error displaying processed image: {e}")
    """Display detection results in a nice format"""
    if not result or not result.get('success'):
        st.error("Failed to get detection result")
        return
    
    # Main status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if result['drowsiness_detected']:
            st.error("⚠️ DROWSINESS DETECTED!")
        else:
            st.success("✅ Alert")
    
    with col2:
        st.metric("Drowsiness Score", f"{result['drowsiness_score']}")
    
    with col3:
        st.metric("Confidence", f"{result.get('confidence', 0):.2f}")
    
    with col4:
        face_status = "✅ Detected" if result['face_detected'] else "❌ Not Detected"
        st.metric("Face", face_status)
    
    # Metrics
    st.subheader("📊 Detection Metrics")
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        ear_value = result['metrics']['ear']
        st.metric(
            "Eye Aspect Ratio (EAR)", 
            f"{ear_value:.3f}",
            help="Lower values indicate closed/droopy eyes"
        )
        if ear_value < 0.19:
            st.warning("Eyes appear closed/droopy")
    
    with metrics_col2:
        mar_value = result['metrics']['mar']
        st.metric(
            "Mouth Aspect Ratio (MAR)", 
            f"{mar_value:.3f}",
            help="Higher values indicate yawning"
        )
        if mar_value > 0.6:
            st.warning("Yawning detected")
    
    with metrics_col3:
        head_movement = result['metrics']['head_movement']
        st.metric(
            "Head Movement", 
            f"{head_movement:.3f}",
            help="Excessive head movement may indicate drowsiness"
        )
        if head_movement > 8:
            st.warning("Excessive head movement")
    
    # Alerts
    if result['alerts']:
        st.subheader("🚨 Alerts")
        for alert in result['alerts']:
            st.warning(f"• {alert}")
    
    # Processed image
    if 'processed_image' in result:
        st.subheader("🖼️ Processed Image")
        try:
            # Decode base64 image
            image_data = base64.b64decode(result['processed_image'])
            image = Image.open(io.BytesIO(image_data))
            st.image(image, caption="Detection Result", use_column_width=True)
        except Exception as e:
            st.error(f"Error displaying processed image: {e}")

def display_history_chart():
    """Display detection history chart"""
    if len(st.session_state.detection_history) > 1:
        st.subheader("📈 Detection History")
        
        # Convert history to DataFrame
        df_data = []
        for i, record in enumerate(st.session_state.detection_history):
            df_data.append({
                'Frame': i,
                'Drowsiness Score': record.get('drowsiness_score', 0),
                'EAR': record['metrics'].get('ear', 0),
                'MAR': record['metrics'].get('mar', 0),
                'Drowsy': 1 if record.get('drowsiness_detected', False) else 0
            })
        
        df = pd.DataFrame(df_data)
        
        # Create plotly chart
        fig = go.Figure()
        
        # Add drowsiness score
        fig.add_trace(go.Scatter(
            x=df['Frame'],
            y=df['Drowsiness Score'],
            mode='lines+markers',
            name='Drowsiness Score',
            line=dict(color='red'),
            yaxis='y'
        ))
        
        # Add EAR
        fig.add_trace(go.Scatter(
            x=df['Frame'],
            y=df['EAR'] * 100,  # Scale for visibility
            mode='lines',
            name='EAR (x100)',
            line=dict(color='blue'),
            yaxis='y'
        ))
        
        # Add drowsiness threshold line
        fig.add_hline(y=70, line_dash="dash", line_color="orange", 
                     annotation_text="Drowsiness Threshold")
        
        fig.update_layout(
            title="Detection Metrics Over Time",
            xaxis_title="Frame Number",
            yaxis_title="Values",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def main():
    # Header
    st.title("😴 Drowsiness Detection System")
    st.markdown("---")
    
    # Check API status
    api_status = check_api_health()
    if not api_status:
        st.error("🔴 FastAPI server is not running! Please start the server first.")
        st.code("uvicorn main:app --reload", language="bash")
        st.stop()
    else:
        st.success("🟢 FastAPI server is running")
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "Detection Mode",
        ["Single Image", "Video Mode (Frame by Frame)"],
        help="Choose between single image detection or video frame processing"
    )
    
    st.session_state.video_mode = (mode == "Video Mode (Frame by Frame)")
    
    # Configuration section
    st.sidebar.subheader("⚙️ Configuration")
    if st.sidebar.button("Get Current Config"):
        try:
            response = requests.get(f"{API_BASE_URL}/config")
            if response.status_code == 200:
                config = response.json()['config']
                st.sidebar.json(config)
            else:
                st.sidebar.error("Failed to get configuration")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
    
    # Reset button
    if st.sidebar.button("🔄 Reset Detector State"):
        try:
            response = requests.post(f"{API_BASE_URL}/reset")
            if response.status_code == 200:
                st.sidebar.success("Detector state reset successfully")
                st.session_state.detection_history.clear()
            else:
                st.sidebar.error("Failed to reset detector")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
    
    # Statistics
    st.sidebar.subheader("📊 Statistics")
    if st.sidebar.button("Show Statistics"):
        try:
            response = requests.get(f"{API_BASE_URL}/statistics")
            if response.status_code == 200:
                stats = response.json()['statistics']
                st.sidebar.json(stats)
            else:
                st.sidebar.error("Failed to get statistics")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Detection", "📊 Analytics", "ℹ️ About"])
    
    with tab1:
        if mode == "Single Image":
            st.subheader("📷 Single Image Detection")
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png'],
                help="Upload an image containing a face for drowsiness detection"
            )
            
            if uploaded_file is not None:
                # Display original image
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Image")
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                
                with col2:
                    st.subheader("Detection Results")
                    
                    if st.button("🔍 Analyze Image", type="primary"):
                        with st.spinner("Analyzing image..."):
                            # Convert to bytes
                            img_bytes = io.BytesIO()
                            image.save(img_bytes, format='JPEG')
                            img_bytes = img_bytes.getvalue()
                            
                            # Call API
                            result = call_detection_api(img_bytes, "detect/image")
                            
                            if result:
                                display_detection_result(result)
                                
                                # Add to history
                                st.session_state.detection_history.append(result)
        
        else:  # Video Mode
            st.subheader("🎥 Video Mode (Frame by Frame)")
            st.info("In video mode, the detector maintains state between frames for more accurate detection.")
            
            # Camera input
            camera_input = st.camera_input("Take a photo")
            
            if camera_input is not None:
                # Display captured image
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Captured Frame")
                    st.image(camera_input, caption="Current Frame", use_column_width=True)
                
                with col2:
                    st.subheader("Detection Results")
                    
                    if st.button("🔍 Process Frame", type="primary"):
                        with st.spinner("Processing frame..."):
                            # Call API with video-frame endpoint
                            result = call_detection_api(camera_input.getvalue(), "detect/video-frame")
                            
                            if result:
                                display_detection_result(result)
                                
                                # Add to history
                                st.session_state.detection_history.append(result)
            
            # File upload for video mode
            st.markdown("---")
            st.subheader("Or upload an image for video-style processing")
            
            uploaded_file = st.file_uploader(
                "Upload frame",
                type=['jpg', 'jpeg', 'png'],
                key="video_upload"
            )
            
            if uploaded_file is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Frame to Process", use_column_width=True)
                
                with col2:
                    if st.button("🔍 Process Video Frame", type="primary"):
                        with st.spinner("Processing frame..."):
                            # Convert to bytes
                            img_bytes = io.BytesIO()
                            image.save(img_bytes, format='JPEG')
                            img_bytes = img_bytes.getvalue()
                            
                            result = call_detection_api(img_bytes, "detect/video-frame")
                            
                            if result:
                                display_detection_result(result)
                                st.session_state.detection_history.append(result)
    
    with tab2:
        st.subheader("📊 Analytics Dashboard")
        
        # Display history chart
        display_history_chart()
        
        # Summary statistics
        if st.session_state.detection_history:
            st.subheader("📈 Session Summary")
            
            total_frames = len(st.session_state.detection_history)
            drowsy_frames = sum(1 for record in st.session_state.detection_history 
                              if record.get('drowsiness_detected', False))
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Frames", total_frames)
            
            with col2:
                st.metric("Drowsy Frames", drowsy_frames)
            
            with col3:
                drowsy_percentage = (drowsy_frames / total_frames * 100) if total_frames > 0 else 0
                st.metric("Drowsy %", f"{drowsy_percentage:.1f}%")
            
            with col4:
                avg_score = sum(record.get('drowsiness_score', 0) 
                              for record in st.session_state.detection_history) / total_frames
                st.metric("Avg Score", f"{avg_score:.1f}")
        
        else:
            st.info("No detection data available. Start analyzing images to see analytics.")
    
    with tab3:
        st.subheader("ℹ️ About This System")
        
        st.markdown("""
        ### Drowsiness Detection System
        
        This system uses computer vision and machine learning to detect signs of drowsiness in real-time:
        
        **Detection Methods:**
        - **Eye Aspect Ratio (EAR)**: Measures eye openness
        - **Mouth Aspect Ratio (MAR)**: Detects yawning
        - **Head Movement**: Tracks head position changes
        
        **Features:**
        - Real-time processing
        - Configurable thresholds
        - Video mode with frame context
        - Statistical analysis
        - Visual feedback
        
        **Thresholds:**
        - EAR < 0.19: Eyes closed/droopy
        - MAR > 0.6: Yawning detected
        - Head Movement > 8: Excessive movement
        - Drowsiness Score > 70: Alert triggered
        """)
        
        st.subheader("🔧 Technical Details")
        st.markdown("""
        - **Backend**: FastAPI
        - **Frontend**: Streamlit
        - **Computer Vision**: OpenCV, dlib
        - **Face Detection**: 68-point facial landmarks
        - **Processing**: Real-time frame analysis
        """)

if __name__ == "__main__":
    main()