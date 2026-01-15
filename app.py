"""
Streamlit App for Pupil Distance Measurement

This app allows users to upload or capture an image, then processes it
to measure inter-pupillary distance using a credit card as reference.
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
from PIL import Image

# Import the measurement module
from pupil_distance_measurement import PupilDistanceMeasurer

# Page configuration
st.set_page_config(
    page_title="Pupil Distance Measurement",
    page_icon="👁️",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .stImage {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>👁️ Pupil Distance Measurement</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center; color: #666;'>
    Measure your inter-pupillary distance using a credit card as reference.
    Hold a credit card horizontally under your nose with the magnetic stripe visible.
</p>
""", unsafe_allow_html=True)

# Initialize session state
if 'measurer' not in st.session_state:
    with st.spinner("Loading face detection model..."):
        st.session_state.measurer = PupilDistanceMeasurer()

if 'results' not in st.session_state:
    st.session_state.results = None

if 'output_image' not in st.session_state:
    st.session_state.output_image = None

if 'last_processed_image' not in st.session_state:
    st.session_state.last_processed_image = None


def process_uploaded_image(image_data, source_name="uploaded_image"):
    """Process an uploaded image and return results."""
    # Create a temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the uploaded image as PNG to avoid JPEG compression artifacts
        temp_image_path = os.path.join(temp_dir, f"{source_name}.png")
        
        # Convert to numpy array if needed
        if isinstance(image_data, np.ndarray):
            # Image from camera is already numpy array (RGB)
            image_rgb = image_data
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        else:
            # Image from file upload
            image_pil = Image.open(image_data)
            
            # Handle EXIF orientation (important for phone photos)
            try:
                from PIL import ExifTags
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                exif = image_pil._getexif()
                if exif is not None:
                    orientation_value = exif.get(orientation)
                    if orientation_value == 3:
                        image_pil = image_pil.rotate(180, expand=True)
                    elif orientation_value == 6:
                        image_pil = image_pil.rotate(270, expand=True)
                    elif orientation_value == 8:
                        image_pil = image_pil.rotate(90, expand=True)
            except (AttributeError, KeyError, IndexError, TypeError):
                # No EXIF data or orientation tag
                pass
            
            # Convert to RGB if necessary
            if image_pil.mode == 'RGBA':
                image_pil = image_pil.convert('RGB')
            elif image_pil.mode != 'RGB':
                image_pil = image_pil.convert('RGB')
            
            image_rgb = np.array(image_pil)
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Save as PNG (lossless) for processing
        cv2.imwrite(temp_image_path, image_bgr)
        
        try:
            # Process the image
            results = st.session_state.measurer.process_image(temp_image_path, temp_dir)
            
            # Read the output image
            output_image_path = results.get('output_image')
            if output_image_path and os.path.exists(output_image_path):
                output_image = cv2.imread(output_image_path)
                output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            else:
                output_image_rgb = None
            
            return results, output_image_rgb
            
        except ValueError as e:
            st.error(f"⚠️ {str(e)}")
            return None, None
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            return None, None


# Instructions section
with st.expander("📋 Instructions", expanded=False):
    st.markdown("""
    ### How to get accurate measurements:
    
    1. **Hold a standard credit card** horizontally under your nose with the magnetic stripe (black stripe) visible
    2. **Face the camera directly** - look straight ahead
    3. **Ensure good lighting** - avoid shadows on your face
    4. **Keep still** while the image is captured
    
    ### What the app detects:
    - **Yellow rectangle**: Detected magnetic stripe on the credit card
    - **Green dots**: Detected pupil centers
    - **Green line**: Connection between pupils (inter-pupillary distance)
    """)

# Image input section
st.markdown("---")
st.subheader("📸 Capture or Upload Image")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📷 Take Photo", "📁 Upload Image"])

with tab1:
    camera_image = st.camera_input("Take a photo with your credit card under your nose")
    
    if camera_image is not None:
        # Create a unique identifier for this image
        image_id = f"camera_{camera_image.id}"
        
        # Auto-process if this is a new image
        if st.session_state.last_processed_image != image_id:
            with st.spinner("Processing image..."):
                results, output_image = process_uploaded_image(camera_image, "camera_capture")
                st.session_state.results = results
                st.session_state.output_image = output_image
                st.session_state.last_processed_image = image_id

with tab2:
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a photo of yourself holding a credit card under your nose"
    )
    
    if uploaded_file is not None:
        # Create a unique identifier for this image
        image_id = f"upload_{uploaded_file.name}_{uploaded_file.size}"
        
        # Auto-process if this is a new image
        if st.session_state.last_processed_image != image_id:
            with st.spinner("Processing image..."):
                results, output_image = process_uploaded_image(uploaded_file, uploaded_file.name.split('.')[0])
                st.session_state.results = results
                st.session_state.output_image = output_image
                st.session_state.last_processed_image = image_id

# Results section
if st.session_state.results is not None and st.session_state.output_image is not None:
    st.markdown("---")
    st.subheader("📊 Results")
    
    results = st.session_state.results
    
    # Display the main measurement prominently
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class='result-box' style='text-align: center;'>
            <p style='margin: 0; color: #666;'>Inter-Pupillary Distance</p>
            <p class='metric-value'>{:.1f} mm</p>
        </div>
        """.format(results['pupil_distance_mm']), unsafe_allow_html=True)
    
    # Display the output image
    st.image(
        st.session_state.output_image,
        caption="Processed Image - Yellow: Card detection, Green: Pupil detection",
        use_container_width=True
    )
    
    # Additional details in an expander
    with st.expander("🔍 Detailed Measurements"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Pupil Distance", f"{results['pupil_distance_mm']:.2f} mm")
            st.metric("Pupil Distance (pixels)", f"{results['pupil_distance_pixels']:.1f} px")
            st.metric("Scale Factor", f"{results['mm_per_pixel']:.4f} mm/px")
        
        with col2:
            st.metric("Card Width (detected)", f"{results['card_width_pixels']:.1f} px")
            st.metric("Card Width (reference)", f"{results['card_width_mm']:.1f} mm")
            
            # Display pupil positions
            left_pupil = results['left_pupil']
            right_pupil = results['right_pupil']
            st.markdown(f"**Left Pupil:** ({left_pupil[0]:.1f}, {left_pupil[1]:.1f})")
            st.markdown(f"**Right Pupil:** ({right_pupil[0]:.1f}, {right_pupil[1]:.1f})")
        
        # Debug info for troubleshooting
        st.markdown("---")
        st.markdown("**Debug Info:**")
        if st.session_state.output_image is not None:
            h, w = st.session_state.output_image.shape[:2]
            st.markdown(f"- Processed image size: {w} x {h} pixels")
    
    # Clear results button
    if st.button("🔄 Clear Results"):
        st.session_state.results = None
        st.session_state.output_image = None
        st.session_state.last_processed_image = None
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: #888; font-size: 0.8rem;'>
    This tool uses MediaPipe for face detection and a credit card's magnetic stripe 
    (79mm standard width) as a reference for accurate measurements.
</p>
""", unsafe_allow_html=True)
