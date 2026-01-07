# pages/detection.py - Detection page

import streamlit as st
import requests
from PIL import Image
import io
import base64

# Configuration
API_URL = "http://localhost:8000"

st.title("🔍 L-Shape ML Detection")
st.markdown("Upload an image to detect **Tyres** within the L-shape zone.")

# Check if configuration is uploaded
if not st.session_state.config_uploaded:
    st.warning("⚠️ Please upload L-shape configuration first in the Configuration page")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    st.header("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload an image to detect tyres"
    )

    # Option to use test images
    st.markdown("---")
    st.subheader("Or use a test image")
    test_image = st.selectbox(
        "Select test image",
        options=["None", "a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    )

    if test_image != "None":
        try:
            with open(f"test_images/{test_image}", "rb") as f:
                uploaded_file = io.BytesIO(f.read())
                uploaded_file.name = test_image
        except FileNotFoundError:
            st.warning(f"Test image {test_image} not found")

with col2:
    st.header("Results")

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Reset file pointer for API request
        if hasattr(uploaded_file, 'seek'):
            uploaded_file.seek(0)

        # Detect button
        if st.button("🔍 Detect Objects", type="primary"):
            with st.spinner("Analyzing image..."):
                try:
                    # Prepare file for upload
                    if hasattr(uploaded_file, 'read'):
                        files = {"file": (uploaded_file.name, uploaded_file.read(), "image/jpeg")}
                    else:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")}

                    # Send request to API
                    response = requests.post(f"{API_URL}/detect", files=files, timeout=30)

                    if response.status_code == 200:
                        result = response.json()
                        detections = result.get("detections", [])
                        annotated_image_b64 = result.get("annotated_image")

                        # Display annotated image
                        if annotated_image_b64:
                            annotated_bytes = base64.b64decode(annotated_image_b64)
                            annotated_image = Image.open(io.BytesIO(annotated_bytes))
                            st.image(annotated_image, caption="Detection Result", use_container_width=True)

                        if detections:
                            st.success(f"Found {len(detections)} object(s)")

                            # Display detections
                            for det in detections:
                                icon = "🛞"
                                zone_status = "🟢 Inside L-Shape" if det["inside_lshape"] else "🔴 Outside L-Shape"

                                st.markdown(f"""
                                **{icon} {det['class']}** (Segment {det['segment_index']})
                                Status: {zone_status}
                                """)
                                st.divider()

                            # Show raw JSON
                            with st.expander("View Raw JSON Response"):
                                st.json(result)
                        else:
                            st.info("No objects detected in the image")
                    else:
                        st.error(f"API Error: {response.status_code}")
                        st.text(response.text)

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Make sure it's running on port 8000.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.info("👆 Upload an image or select a test image to get started")
