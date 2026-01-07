# frontend.py - Streamlit frontend for L-Shape Detection API

import streamlit as st
import requests

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="L-Shape Detection",
    page_icon="🚛",
    layout="wide"
)

# Initialize session state
if 'config_uploaded' not in st.session_state:
    st.session_state.config_uploaded = False
if 'lshape_config' not in st.session_state:
    st.session_state.lshape_config = None
if 'restricted_areas' not in st.session_state:
    st.session_state.restricted_areas = []

# Sidebar - API status
with st.sidebar:
    st.header("API Status")
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        if response.status_code == 200:
            st.success("✅ API is running")
            st.json(response.json())
        else:
            st.error("❌ API returned error")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API")
        st.info("Make sure the API is running:\n```\nuvicorn app:app --reload --port 8000\n```")

# Define pages
config_page = st.Page("pages/config.py", title="Configuration", icon="⚙️")
detection_page = st.Page("pages/detection.py", title="Detection", icon="🔍")

# Navigation
pg = st.navigation([config_page, detection_page])

# Footer
st.markdown("---")
st.markdown("*L-Shape ML Detection API - Powered by YOLOv11 & FastAPI*")

# Run the selected page
pg.run()
