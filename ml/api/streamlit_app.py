import streamlit as st
import requests
from PIL import Image
import io
import os

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{API_URL}/predict"
HEALTH_ENDPOINT = f"{API_URL}/health"

st.set_page_config(page_title="BCS Cow Classifier", page_icon="🐄")

st.title("🐄 Dairy Cow BCS Classifier")
st.write("Upload an image of a **cow's rear view** to classify its Body Condition Score (BCS).")

# Sidebar for settings
st.sidebar.header("Settings")
api_url_input = st.sidebar.text_input("API URL", value=API_URL)
if api_url_input != API_URL:
    PREDICT_ENDPOINT = f"{api_url_input}/predict"
    HEALTH_ENDPOINT = f"{api_url_input}/health"

# Check API health status
st.sidebar.markdown("---")
st.sidebar.subheader("API Status")
try:
    health_response = requests.get(HEALTH_ENDPOINT, timeout=5)
    if health_response.status_code == 200:
        health_data = health_response.json()
        st.sidebar.success("✅ API Connected")
        if 'details' in health_data:
            details = health_data['details']
            st.sidebar.caption(f"BCS Model: {details.get('bcs_model', 'unknown')}")
            st.sidebar.caption(f"Validator Model: {details.get('validator_model', 'unknown')}")
    else:
        st.sidebar.warning("⚠️ API Unhealthy")
except:
    st.sidebar.error("❌ API Not Connected")

# Info box about image requirements
st.info("📸 **Catatan:** Pastikan gambar yang diupload adalah tampilan **bagian belakang sapi**. Gambar selain itu akan ditolak oleh sistem.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Create columns for side-by-side layout
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Button and Status in the left column (below image)
        if st.button("Predict", use_container_width=True):
            with st.spinner("Validating and classifying..."):
                try:
                    # Prepare file for API
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    
                    # Call API
                    response = requests.post(PREDICT_ENDPOINT, files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display status in left column
                        st.success("✅ Classification Successful!")
                        
                        # Display results in right column
                        with col2:
                            st.write("### 📊 Classification Results")
                            metric_col1, metric_col2 = st.columns(2)
                            with metric_col1:
                                st.metric("Predicted BCS", result['predicted_class'])
                            with metric_col2:
                                st.metric("Confidence", f"{result['confidence']:.2%}")
                            
                            # Display validation info
                            if 'validation' in result:
                                validation = result['validation']
                                if validation.get('validator_active'):
                                    st.caption("✅ Image validated as cow rear view")
                            
                            # Display warning if low confidence
                            if result.get('warning'):
                                st.warning(f"⚠️ {result['warning']}")
                            
                            # Display probabilities
                            st.subheader("Class Probabilities")
                            probs = result['all_probabilities']
                            st.bar_chart(probs)
                    
                    elif response.status_code == 400:
                        # Validation error - not a cow rear image
                        error_data = response.json()
                        
                        with col2:
                            st.error("❌ Image Validation Failed")
                            st.warning(error_data.get('message', 'Unknown error'))
                            
                            if 'validation' in error_data:
                                validation = error_data['validation']
                                st.write("**Validation Details:**")
                                st.write(f"- Detected as: `{validation.get('detected_as', 'unknown')}`")
                                st.write(f"- Confidence: `{validation.get('confidence', 0):.2%}`")
                            
                            st.info("💡 **Tips:** Upload gambar yang menunjukkan tampilan belakang sapi dengan jelas untuk mendapatkan prediksi BCS yang akurat.")
                        
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error(f"Could not connect to API at {PREDICT_ENDPOINT}. Is it running?")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

st.markdown("---")
st.caption("Powered by YOLOv8 and FastAPI | With Image Validation 🔍")
