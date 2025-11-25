import streamlit as st
import requests
from PIL import Image
import io
import os

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{API_URL}/predict"

st.set_page_config(page_title="BCS Cow Classifier", page_icon="🐄")

st.title("🐄 Dairy Cow BCS Classifier")
st.write("Upload an image of a cow to classify its Body Condition Score (BCS).")

# Sidebar for settings
st.sidebar.header("Settings")
api_url_input = st.sidebar.text_input("API URL", value=API_URL)
if api_url_input != API_URL:
    PREDICT_ENDPOINT = f"{api_url_input}/predict"

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Create columns for side-by-side layout
    col1, col2 = st.columns(2)
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width='stretch')
        
        # Button and Status in the left column (below image)
        if st.button("Predict", width='stretch'):
            with st.spinner("Classifying..."):
                try:
                    # Prepare file for API
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    
                    # Call API
                    response = requests.post(PREDICT_ENDPOINT, files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display status in left column
                        st.success("Classification Successful!")
                        
                        # Display results in right column
                        with col2:
                            st.write("### Classification Results")
                            metric_col1, metric_col2 = st.columns(2)
                            with metric_col1:
                                st.metric("Predicted BCS", result['predicted_class'])
                            with metric_col2:
                                st.metric("Confidence", f"{result['confidence']:.2%}")
                            
                            # Display probabilities
                            st.subheader("Class Probabilities")
                            probs = result['all_probabilities']
                            st.bar_chart(probs)
                        
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error(f"Could not connect to API at {PREDICT_ENDPOINT}. Is it running?")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

st.markdown("---")
st.caption("Powered by YOLOv8 and FastAPI")
