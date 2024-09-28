import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import tempfile

# Streamlit app title
st.title("Garbage Classification using AI")

# Streamlit sidebar for image upload
st.sidebar.title("Upload your image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# API credentials and URL
API_URL = "https://detect.roboflow.com"
API_KEY = "4rGsRFTwYiQvCzUsJgUH"
MODEL_ID = "garbage-classification-3/2"

# Initialize the InferenceHTTPClient
client = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Save the image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file, format="JPEG")
        temp_file_path = temp_file.name
    
    # Send the image to the inference API
    result = client.infer(temp_file_path, model_id=MODEL_ID)
    
    # Display the results
    st.write("Inference Results:")
    st.json(result)