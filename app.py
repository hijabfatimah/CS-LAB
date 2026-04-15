import streamlit as st
import requests
from PIL import Image
import io

# --- Page Setup ---
st.set_page_config(page_title="AI Vision Pro", layout="centered")
st.title("🤖 Intelligent Image Classifier")
st.write("Gallery se upload karein ya Camera use karein. (No API Key Required)")

# --- Stable Model URL ---
# Agar ye model bhi 404 de, to iska matlab Hugging Face ka server temporary down hai.
API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
headers = {"Authorization": "Bearer hf_xxxx"} 

def query(image_bytes):
    try:
        response = requests.post(API_URL, headers=headers, data=image_bytes)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            return {"error": "Model load ho raha hai...", "loading": True}
        else:
            return {"error": f"API ne error diya: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

# --- UI Layout ---
tab1, tab2 = st.tabs(["📤 Upload Image", "📸 Camera Input"])

img_file = None

with tab1:
    uploaded_file = st.file_uploader("Image select karein (PNG, JPG)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img_file = uploaded_file

with tab2:
    camera_file = st.camera_input("Photo khainchein")
    if camera_file:
        img_file = camera_file

# --- Processing ---
if img_file is not None:
    img = Image.open(img_file)
    
    # PNG transparency fix
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    
    st.image(img, caption="Processing...", use_container_width=True)
    
    with st.spinner('AI analysis kar raha hai...'):
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        output = query(byte_im)

    st.subheader("AI Prediction Result:")
    
    if isinstance(output, list) and len(output) > 0:
        for item in output:
            label = item.get('label', 'Unknown')
            score = item.get('score', 0)
            st.write(f"**{label.split(',')[0].capitalize()}**") # Label ko clean kiya
            st.progress(float(score))
            st.caption(f"Confidence: {round(score*100, 1)}%")
            
    elif isinstance(output, dict) and "error" in output:
        if output.get("loading"):
            st.warning("Model pehli bar load ho raha hai. 15-20 seconds baad Page Refresh karein.")
        else:
            st.error(output["error"])
    else:
        st.error("API se sahi response nahi mila. Dobara try karein.")

st.markdown("---")
st.caption("Note: Yeh app Google ViT model use karti hai.")
