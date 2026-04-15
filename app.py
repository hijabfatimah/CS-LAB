import streamlit as st
import requests
from PIL import Image
import io

# --- Page Setup ---
st.set_page_config(page_title="AI Object Detector", layout="centered")
st.title("🤖 AI Image Recognition")
st.write("Gallery se image upload karein ya Camera use karein.")

# Hugging Face Free API URL
API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
headers = {"Authorization": "Bearer hf_xxxx"} 

def query(image_bytes):
    response = requests.post(API_URL, headers=headers, data=image_bytes)
    return response.json()

# --- UI Tabs ---
tab1, tab2 = st.tabs(["📤 Image Upload", "📸 Camera Input"])

img_file = None

with tab1:
    uploaded_file = st.file_uploader("Image select karein...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img_file = uploaded_file

with tab2:
    camera_file = st.camera_input("Photo khainchein")
    if camera_file is not None:
        img_file = camera_file

# --- AI Logic ---
if img_file is not None:
    img = Image.open(img_file)
    
    # FIX: RGBA (PNG) ko RGB mein convert karein taake JPEG save karte waqt error na aaye
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    
    st.image(img, caption="Aapki Select karda Photo", use_container_width=True)
    
    with st.spinner('AI analysis kar raha hai...'):
        buf = io.BytesIO()
        # Ab yeh safely save ho jayega
        img.save(buf, format="JPEG")
        byte_im = buf.getvalue()

        output = query(byte_im)

    st.subheader("AI Prediction Result:")
    
    if isinstance(output, list) and len(output) > 0:
        # Agar API error message de (e.g. model loading)
        if "error" in output:
             st.error(f"API Error: {output['error']}")
        else:
            for item in output:
                label = item.get('label', 'Unknown')
                score = item.get('score', 0)
                st.write(f"**{label.capitalize()}**")
                st.progress(score)
                st.caption(f"Confidence: {round(score*100, 2)}%")
    elif isinstance(output, dict) and "error" in output:
        st.warning("Model load ho raha hai, 20-30 seconds baad dobara try karein.")
    else:
        st.error("Result fetch nahi ho saka.")

st.markdown("---")
st.caption("Note: Yeh application Hugging Face API ka istemal karti hai.")
