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
headers = {"Authorization": "Bearer hf_xxxx"} # Agar token ho to yahan dalen, warna default chalta hai

def query(image_bytes):
    response = requests.post(API_URL, headers=headers, data=image_bytes)
    return response.json()

# --- UI Tabs for Input Options ---
tab1, tab2 = st.tabs(["📤 Image Upload", "📸 Camera Input"])

img_file = None

with tab1:
    uploaded_file = st.file_uploader("Apni image select karein...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img_file = uploaded_file

with tab2:
    camera_file = st.camera_input("Photo khainchein")
    if camera_file is not None:
        img_file = camera_file

# --- AI Logic ---
if img_file is not None:
    # Image display
    img = Image.open(img_file)
    st.image(img, caption="Aapki Select karda Photo", use_container_width=True)
    
    with st.spinner('AI analysis kar raha hai...'):
        # Image ko bytes mein convert karein
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        byte_im = buf.getvalue()

        # AI Model se result lein
        output = query(byte_im)

    # Results dikhayein
    st.subheader("AI Prediction Result:")
    
    # Check if output is valid and not an error
    if isinstance(output, list) and len(output) > 0:
        for item in output:
            label = item.get('label', 'Unknown')
            score = item.get('score', 0)
            
            # Progress bar for visual appeal
            st.write(f"**{label.capitalize()}**")
            st.progress(score)
            st.caption(f"Confidence: {round(score*100, 2)}%")
    else:
        st.error("AI results fetch nahi kar saka. Shayad API limit hit ho gayi ho ya internet ka masla ho.")

st.markdown("---")
st.caption("Note: Yeh application Hugging Face API ka istemal karti hai.")
