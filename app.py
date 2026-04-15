import streamlit as st
import requests
from PIL import Image
import io

# --- Page Setup ---
st.set_page_config(page_title="AI Object Detector", layout="centered")
st.title("🤖 AI Image Recognition")
st.write("Yeh app bina OpenCV aur bina Mediapipe ke chalti hai.")

# Hugging Face Free API (No payment required)
# Aap koi bhi free model use kar sakte hain
API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
# Aap apna free token yahan dal sakte hain ya isay public model ke liye use karein
headers = {"Authorization": "Bearer hf_xxxx"} # Optional: Agar limit hit ho to token dalen

def query(image_bytes):
    response = requests.post(API_URL, headers=headers, data=image_bytes)
    return response.json()

# --- UI / Camera ---
img_file = st.camera_input("Koi bhi cheez camera ke samne layein")

if img_file is not None:
    # Image display
    img = Image.open(img_file)
    st.image(img, caption="Aapki Photo", use_column_width=True)
    
    with st.spinner('AI soch raha hai...'):
        # Image ko bytes mein convert karein
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        byte_im = buf.getvalue()

        # AI Model se result lein
        output = query(byte_im)

    # Results dikhayein
    st.subheader("AI Result:")
    if isinstance(output, list) and len(output) > 0:
        for item in output:
            label = item.get('label', 'Unknown')
            score = item.get('score', 0)
            st.info(f"Mery mutabiq yeh **{label}** hai. (Confidence: {round(score*100, 2)}%)")
    else:
        st.error("AI samajh nahi saka. Dobara koshish karein.")

st.markdown("---")
st.caption("Note: Yeh application server-side error se bachne ke liye API use karti hai.")
