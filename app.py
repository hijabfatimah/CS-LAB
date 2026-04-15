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
# Yahan 'hf_xxxx' ki jagah apna Hugging Face token dalna behtar hai agar bar bar error aaye
headers = {"Authorization": "Bearer hf_xxxx"} 

def query(image_bytes):
    response = requests.post(API_URL, headers=headers, data=image_bytes)
    # Check karein ke response theek hai ya nahi
    if response.status_code == 200:
        try:
            return response.json()
        except Exception:
            return {"error": "JSON parse nahi ho saka."}
    else:
        return {"error": f"API Server ne error diya: {response.status_code}", "detail": response.text}

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
    
    # RGBA to RGB conversion
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    
    st.image(img, caption="Aapki Photo", use_container_width=True)
    
    with st.spinner('AI analysis kar raha hai...'):
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        byte_im = buf.getvalue()

        output = query(byte_im)

    st.subheader("AI Prediction Result:")
    
    # Error Handling Logic
    if isinstance(output, list) and len(output) > 0:
        for item in output:
            label = item.get('label', 'Unknown')
            score = item.get('score', 0)
            st.write(f"**{label.capitalize()}**")
            st.progress(score)
            st.caption(f"Confidence: {round(score*100, 2)}%")
            
    elif isinstance(output, dict) and "error" in output:
        # Agar model pehli baar load ho raha ho
        if "estimated_time" in output:
            st.warning(f"Model load ho raha hai. Taqreeban {round(output['estimated_time'], 1)} seconds mein tayyar ho jayega. Dobara try karein.")
        else:
            st.error(f"Masla aya: {output['error']}")
    else:
        st.error("API se sahi response nahi mila. Shayad internet ya API key ka masla hai.")

st.markdown("---")
st.caption("Note: Yeh application Hugging Face API ka istemal karti hai.")
