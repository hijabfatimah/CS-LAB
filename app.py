import streamlit as st
import requests
from PIL import Image
import io

# --- Page Setup ---
st.set_page_config(page_title="AI Object Detector", layout="centered")
st.title("🤖 AI Image Recognition")
st.write("Gallery se image upload karein ya Camera use karein.")

# --- Model URL ---
API_URL = "https://api-inference.huggingface.co/models/microsoft/resnet-50"
headers = {"Authorization": "Bearer hf_xxxx"} 

def query(image_bytes):
    try:
        response = requests.post(API_URL, headers=headers, data=image_bytes)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            return {"error": "Model load ho raha hai (503).", "loading": True}
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

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
    
    # RGBA to RGB conversion (Important for JPEG)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    
    st.image(img, caption="Selected Photo", use_container_width=True)
    
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
            # Yahan fix kiya gaya hai
            st.write(f"**{label.capitalize()}**")
            st.progress(float(score))
            st.caption(f"Confidence: {round(score*100, 2)}%")
            
    elif isinstance(output, dict) and "error" in output:
        if output.get("loading"):
            st.warning("Model pehli bar load ho raha hai. 20 seconds baad dubara koshish karein.")
        else:
            st.error(output["error"])
    else:
        st.error("Response fetch nahi ho saka. Internet check karein.")

st.markdown("---")
st.caption("Note: Yeh application Microsoft ResNet model istemal karti hai.")
