import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="AI Computer Vision App", layout="wide")

st.title("🤖 AI Computer Vision: Hand Tracker")
st.markdown("""
Aap is application ke zariye real-time mein hand landmarks detect kar sakte hain. 
Yeh model **Mediapipe** par mabni hai aur is mein koi API key ki zaroorat nahi.
""")

# --- Sidebar Settings ---
st.sidebar.header("Settings")
detection_confidence = st.sidebar.slider("Min Detection Confidence", 0.0, 1.0, 0.5)
tracking_confidence = st.sidebar.slider("Min Tracking Confidence", 0.0, 1.0, 0.5)

# --- Initialize Mediapipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=detection_confidence,
    min_tracking_confidence=tracking_confidence
)

# --- Camera Input ---
img_file_buffer = st.camera_input("Apna hath camera ke samne layein")

if img_file_buffer is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Image ko RGB mein convert karna zaroori hai (Mediapipe RGB use karta hai)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Agar hath detect ho jayein to landmarks draw karein
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )
        
        st.success(f"Detected {len(results.multi_hand_landmarks)} hand(s)!")
    else:
        st.warning("Koi hath detect nahi hua.")

    # Final Image Display
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

st.sidebar.info("Yeh app Streamlit aur Mediapipe ke sath banayi gayi hai.")
