import streamlit as st
import numpy as np

# Error handling for missing modules
try:
    import cv2
    import mediapipe as mp
except ImportError:
    st.error("Libraries install ho rahi hain ya missing hain. Please 'Reboot App' par click karein.")
    st.info("Ensure requirements.txt has 'opencv-python-headless'")
    st.stop()

# --- App UI ---
st.set_page_config(page_title="AI Vision App", layout="centered")
st.title("🤖 Free AI Computer Vision")
st.caption("No API | No Payment | Open Source")

# Mediapipe Logic
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Camera Input
img_file = st.camera_input("Take a photo of your hand")

if img_file:
    # Processing
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    # Drawing
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        st.success("Hand Detected!")
    else:
        st.warning("No hand detected. Try again.")
        
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
