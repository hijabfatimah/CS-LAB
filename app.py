import streamlit as st
import numpy as np

# OpenCV aur Mediapipe ko safe tareeqay se import karein
try:
    import cv2
    import mediapipe as mp
except ImportError as e:
    st.error(f"Module load nahi ho saka: {e}")
    st.stop()

# --- UI Setup ---
st.set_page_config(page_title="AI Hand Tracker", layout="centered")
st.title("🖐️ AI Hand Landmark Detection")
st.write("Sirf 2 files ke sath chalti hui computer vision application.")

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# User Friendly Camera Input
img_file = st.camera_input("Apne hath ki photo lein")

if img_file is not None:
    # Convert image
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process
    results = hands.process(image_rgb)
    
    # Draw landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        st.success("Hath detect ho gaya!")
    else:
        st.warning("Hath nazar nahi aa raha.")
        
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
