import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Page Setting
st.set_page_config(page_title="AI Vision", layout="centered")

st.title("🖐️ AI Hand Detector")
st.write("Yeh application bina kisi API ke kaam karti hai.")

# Mediapipe setup (Pre-trained Model)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7)

# Camera Input UI
image_file = st.camera_input("Apne hath ki photo lein")

if image_file is not None:
    # Image ko process karne ke liye convert karein
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    
    # BGR to RGB (Mediapipe ke liye)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Agar hath mil jaye to us par points draw karein
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        st.success("Hath mil gaya!")
    else:
        st.warning("Hath nazar nahi aa raha, dobara koshish karein.")

    # Screen par dikhayein
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
