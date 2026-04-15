import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image

# --- Page Setup ---
st.set_page_config(page_title="AI Hand Tracker", layout="centered")
st.title("🖐️ AI Hand Landmark Detector")
st.write("Yeh version bina OpenCV ke chalti hai, taake koi error na aaye.")

# --- Mediapipe Settings ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Model load karein (Free aur Pre-trained)
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=2, 
    min_detection_confidence=0.5
)

# --- UI / Camera ---
img_file = st.camera_input("Apne hath ki photo khainchein")

if img_file is not None:
    # Image ko PIL ke zariye open karein
    img = Image.open(img_file)
    # Numpy array mein badlein (Mediapipe ke liye)
    img_array = np.array(img)
    
    # AI Model Process
    results = hands.process(img_array)

    # Drawing (Hum direct image par draw karenge)
    if results.multi_hand_landmarks:
        # Landmarks draw karne ke liye humein image ko editable banana hoga
        annotated_image = img_array.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )
        st.success(f"Detected {len(results.multi_hand_landmarks)} hand(s)!")
        st.image(annotated_image, caption="AI Result", use_column_width=True)
    else:
        st.warning("Hath detect nahi hua. Dobara koshish karein.")
        st.image(img, caption="Original Image", use_column_width=True)

st.info("Note: Agar ab bhi error aaye, to 'Manage App' mein ja kar 'Reboot' lazmi karein.")
