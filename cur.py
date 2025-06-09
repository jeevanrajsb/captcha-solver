import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pytesseract
from ultralytics import YOLO
import joblib
import os
import re
from PIL import Image

# Load models and encoders
cnn_model = tf.keras.models.load_model("FINALMODELS/captchasolve.h5")
label_encoder = joblib.load('trained_label_encoder.pkl')
yolo_model = YOLO("yolo_model2/best.pt")

# Helper functions
def check_letter_tarakom(pic):
    _, pic = cv2.threshold(pic, 127, 255, cv2.THRESH_BINARY)
    s = 90 - (np.sum(pic, axis=0, keepdims=True) / 255)
    total = len(s[0])
    howmanyblack = sum(1 for i in s[0] if np.sum(i) >= 175)
    return total - howmanyblack <= 22

def preprocess_letter(letter_crop):
    resized_letter = cv2.resize(letter_crop, (32, 52))
    resized_letter = cv2.cvtColor(resized_letter, cv2.COLOR_BGR2GRAY)
    _, binarized = cv2.threshold(resized_letter, 128, 255, cv2.THRESH_BINARY)
    return resized_letter

def solve_with_model(image):
    results = yolo_model(image)
    detections = sorted(results[0].boxes, key=lambda box: box.xyxy[0][0])

    letters = []
    last_one = []
    count = 0

    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        crop = image[y1:y2, x1:x2]

        if last_one:
            if int(x1) - last_one[0] > 10:
                pre = preprocess_letter(crop)
                if count == 8:
                    if check_letter_tarakom(crop):
                        letters.append(pre)
                elif count <= 7:
                    letters.append(pre)
                last_one = [x1, y1, x2, y2]
                count += 1
        else:
            pre = preprocess_letter(crop)
            if check_letter_tarakom(crop):
                letters.append(pre)
            last_one = [x1, y1, x2, y2]
            count += 1

    allpredicted = []
    for letter in letters:
        sample_image = letter.reshape(1, 52, 32, 1)
        prediction = cnn_model.predict(sample_image)
        predicted_class = label_encoder.inverse_transform([prediction.argmax()])
        allpredicted.append(predicted_class[0])

    return "".join(allpredicted)

def solve_with_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    text = pytesseract.image_to_string(thresh, config='--psm 8')
    return text.strip()

# Security Analysis
def analyze_captcha(captcha_text):
    analysis = {
        "length": len(captcha_text),
        "contains_digits": any(c.isdigit() for c in captcha_text),
        "contains_uppercase": any(c.isupper() for c in captcha_text),
        "contains_lowercase": any(c.islower() for c in captcha_text),
        "contains_special": any(not c.isalnum() for c in captcha_text),
        "strength": "Weak"
    }

    score = sum([analysis["contains_digits"], analysis["contains_uppercase"],
                 analysis["contains_lowercase"], analysis["contains_special"]])
    
    if analysis["length"] >= 6 and score >= 3:
        analysis["strength"] = "Strong"
    elif analysis["length"] >= 4 and score >= 2:
        analysis["strength"] = "Moderate"
    
    return analysis

# Streamlit UI
st.set_page_config(page_title="CAPTCHA Solver & Analyzer", layout="wide")
st.title("🔐 AI CAPTCHA Solver with Security Analysis")

uploaded_file = st.file_uploader("Upload CAPTCHA Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, caption="Uploaded CAPTCHA", use_column_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧠 Solve with Model"):
            try:
                prediction = solve_with_model(img)
                st.success(f"Predicted CAPTCHA: {prediction}")
                sec = analyze_captcha(prediction)
                st.write("🔍 Security Analysis:")
                st.json(sec)
            except Exception as e:
                st.error(f"Model failed: {e}")
    with col2:
        if st.button("🧾 Solve with OCR"):
            try:
                text = solve_with_ocr(img)
                st.warning(f"OCR Result: {text}")
                sec = analyze_captcha(text)
                st.write("🔍 Security Analysis:")
                st.json(sec)
            except Exception as e:
                st.error(f"OCR failed: {e}")
