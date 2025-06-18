# import streamlit as st
# import cv2
# import numpy as np
# import tensorflow as tf
# import pytesseract
# from ultralytics import YOLO
# import joblib
# import os
# import re
# from PIL import Image

# # Load models and encoders
# cnn_model = tf.keras.models.load_model("FINALMODELS/captchasolve.h5")
# label_encoder = joblib.load('trained_label_encoder.pkl')
# yolo_model = YOLO("yolo_model2/best.pt")

# # Helper functions
# def check_letter_tarakom(pic):
#     _, pic = cv2.threshold(pic, 127, 255, cv2.THRESH_BINARY)
#     s = 90 - (np.sum(pic, axis=0, keepdims=True) / 255)
#     total = len(s[0])
#     howmanyblack = sum(1 for i in s[0] if np.sum(i) >= 175)
#     return total - howmanyblack <= 22

# def preprocess_letter(letter_crop):
#     resized_letter = cv2.resize(letter_crop, (32, 52))
#     resized_letter = cv2.cvtColor(resized_letter, cv2.COLOR_BGR2GRAY)
#     _, binarized = cv2.threshold(resized_letter, 128, 255, cv2.THRESH_BINARY)
#     return resized_letter

# def solve_with_model(image):
#     results = yolo_model(image)
#     detections = sorted(results[0].boxes, key=lambda box: box.xyxy[0][0])

#     letters = []
#     last_one = []
#     count = 0

#     for box in detections:
#         x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#         crop = image[y1:y2, x1:x2]

#         if last_one:
#             if int(x1) - last_one[0] > 10:
#                 pre = preprocess_letter(crop)
#                 if count == 8:
#                     if check_letter_tarakom(crop):
#                         letters.append(pre)
#                 elif count <= 7:
#                     letters.append(pre)
#                 last_one = [x1, y1, x2, y2]
#                 count += 1
#         else:
#             pre = preprocess_letter(crop)
#             if check_letter_tarakom(crop):
#                 letters.append(pre)
#             last_one = [x1, y1, x2, y2]
#             count += 1

#     allpredicted = []
#     for letter in letters:
#         sample_image = letter.reshape(1, 52, 32, 1)
#         prediction = cnn_model.predict(sample_image)
#         predicted_class = label_encoder.inverse_transform([prediction.argmax()])
#         allpredicted.append(predicted_class[0])

#     return "".join(allpredicted)

# def solve_with_ocr(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
#     text = pytesseract.image_to_string(thresh, config='--psm 8')
#     return text.strip()

# # Security Analysis
# def analyze_captcha(captcha_text):
#     analysis = {
#         "length": len(captcha_text),
#         "contains_digits": any(c.isdigit() for c in captcha_text),
#         "contains_uppercase": any(c.isupper() for c in captcha_text),
#         "contains_lowercase": any(c.islower() for c in captcha_text),
#         "contains_special": any(not c.isalnum() for c in captcha_text),
#         "strength": "Weak"
#     }

#     score = sum([analysis["contains_digits"], analysis["contains_uppercase"],
#                  analysis["contains_lowercase"], analysis["contains_special"]])
    
#     if analysis["length"] >= 6 and score >= 3:
#         analysis["strength"] = "Strong"
#     elif analysis["length"] >= 4 and score >= 2:
#         analysis["strength"] = "Moderate"
    
#     return analysis

# # Streamlit UI
# st.set_page_config(page_title="CAPTCHA Solver & Analyzer", layout="wide")
# st.title("🔐 AI CAPTCHA Solver with Security Analysis")

# uploaded_file = st.file_uploader("Upload CAPTCHA Image", type=["png", "jpg", "jpeg"])

# if uploaded_file is not None:
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     img = cv2.imdecode(file_bytes, 1)
#     st.image(img, caption="Uploaded CAPTCHA", use_column_width=True)

#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("🧠 Solve with Model"):
#             try:
#                 prediction = solve_with_model(img)
#                 st.success(f"Predicted CAPTCHA: {prediction}")
#                 sec = analyze_captcha(prediction)
#                 st.write("🔍 Security Analysis:")
#                 st.json(sec)
#             except Exception as e:
#                 st.error(f"Model failed: {e}")
#     with col2:
#         if st.button("🧾 Solve with OCR"):
#             try:
#                 text = solve_with_ocr(img)
#                 st.warning(f"OCR Result: {text}")
#                 sec = analyze_captcha(text)
#                 st.write("🔍 Security Analysis:")
#                 st.json(sec)
#             except Exception as e:
#                 st.error(f"OCR failed: {e}")


import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pytesseract
from ultralytics import YOLO
import joblib
import hashlib
import io
from PIL import Image

# Load models and encoders
cnn_model = tf.keras.models.load_model("FINALMODELS/captchasolve.h5")
label_encoder = joblib.load('trained_label_encoder.pkl')
yolo_model = YOLO("yolo_model2/best.pt")

# --- CAPTCHA solving functions ---

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
                if count == 8 and check_letter_tarakom(crop):
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

# --- Security analysis functions ---

BLACKLISTED_HASHES = {
    "d41d8cd98f00b204e9800998ecf8427e"  # Placeholder
}

def get_image_hash(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()

def is_blacklisted(image_bytes):
    return get_image_hash(image_bytes) in BLACKLISTED_HASHES

def analyze_entropy(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-7))
    return entropy

def detect_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def detect_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noise_level = np.std(gray)
    return noise_level

def check_model_consistency(image, times=3):
    results = [solve_with_model(image) for _ in range(times)]
    return len(set(results)) == 1, results

def analyze_captcha_strength(captcha_text):
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

def calculate_risk_score(entropy, blur, noise, consistent, blacklisted):
    score = 0
    if entropy < 4: score += 2
    if blur < 100: score += 2
    if noise < 20: score += 1
    if not consistent: score += 3
    if blacklisted: score += 5

    if score >= 7:
        return "High", score
    elif score >= 4:
        return "Moderate", score
    else:
        return "Low", score

# --- Streamlit UI ---

st.set_page_config(page_title="CAPTCHA Solver & Analyzer", layout="wide")
st.title("🔐 AI CAPTCHA Solver with Advanced Security Analysis")

uploaded_file = st.file_uploader("Upload CAPTCHA Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    img_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    st.image(img, caption="Uploaded CAPTCHA", use_column_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧠 Solve with Model"):
            try:
                prediction = solve_with_model(img)
                st.success(f"Predicted CAPTCHA: {prediction}")
                strength = analyze_captcha_strength(prediction)
                st.write("🔐 CAPTCHA Strength:")
                st.json(strength)
            except Exception as e:
                st.error(f"Model Error: {e}")
    with col2:
        if st.button("🧾 Solve with OCR"):
            try:
                text = solve_with_ocr(img)
                st.warning(f"OCR Result: {text}")
                strength = analyze_captcha_strength(text)
                st.write("🔐 CAPTCHA Strength:")
                st.json(strength)
            except Exception as e:
                st.error(f"OCR Error: {e}")

    st.markdown("---")
    st.subheader("🛡️ Security Analysis")

    entropy_val = analyze_entropy(img)
    blur_val = detect_blur(img)
    noise_val = detect_noise(img)
    consistent, results = check_model_consistency(img)
    blacklisted = is_blacklisted(file_bytes)
    risk_level, risk_score = calculate_risk_score(entropy_val, blur_val, noise_val, consistent, blacklisted)

    st.write(f"📊 **Entropy**: {entropy_val:.2f} {'⚠️ Low' if entropy_val < 4 else '✅ OK'}")
    st.write(f"🔍 **Blur (Laplacian Var)**: {blur_val:.2f} {'⚠️ Blurry' if blur_val < 100 else '✅ Sharp'}")
    st.write(f"📉 **Noise (std dev)**: {noise_val:.2f} {'⚠️ Low noise' if noise_val < 20 else '✅ Noisy'}")
    st.write(f"🔁 **Model Consistency**: {'✅ Consistent' if consistent else '⚠️ Inconsistent'} → {results}")
    st.write(f"🗂️ **Blacklist Check**: {'❌ Blacklisted' if blacklisted else '✅ Safe'}")
    st.write(f"🚨 **Overall Risk Level**: **{risk_level}** (Score: {risk_score})")
else:
    st.info("📤 Upload a CAPTCHA image to begin analysis.")
