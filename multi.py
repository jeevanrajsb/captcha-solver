# import streamlit as st
# import numpy as np
# import cv2
# from PIL import Image
# import pytesseract
# import tensorflow as tf
# import joblib
# import hashlib
# from ultralytics import YOLO

# st.set_page_config(page_title="AI CAPTCHA Solver", layout="wide")


# # ------------------------------
# # Load models
# @st.cache_resource
# def load_models():
#     cnn_model = tf.keras.models.load_model("FINALMODELS/captchasolve.h5")
#     label_encoder = joblib.load("trained_label_encoder.pkl")
#     yolo_model = YOLO("yolo_model2/best.pt")
#     return cnn_model, label_encoder, yolo_model

# cnn_model, label_encoder, yolo_model = load_models()

# # ------------------------------
# # Preprocessing & Prediction
# def preprocess_letter(letter_crop):
#     resized = cv2.resize(letter_crop, (32, 52))
#     gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
#     return binary

# def check_letter_tarakom(pic):
#     _, pic = cv2.threshold(pic, 127, 255, cv2.THRESH_BINARY)
#     s = 90 - (np.sum(pic, axis=0, keepdims=True) / 255)
#     total = len(s[0])
#     howmanyblack = sum(1 for i in s[0] if np.sum(i) >= 175)
#     return total - howmanyblack <= 22

# def solve_with_model(image):
#     results = yolo_model(image)
#     detections = sorted(results[0].boxes, key=lambda box: box.xyxy[0][0])
#     letters, last_one = [], []
#     count = 0
#     for box in detections:
#         x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#         crop = image[y1:y2, x1:x2]
#         if not last_one or x1 - last_one[0] > 10:
#             processed = preprocess_letter(crop)
#             if count == 8 and check_letter_tarakom(crop):
#                 letters.append(processed)
#             elif count <= 7:
#                 letters.append(processed)
#             last_one = [x1, y1, x2, y2]
#             count += 1
#     predictions = []
#     for letter in letters:
#         input_data = letter.reshape(1, 52, 32, 1)
#         prediction = cnn_model.predict(input_data)
#         predicted_char = label_encoder.inverse_transform([prediction.argmax()])[0]
#         predictions.append(predicted_char)
#     return "".join(predictions)

# def solve_with_ocr(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, binarized = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
#     return pytesseract.image_to_string(binarized, config='--psm 8').strip()

# # ------------------------------
# # Security Analysis
# def entropy(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
#     hist /= hist.sum()
#     return -np.sum(hist * np.log2(hist + 1e-7))

# def detect_blur(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     return cv2.Laplacian(gray, cv2.CV_64F).var() < 100

# def detect_noise(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
#     noise_ratio = np.sum(binary == 0) / binary.size
#     return noise_ratio > 0.6

# def prediction_consistency(image, rounds=3):
#     results = [solve_with_model(image) for _ in range(rounds)]
#     return results, len(set(results)) == 1

# def analyze_captcha(text):
#     score = sum([
#         any(c.isdigit() for c in text),
#         any(c.isupper() for c in text),
#         any(c.islower() for c in text),
#         any(not c.isalnum() for c in text)
#     ])
#     if len(text) >= 6 and score >= 3:
#         strength = "Strong"
#     elif len(text) >= 4 and score >= 2:
#         strength = "Moderate"
#     else:
#         strength = "Weak"
#     return {
#         "Length": len(text),
#         "Contains Digits": any(c.isdigit() for c in text),
#         "Contains Uppercase": any(c.isupper() for c in text),
#         "Contains Lowercase": any(c.islower() for c in text),
#         "Contains Special Chars": any(not c.isalnum() for c in text),
#         "Strength": strength
#     }

# # ------------------------------
# # Multi-page Streamlit UI
# # st.set_page_config(page_title="AI CAPTCHA Solver", layout="wide")

# page = st.sidebar.selectbox("Navigate", ["Home", "OCR Solver", "Model Solver", "Security Analysis"])

# st.sidebar.markdown("---")
# st.sidebar.markdown("🔒 Built with AI, YOLO, CNN, and Streamlit")

# if page == "Home":
#     st.title("🤖 AI CAPTCHA Solver")
#     st.markdown("""
#     This professional-grade CAPTCHA solver uses:
    
#     - **YOLO** for character detection  
#     - **CNN** for character classification  
#     - **Tesseract OCR** for baseline comparison  
#     - **Security Analysis** for CAPTCHA robustness
    
#     Navigate using the sidebar to:
#     - Solve CAPTCHA with OCR or custom AI
#     - Run advanced security diagnostics
#     """)

# elif page in ["OCR Solver", "Model Solver", "Security Analysis"]:
#     uploaded_file = st.file_uploader("Upload CAPTCHA Image", type=["png", "jpg", "jpeg"])

#     if uploaded_file:
#         file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#         img = cv2.imdecode(file_bytes, 1)
#         st.image(img, caption="Uploaded CAPTCHA", use_column_width=True)

#         if page == "OCR Solver":
#             st.header("📄 Solve CAPTCHA using Tesseract OCR")
#             if st.button("Solve"):
#                 text = solve_with_ocr(img)
#                 st.success(f"OCR Prediction: `{text}`")
#                 st.write("🔍 Security Analysis")
#                 st.json(analyze_captcha(text))

#         elif page == "Model Solver":
#             st.header("🧠 Solve CAPTCHA using YOLO + CNN")
#             if st.button("Solve"):
#                 result = solve_with_model(img)
#                 st.success(f"Model Prediction: `{result}`")
#                 st.write("🔍 Security Analysis")
#                 st.json(analyze_captcha(result))

#         elif page == "Security Analysis":
#             st.header("🔐 CAPTCHA Security Analysis")

#             entropy_val = entropy(img)
#             st.info(f"Entropy: {entropy_val:.2f}")
#             if entropy_val < 4.0:
#                 st.warning("⚠️ Low entropy – CAPTCHA may be too uniform or manipulated.")

#             if detect_blur(img):
#                 st.error("Blurry image detected – Possible adversarial attack.")
#             else:
#                 st.success("No significant blur detected.")

#             if detect_noise(img):
#                 st.error("High noise detected – Possible salt & pepper attack.")
#             else:
#                 st.success("No significant noise detected.")

#             preds, consistent = prediction_consistency(img)
#             st.write(f"Model predictions: {preds}")
#             if consistent:
#                 st.success("Model is consistent across multiple runs.")
#             else:
#                 st.error("Inconsistencies detected – Potential instability or attack.")
#     else:
#         st.info("Upload an image to continue.")


import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pytesseract
import tensorflow as tf
import joblib
from ultralytics import YOLO

# Set page config first
st.set_page_config(page_title="AI CAPTCHA Dashboard", layout="wide")

# Load models (cached)
@st.cache_resource
def load_models():
    cnn_model = tf.keras.models.load_model("FINALMODELS/captchasolve.h5")
    label_encoder = joblib.load("trained_label_encoder.pkl")
    yolo_model = YOLO("yolo_model2/best.pt")
    return cnn_model, label_encoder, yolo_model

cnn_model, label_encoder, yolo_model = load_models()

# Preprocessing functions
def preprocess_letter(letter_crop):
    resized = cv2.resize(letter_crop, (32, 52))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return binary

def check_letter_tarakom(pic):
    _, pic = cv2.threshold(pic, 127, 255, cv2.THRESH_BINARY)
    s = 90 - (np.sum(pic, axis=0, keepdims=True) / 255)
    total = len(s[0])
    black = sum(1 for i in s[0] if np.sum(i) >= 175)
    return total - black <= 22

# Model-based CAPTCHA solver
def solve_with_model(image):
    results = yolo_model(image)
    detections = sorted(results[0].boxes, key=lambda box: box.xyxy[0][0])
    letters, last_one = [], []
    count = 0
    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        crop = image[y1:y2, x1:x2]
        if not last_one or x1 - last_one[0] > 10:
            processed = preprocess_letter(crop)
            if count == 8 and check_letter_tarakom(crop):
                letters.append(processed)
            elif count <= 7:
                letters.append(processed)
            last_one = [x1, y1, x2, y2]
            count += 1
    predictions = []
    for letter in letters:
        input_data = letter.reshape(1, 52, 32, 1)
        prediction = cnn_model.predict(input_data)
        predicted_char = label_encoder.inverse_transform([prediction.argmax()])[0]
        predictions.append(predicted_char)
    return "".join(predictions)

# OCR-based solver
def solve_with_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binarized = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return pytesseract.image_to_string(binarized, config='--psm 8').strip()

# Security metrics
def entropy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist /= hist.sum()
    return -np.sum(hist * np.log2(hist + 1e-7))

def detect_blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < 100

def detect_noise(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return np.sum(binary == 0) / binary.size > 0.6

def prediction_consistency(image, rounds=3):
    results = [solve_with_model(image) for _ in range(rounds)]
    return results, len(set(results)) == 1

def analyze_captcha(text):
    score = sum([
        any(c.isdigit() for c in text),
        any(c.isupper() for c in text),
        any(c.islower() for c in text),
        any(not c.isalnum() for c in text)
    ])
    if len(text) >= 6 and score >= 3:
        strength = "Strong"
    elif len(text) >= 4 and score >= 2:
        strength = "Moderate"
    else:
        strength = "Weak"
    return {
        "Length": len(text),
        "Contains Digits": any(c.isdigit() for c in text),
        "Contains Uppercase": any(c.isupper() for c in text),
        "Contains Lowercase": any(c.islower() for c in text),
        "Contains Special Chars": any(not c.isalnum() for c in text),
        "Strength": strength
    }

# Sidebar navigation
page = st.sidebar.radio("📋 Navigation", ["Dashboard", "OCR Solver", "Model Solver", "Security Analysis"])
st.sidebar.markdown("---")
st.sidebar.markdown("🔐 Built with YOLO + CNN + OCR")

# Page 1 - Home Dashboard
if page == "Dashboard":
    st.title("🤖 AI CAPTCHA Solver Dashboard")
    st.markdown("""
    Welcome to the **AI CAPTCHA Solver**!  
    This tool leverages:
    - 🧠 Deep Learning (CNN)
    - 🔍 Object Detection (YOLO)
    - 📜 OCR (Tesseract)
    - 🛡️ Security Analysis

    Use the sidebar to:
    - Compare OCR and AI predictions  
    - Evaluate CAPTCHA security using entropy, noise, blur, and model stability
    """)
    st.image("https://miro.medium.com/v2/resize:fit:1200/1*JksgSMiEODR5wWRNjzDWBw.gif", use_column_width=True)

# Shared Upload
elif page in ["OCR Solver", "Model Solver", "Security Analysis"]:
    uploaded_file = st.file_uploader("📤 Upload CAPTCHA Image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, caption="Uploaded CAPTCHA", use_column_width=True)

        if page == "OCR Solver":
            st.header("📄 Solve using Tesseract OCR")
            if st.button("Run OCR"):
                text = solve_with_ocr(img)
                st.success(f"Prediction: `{text}`")
                st.subheader("🔍 CAPTCHA Security Analysis")
                st.json(analyze_captcha(text))

        elif page == "Model Solver":
            st.header("🧠 Solve using YOLO + CNN")
            if st.button("Run Model"):
                result = solve_with_model(img)
                st.success(f"Prediction: `{result}`")
                st.subheader("🔍 CAPTCHA Security Analysis")
                st.json(analyze_captcha(result))

        elif page == "Security Analysis":
            st.header("🔐 CAPTCHA Security Analysis")

            entropy_val = entropy(img)
            st.metric("Entropy", f"{entropy_val:.2f}", delta=None)
            if entropy_val < 4.0:
                st.warning("⚠️ Low entropy – Image may be overly uniform.")
            else:
                st.success("✅ Entropy level is acceptable.")

            if detect_blur(img):
                st.error("🌀 Blur detected – Possible image distortion.")
            else:
                st.success("✅ No significant blur.")

            if detect_noise(img):
                st.error("🌪️ High noise detected – Possible adversarial artifact.")
            else:
                st.success("✅ Noise level acceptable.")

            st.subheader("🎯 Prediction Consistency Check")
            preds, stable = prediction_consistency(img)
            st.write("Predictions across runs:", preds)
            if stable:
                st.success("✅ Model is consistent.")
            else:
                st.error("❗ Inconsistent predictions – Possible instability or attack.")
    else:
        st.info("⬆️ Upload an image to begin.")

