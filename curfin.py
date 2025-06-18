import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pytesseract
import tensorflow as tf
import joblib
from ultralytics import YOLO
from skimage import exposure, filters

# --- PAGE CONFIG & THEME ---
st.set_page_config(
    page_title="AI CAPTCHA Solver Pro",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://img.icons8.com/ios-filled/100/robot-2.png", width=80)
st.sidebar.title("AI CAPTCHA Solver Pro")
page = st.sidebar.selectbox(
    "🔧 Navigation",
    ["Home", "OCR Solver", "Model Solver", "Security Analysis"],
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.markdown("🛡️ Built with YOLOv8 + CNN + Streamlit")

# --- LOAD MODELS (CACHED) ---
@st.cache_resource
def load_models():
    cnn_model = tf.keras.models.load_model("FINALMODELS/captchasolve.h5")
    label_encoder = joblib.load("trained_label_encoder.pkl")
    yolo_model = YOLO("yolo_model2/best.pt")
    return cnn_model, label_encoder, yolo_model

cnn_model, label_encoder, yolo_model = load_models()

# --- UTILITY FUNCTIONS ---
def preprocess_letter(letter_crop):
    resized = cv2.resize(letter_crop, (32, 52))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return binary

def check_letter_tarakom(pic):
    _, pic = cv2.threshold(pic, 127, 255, cv2.THRESH_BINARY)
    s = 90 - (np.sum(pic, axis=0, keepdims=True) / 255)
    total = len(s[0])
    howmanyblack = sum(1 for i in s[0] if np.sum(i) >= 175)
    return total - howmanyblack <= 22

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

def solve_with_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binarized = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return pytesseract.image_to_string(binarized, config='--psm 8').strip()

def entropy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist /= hist.sum()
    ent = -np.sum(hist * np.log2(hist + 1e-7))
    return round(ent, 2)

def detect_blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    var_lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    return var_lap, var_lap < 100

def detect_noise(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    mean, stddev = cv2.meanStdDev(lap)
    return stddev[0][0], stddev[0][0] > 10

def detect_contrast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = gray.std()
    return contrast, contrast < 30

def detect_brightness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness, brightness < 40 or brightness > 220

def edge_density(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    density = np.sum(edges > 0) / edges.size
    return round(density, 2), density < 0.05

def prediction_consistency(image, rounds=3):
    results = [solve_with_model(image) for _ in range(rounds)]
    unique = list(set(results))
    return results, len(unique) == 1

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
        "Overall Strength": strength
    }

# --- PAGE CONTENTS ---
if page == "Home":
    col1, col2 = st.columns([1,2])
    with col1:
        st.image("https://img.icons8.com/ios-filled/500/robot-2.png", width=180)
    with col2:
        st.title("🤖 AI CAPTCHA Solver Pro")
        st.markdown("""
        Welcome to the **AI CAPTCHA Solver Pro**!  
        This dashboard combines advanced AI (YOLOv8 + CNN) and OCR to solve and analyze CAPTCHA images.

        **Features:**
        - 📝 OCR-based solver
        - 🧠 Model-based solver (YOLO + CNN)
        - 🔐 Security analysis with actionable insights

        **How to use:**
        1. Select a page from the sidebar.
        2. Upload your CAPTCHA image.
        3. Get solutions, analysis, and security recommendations.
        """)
    st.markdown("---")
    st.info("Pro Tip: Try the Security Analysis page for a full executive summary of your CAPTCHA's robustness.")

elif page == "OCR Solver":
    st.header("📝 OCR-Based CAPTCHA Solver")
    uploaded_file = st.file_uploader("📤 Upload a CAPTCHA image", type=["png", "jpg", "jpeg"], key="ocr")
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, caption="Uploaded CAPTCHA", use_column_width=True)
        if st.button("Solve with Tesseract OCR"):
            text = solve_with_ocr(img)
            st.success(f"🧾 OCR Result: `{text}`")
            st.subheader("🧪 CAPTCHA Complexity Analysis")
            st.json(analyze_captcha(text))
    else:
        st.info("📎 Please upload a CAPTCHA image to proceed.")

elif page == "Model Solver":
    st.header("🧠 Model-Based CAPTCHA Solver (YOLO + CNN)")
    uploaded_file = st.file_uploader("📤 Upload a CAPTCHA image", type=["png", "jpg", "jpeg"], key="model")
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, caption="Uploaded CAPTCHA", use_column_width=True)
        if st.button("Solve with Custom Model"):
            prediction = solve_with_model(img)
            st.success(f"🔍 Model Prediction: `{prediction}`")
            st.subheader("🧪 CAPTCHA Complexity Analysis")
            st.json(analyze_captcha(prediction))
    else:
        st.info("📎 Please upload a CAPTCHA image to proceed.")

elif page == "Security Analysis":
    st.header("🔐 Security Analysis Report")
    uploaded_file = st.file_uploader("📤 Upload a CAPTCHA image", type=["png", "jpg", "jpeg"], key="security")
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, caption="Uploaded CAPTCHA", use_column_width=True)

        # --- Metric Computations ---
        e = entropy(img)
        blur_val, blur_flag = detect_blur(img)
        noise_stddev, noisy = detect_noise(img)
        contrast, low_contrast = detect_contrast(img)
        brightness, bad_brightness = detect_brightness(img)
        edensity, low_edges = edge_density(img)
        preds, consistent = prediction_consistency(img)
        prediction = solve_with_model(img)
        complexity = analyze_captcha(prediction)

        # --- Color Status Formatter ---
        def status_line(label, value, is_bad, reason, threshold_info):
            icon = "❌" if is_bad else "✅"
            color = "red" if is_bad else "green"
            st.markdown(
                f"<span style='color:{color}; font-size:16px'>{icon} <strong>{label}:</strong> {value:.2f} — {reason} ({threshold_info})</span>",
                unsafe_allow_html=True
            )

        st.subheader("📊 Metric-Based Analysis")

        status_line("Entropy", e, e < 4,
                    "Too low — image lacks randomness; easy to segment", "Ideal > 4")

        status_line("Laplacian Variance (Blur)", blur_val, blur_flag,
                    "Too low — blurry images are easier to OCR", "Ideal > 100")

        status_line("Noise StdDev", noise_stddev, noisy,
                    "Too high — noise can degrade human readability", "Ideal < 10")

        status_line("Contrast", contrast, low_contrast,
                    "Too low — makes character edges unclear", "Ideal > 30")

        status_line("Brightness", brightness, bad_brightness,
                    "Poor lighting — characters may be washed out or hidden", "Ideal 40–220")

        status_line("Edge Density", edensity, low_edges,
                    "Too few edges — makes segmentation easy", "Ideal > 0.05")

        # --- Prediction Consistency ---
        st.markdown("---")
        st.subheader("🧠 Prediction Consistency")
        st.write(f"Predictions from 3 model runs: `{preds}`")
        st.markdown(
            f"{'✅' if consistent else '❌'} <strong>Model Prediction:</strong> {'Consistent' if consistent else 'Inconsistent'}",
            unsafe_allow_html=True
        )

        # --- CAPTCHA Complexity ---
        st.subheader("🔤 CAPTCHA Complexity Evaluation")
        st.json(complexity)

        # --- Executive Summary ---
        st.markdown("---")
        st.header("📝 Executive Summary")
        issues = []
        if e < 4: issues.append("Low entropy — predictable pattern")
        if blur_flag: issues.append("Blur detected — unclear character edges")
        if noisy: issues.append("High noise — may degrade readability")
        if low_contrast: issues.append("Low contrast — poor distinction")
        if bad_brightness: issues.append("Brightness outside optimal range")
        if low_edges: issues.append("Low edge density — minimal texture")
        if not consistent: issues.append("Inconsistent predictions — unreliable decoding")
        if complexity['Overall Strength'] == "Weak": issues.append("Weak CAPTCHA complexity — simple characters")

        if issues:
            verdict = "❌ CAPTCHA is Weak or Moderate"
            recommendation = "Consider adding background noise, random distortions, edge complexity, and diverse fonts."
        else:
            verdict = "✅ CAPTCHA is Strong"
            recommendation = "Current configuration is effective. Maintain randomness, clarity, and distortion balance."

        st.subheader(verdict)
        st.info(f"**Recommendation:** {recommendation}")

        if issues:
            st.markdown("### 🔍 Issues Identified:")
            for issue in issues:
                st.markdown(f"- ❌ {issue}")
        else:
            st.markdown("### ✅ No major security weaknesses detected.")

    else:
        st.info("📎 Please upload a CAPTCHA image to proceed.")

