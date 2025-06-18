import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pytesseract
import tensorflow as tf
import joblib
from ultralytics import YOLO

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

        # --- Security Checks ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔢 Entropy Check")
            e = entropy(img)
            st.metric("Entropy Value", e)
            if e < 4.0:
                st.error("Low entropy — CAPTCHA may be too simple or artificially generated.")
                entropy_reason = "Low entropy: CAPTCHA is too simple or repetitive."
            else:
                st.success("Good entropy — Suggests real variation and complexity.")
                entropy_reason = "Good entropy: Sufficient variation and complexity."

            st.subheader("🌫️ Blur Detection")
            lap_var, is_blurry = detect_blur(img)
            st.metric("Laplacian Variance", f"{lap_var:.2f}")
            if is_blurry:
                st.warning("Blur detected — May indicate image obfuscation or attack.")
                blur_reason = "Blur detected: Image may be obfuscated or under attack."
            else:
                st.success("No significant blur detected.")
                blur_reason = "No significant blur detected."

            st.subheader("🌪️ Noise Detection")
            stddev, noisy = detect_noise(img)
            st.metric("Noise StdDev", f"{stddev:.2f}")
            if noisy:
                st.warning("High noise — Could be an adversarial CAPTCHA (salt & pepper noise).")
                noise_reason = "High noise: Possible adversarial CAPTCHA."
            else:
                st.success("Noise level is acceptable.")
                noise_reason = "Noise level is acceptable."

        with col2:
            st.subheader("🔁 Prediction Consistency")
            preds, consistent = prediction_consistency(img)
            st.write(f"Model Predictions (multiple runs): {preds}")
            if consistent:
                st.success("Predictions are consistent — Model is stable.")
                consistency_reason = "Predictions are consistent: Model is stable."
            else:
                st.error("Inconsistent predictions — CAPTCHA may be dynamic or model unstable.")
                consistency_reason = "Inconsistent predictions: CAPTCHA may be dynamic or model unstable."

            st.subheader("🧪 CAPTCHA Complexity")
            prediction = solve_with_model(img)
            complexity = analyze_captcha(prediction)
            st.json(complexity)
            if complexity["Overall Strength"] == "Weak":
                complexity_reason = "Weak CAPTCHA: Lacks complexity, easily solvable."
            elif complexity["Overall Strength"] == "Moderate":
                complexity_reason = "Moderate CAPTCHA: Some complexity, but may be vulnerable."
            else:
                complexity_reason = "Strong CAPTCHA: Good complexity."

            

        # --- Executive Summary ---
        st.markdown("---")
        st.header("📝 Executive Security Summary")
        reasons = [entropy_reason, blur_reason, noise_reason, consistency_reason, complexity_reason]
        if "Low entropy" in entropy_reason or "Weak CAPTCHA" in complexity_reason:
            verdict = "❌ CAPTCHA is Weak: " + " ".join([r for r in reasons if "Low entropy" in r or "Weak CAPTCHA" in r])
            recommendation = "Increase character variety, add more noise/distortion, and avoid simple patterns."
        elif "Moderate CAPTCHA" in complexity_reason or "High noise" in noise_reason:
            verdict = "⚠️ CAPTCHA is Moderate: " + " ".join([r for r in reasons if "Moderate CAPTCHA" in r or "High noise" in r])
            recommendation = "Consider increasing complexity and reducing noise."
        else:
            verdict = "✅ CAPTCHA is Strong: No major vulnerabilities detected."
            recommendation = "Maintain current security measures."

        st.subheader(verdict)
        st.info(f"**Recommendation:** {recommendation}")

    else:
        st.info("📎 Please upload a CAPTCHA image to proceed.")

# import streamlit as st
# import numpy as np
# import cv2
# from PIL import Image
# import pytesseract
# import tensorflow as tf
# import joblib
# from ultralytics import YOLO

# # --- PAGE CONFIG & THEME ---
# st.set_page_config(
#     page_title="AI CAPTCHA Solver Pro",
#     layout="wide",
#     page_icon="🤖"
# )

# # --- LOAD MODELS (CACHED) ---
# @st.cache_resource
# def load_models():
#     cnn_model = tf.keras.models.load_model("FINALMODELS/captchasolve.h5")
#     label_encoder = joblib.load("trained_label_encoder.pkl")
#     yolo_model = YOLO("yolo_model2/best.pt")
#     return cnn_model, label_encoder, yolo_model

# cnn_model, label_encoder, yolo_model = load_models()

# # --- UTILITY FUNCTIONS ---
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

# def entropy(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
#     hist /= hist.sum()
#     ent = -np.sum(hist * np.log2(hist + 1e-7))
#     return round(ent, 2)

# def detect_blur(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     var_lap = cv2.Laplacian(gray, cv2.CV_64F).var()
#     return var_lap, var_lap < 100

# def detect_noise(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     lap = cv2.Laplacian(gray, cv2.CV_64F)
#     mean, stddev = cv2.meanStdDev(lap)
#     return stddev[0][0], stddev[0][0] > 10

# def prediction_consistency(image, rounds=3):
#     results = [solve_with_model(image) for _ in range(rounds)]
#     unique = list(set(results))
#     return results, len(unique) == 1

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
#         "Overall Strength": strength
#     }

# # --- DASHBOARD LAYOUT ---
# st.markdown(
#     """
#     <style>
#     .block-container {padding-top:2rem;}
#     .stTabs [data-baseweb="tab-list"] {justify-content: center;}
#     </style>
#     """, unsafe_allow_html=True
# )

# st.image("https://img.icons8.com/ios-filled/100/robot-2.png", width=80)
# st.title("AI CAPTCHA Solver Pro")
# st.markdown("##### Built with YOLOv8 + CNN + Streamlit")
# st.markdown("---")

# tabs = st.tabs([
#     "🏠 Home",
#     "📝 OCR Solver",
#     "🧠 Model Solver",
#     "🔐 Security Analysis"
# ])

# # --- HOME TAB ---
# with tabs[0]:
#     col1, col2 = st.columns([1,2])
#     with col1:
#         st.image("https://img.icons8.com/ios-filled/500/robot-2.png", width=180)
#     with col2:
#         st.header("Welcome to AI CAPTCHA Solver Pro")
#         st.markdown("""
#         This dashboard combines advanced AI (YOLOv8 + CNN) and OCR to solve and analyze CAPTCHA images.

#         **Features:**
#         - 📝 OCR-based solver
#         - 🧠 Model-based solver (YOLO + CNN)
#         - 🔐 Security analysis with actionable insights

#         **How to use:**
#         1. Select a tab above.
#         2. Upload your CAPTCHA image.
#         3. Get solutions, analysis, and security recommendations.
#         """)
#     st.info("Try the Security Analysis tab for a full executive summary of your CAPTCHA's robustness.")

# # --- OCR SOLVER TAB ---
# with tabs[1]:
#     st.header("📝 OCR-Based CAPTCHA Solver")
#     uploaded_file_ocr = st.file_uploader("📤 Upload a CAPTCHA image", type=["png", "jpg", "jpeg"], key="ocr")
#     if uploaded_file_ocr:
#         file_bytes = np.asarray(bytearray(uploaded_file_ocr.read()), dtype=np.uint8)
#         img = cv2.imdecode(file_bytes, 1)
#         st.image(img, caption="Uploaded CAPTCHA", use_column_width=True)
#         if st.button("Solve with Tesseract OCR"):
#             text = solve_with_ocr(img)
#             st.success(f"🧾 OCR Result: `{text}`")
#             st.subheader("🧪 CAPTCHA Complexity Analysis")
#             st.json(analyze_captcha(text))
#     else:
#         st.info("📎 Please upload a CAPTCHA image to proceed.")

# # --- MODEL SOLVER TAB ---
# with tabs[2]:
#     st.header("🧠 Model-Based CAPTCHA Solver (YOLO + CNN)")
#     uploaded_file_model = st.file_uploader("📤 Upload a CAPTCHA image", type=["png", "jpg", "jpeg"], key="model")
#     if uploaded_file_model:
#         file_bytes = np.asarray(bytearray(uploaded_file_model.read()), dtype=np.uint8)
#         img = cv2.imdecode(file_bytes, 1)
#         st.image(img, caption="Uploaded CAPTCHA", use_column_width=True)
#         if st.button("Solve with Custom Model"):
#             prediction = solve_with_model(img)
#             st.success(f"🔍 Model Prediction: `{prediction}`")
#             st.subheader("🧪 CAPTCHA Complexity Analysis")
#             st.json(analyze_captcha(prediction))
#     else:
#         st.info("📎 Please upload a CAPTCHA image to proceed.")

# # --- SECURITY ANALYSIS TAB ---
# with tabs[3]:
#     st.header("🔐 Security Analysis Report")
#     uploaded_file_sec = st.file_uploader("📤 Upload a CAPTCHA image", type=["png", "jpg", "jpeg"], key="security")
#     if uploaded_file_sec:
#         file_bytes = np.asarray(bytearray(uploaded_file_sec.read()), dtype=np.uint8)
#         img = cv2.imdecode(file_bytes, 1)
#         st.image(img, caption="Uploaded CAPTCHA", use_column_width=True)

#         col1, col2 = st.columns(2)
#         with col1:
#             st.subheader("🔢 Entropy Check")
#             e = entropy(img)
#             st.metric("Entropy Value", e)
#             if e < 4.0:
#                 st.error("Low entropy — CAPTCHA may be too simple or artificially generated.")
#                 entropy_reason = "Low entropy: CAPTCHA is too simple or repetitive."
#             else:
#                 st.success("Good entropy — Suggests real variation and complexity.")
#                 entropy_reason = "Good entropy: Sufficient variation and complexity."

#             st.subheader("🌫️ Blur Detection")
#             lap_var, is_blurry = detect_blur(img)
#             st.metric("Laplacian Variance", f"{lap_var:.2f}")
#             if is_blurry:
#                 st.warning("Blur detected — May indicate image obfuscation or attack.")
#                 blur_reason = "Blur detected: Image may be obfuscated or under attack."
#             else:
#                 st.success("No significant blur detected.")
#                 blur_reason = "No significant blur detected."

#             st.subheader("🌪️ Noise Detection")
#             stddev, noisy = detect_noise(img)
#             st.metric("Noise StdDev", f"{stddev:.2f}")
#             if noisy:
#                 st.warning("High noise — Could be an adversarial CAPTCHA (salt & pepper noise).")
#                 noise_reason = "High noise: Possible adversarial CAPTCHA."
#             else:
#                 st.success("Noise level is acceptable.")
#                 noise_reason = "Noise level is acceptable."

#         with col2:
#             st.subheader("🔁 Prediction Consistency")
#             preds, consistent = prediction_consistency(img)
#             st.write(f"Model Predictions (multiple runs): {preds}")
#             if consistent:
#                 st.success("Predictions are consistent — Model is stable.")
#                 consistency_reason = "Predictions are consistent: Model is stable."
#             else:
#                 st.error("Inconsistent predictions — CAPTCHA may be dynamic or model unstable.")
#                 consistency_reason = "Inconsistent predictions: CAPTCHA may be dynamic or model unstable."

#             st.subheader("🧪 CAPTCHA Complexity")
#             prediction = solve_with_model(img)
#             complexity = analyze_captcha(prediction)
#             st.json(complexity)
#             if complexity["Overall Strength"] == "Weak":
#                 complexity_reason = "Weak CAPTCHA: Lacks complexity, easily solvable."
#             elif complexity["Overall Strength"] == "Moderate":
#                 complexity_reason = "Moderate CAPTCHA: Some complexity, but may be vulnerable."
#             else:
#                 complexity_reason = "Strong CAPTCHA: Good complexity."

#         # --- Executive Summary ---
#         st.markdown("---")
#         st.header("📝 Executive Security Summary")
#         reasons = [entropy_reason, blur_reason, noise_reason, consistency_reason, complexity_reason]
#         if "Low entropy" in entropy_reason or "Weak CAPTCHA" in complexity_reason:
#             verdict = "❌ CAPTCHA is Weak: " + " ".join([r for r in reasons if "Low entropy" in r or "Weak CAPTCHA" in r])
#             recommendation = "Increase character variety, add more noise/distortion, and avoid simple patterns."
#         elif "Moderate CAPTCHA" in complexity_reason or "High noise" in noise_reason:
#             verdict = "⚠️ CAPTCHA is Moderate: " + " ".join([r for r in reasons if "Moderate CAPTCHA" in r or "High noise" in r])
#             recommendation = "Consider increasing complexity and reducing noise."
#         else:
#             verdict = "✅ CAPTCHA is Strong: No major vulnerabilities detected."
#             recommendation = "Maintain current security measures."

#         st.subheader(verdict)
#         st.info(f"**Recommendation:** {recommendation}")

#     else:
#         st.info("📎 Please upload a CAPTCHA image to proceed.")
