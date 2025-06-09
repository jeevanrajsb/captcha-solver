import streamlit as st
import cv2
import numpy as np
from PIL import Image
import hashlib
import io
import tensorflow as tf
from ultralytics import YOLO
import joblib

# --- Load models and encoders once ---
@st.cache_resource
def load_models():
    model = tf.keras.models.load_model("FINALMODELS/captchasolve.h5")
    label_encoder = joblib.load('trained_label_encoder.pkl')
    yolo_model = YOLO("yolo_model2/best.pt")
    return model, label_encoder, yolo_model

model, label_encoder, yolo_model = load_models()

# --- Utility functions ---

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
    return binarized

def solve_captcha(image):
    # Detect letters with YOLO
    results = yolo_model(image)
    detections = sorted(results[0].boxes, key=lambda box: box.xyxy[0][0])

    letters = []
    last_one = []
    howmany = 0

    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        letter_crop = image[y1:y2, x1:x2]

        if last_one:
            if int(x1) - last_one[0] > 10:
                resized_letter = preprocess_letter(letter_crop)
                if howmany == 8:
                    if check_letter_tarakom(letter_crop):
                        letters.append(resized_letter)
                elif howmany <= 7:
                    letters.append(resized_letter)
                last_one = [x1, y1, x2, y2]
                howmany += 1
        else:
            resized_letter = preprocess_letter(letter_crop)
            if check_letter_tarakom(letter_crop):
                letters.append(resized_letter)
            last_one = [x1, y1, x2, y2]
            howmany += 1

    predicted_chars = []
    for letter in letters:
        sample = letter.reshape(1, 52, 32, 1)
        pred = model.predict(sample)
        pred_class = label_encoder.inverse_transform([pred.argmax()])[0]
        predicted_chars.append(pred_class)

    return "".join(predicted_chars)

# --- Security analysis functions ---

def check_file_size(uploaded_file, max_mb=5):
    return uploaded_file.size < max_mb * 1024 * 1024

def is_valid_image(file_bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
        return True
    except:
        return False

def analyze_image_entropy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-7))
    return entropy

def detect_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < 100  # low variance = blur

def detect_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    noise_ratio = np.sum(binary == 0) / binary.size
    return noise_ratio > 0.6

BLACKLISTED_HASHES = {
    "d41d8cd98f00b204e9800998ecf8427e"  # Add real known bad hashes here
}

def get_image_hash(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()

def is_blacklisted(image_bytes):
    return get_image_hash(image_bytes) in BLACKLISTED_HASHES

def check_model_consistency(image, repeat=2):
    preds = []
    for _ in range(repeat):
        preds.append(solve_captcha(image))
    return len(set(preds)) == 1

# --- Streamlit UI ---

st.title("CAPTCHA Solver with Security Analysis")

uploaded_file = st.file_uploader("Upload CAPTCHA Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    file_bytes = uploaded_file.read()

    # Basic file validations
    if not check_file_size(uploaded_file):
        st.error("File size exceeds 5MB limit.")
    elif not is_valid_image(file_bytes):
        st.error("Uploaded file is not a valid image.")
    else:
        img = np.array(Image.open(io.BytesIO(file_bytes)).convert('RGB'))

        st.image(img, caption="Uploaded CAPTCHA", use_column_width=True)

        # Solve CAPTCHA
        with st.spinner("Solving CAPTCHA..."):
            try:
                captcha_text = solve_captcha(img)
                st.success(f"Predicted CAPTCHA: **{captcha_text}**")
            except Exception as e:
                st.error(f"Error solving CAPTCHA: {e}")

        # Security Analysis options
        st.header("Security Analysis Options")
        run_entropy = st.checkbox("Check Image Entropy")
        run_blur = st.checkbox("Check for Blur (Adversarial Attack)")
        run_noise = st.checkbox("Check for Noise (Salt & Pepper)")
        run_blacklist = st.checkbox("Check if Image is Blacklisted")
        run_model_consistency = st.checkbox("Check Model Prediction Consistency")

        st.subheader("Security Analysis Results")

        if run_entropy:
            entropy = analyze_image_entropy(img)
            st.info(f"Image Entropy: {entropy:.2f}")
            if entropy < 4.0:
                st.warning("Low entropy detected - image may be too uniform or tampered.")

        if run_blur:
            if detect_blur(img):
                st.warning("Blurry image detected - potential adversarial tampering.")
            else:
                st.success("No significant blur detected.")

        if run_noise:
            if detect_noise(img):
                st.warning("High noise detected - possible salt & pepper noise attack.")
            else:
                st.success("No significant noise detected.")

        if run_blacklist:
            if is_blacklisted(file_bytes):
                st.error("Image is blacklisted - known attack image!")
            else:
                st.success("Image not found in blacklist.")

        if run_model_consistency:
            consistent = check_model_consistency(img)
            if consistent:
                st.success("Model predictions are consistent.")
            else:
                st.error("Inconsistent model predictions detected! Potential inference attack.")

else:
    st.info("Please upload a CAPTCHA image to get started.")







# import streamlit as st
# import cv2
# import numpy as np
# import tensorflow as tf
# from ultralytics import YOLO
# import joblib
# from PIL import Image
# import io

# # -------------------------
# # Page configuration (must be first Streamlit command)
# # -------------------------
# st.set_page_config(page_title="CAPTCHA Solver with Security Analysis", layout="centered")

# # -------------------------
# # Model loading function with caching to avoid reloading on every run
# # -------------------------
# @st.cache_resource
# def load_models():
#     """
#     Load the trained character recognition model, label encoder, and YOLO detection model.
#     The TF model is loaded without compiling to avoid optimizer warnings.
#     """
#     model = tf.keras.models.load_model("FINALMODELS/captchasolve.h5", compile=False)
#     label_encoder = joblib.load('trained_label_encoder.pkl')
#     yolo_model = YOLO("yolo_model2/best.pt")
#     return model, label_encoder, yolo_model

# model, label_encoder, yolo_model = load_models()

# # -------------------------
# # Image preprocessing and CAPTCHA solving
# # -------------------------
# def check_letter_tarakom(image_segment):
#     """
#     Heuristic to filter noisy or irrelevant letter crops by analyzing black pixel density.
#     """
#     _, binary_img = cv2.threshold(image_segment, 127, 255, cv2.THRESH_BINARY)
#     col_sums = 90 - (np.sum(binary_img, axis=0, keepdims=True) / 255)
#     total_columns = len(col_sums[0])
#     black_columns = sum(1 for col in col_sums[0] if np.sum(col) >= 175)
#     return (total_columns - black_columns) <= 22

# def preprocess_letter(letter_crop):
#     """
#     Resize and binarize letter images before feeding into the CNN model.
#     """
#     resized = cv2.resize(letter_crop, (32, 52))
#     gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
#     return binary

# def solve_captcha(image_np):
#     """
#     Detect characters in the CAPTCHA using YOLO, preprocess each letter, 
#     and classify with the CNN model.
#     """
#     results = yolo_model(image_np)
#     detections = sorted(results[0].boxes, key=lambda box: box.xyxy[0][0])

#     letters = []
#     last_box_coords = []
#     count = 0

#     for box in detections:
#         x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#         letter_crop = image_np[y1:y2, x1:x2]

#         if last_box_coords:
#             # Ensure sufficient spacing between characters
#             if x1 - last_box_coords[0] > 10:
#                 preprocessed = preprocess_letter(letter_crop)
#                 if count == 8:
#                     if check_letter_tarakom(letter_crop):
#                         letters.append(preprocessed)
#                 elif count <= 7:
#                     letters.append(preprocessed)
#                 last_box_coords = [x1, y1, x2, y2]
#                 count += 1
#         else:
#             preprocessed = preprocess_letter(letter_crop)
#             if check_letter_tarakom(letter_crop):
#                 letters.append(preprocessed)
#             last_box_coords = [x1, y1, x2, y2]
#             count += 1

#     # Predict each letter using CNN + label encoder
#     predictions = []
#     for letter in letters:
#         input_img = letter.reshape(1, 52, 32, 1)
#         pred_probs = model.predict(input_img)
#         pred_class = label_encoder.inverse_transform([pred_probs.argmax()])
#         predictions.append(pred_class[0])

#     return "".join(predictions)

# # -------------------------
# # Security Analysis Utilities
# # -------------------------
# def is_valid_image(file_bytes):
#     """
#     Validate uploaded file is a proper image and not corrupted.
#     """
#     try:
#         image = Image.open(file_bytes)
#         image.verify()
#         return True
#     except Exception:
#         return False

# def check_file_size(uploaded_file, max_mb=5):
#     """
#     Ensure uploaded file is under maximum allowed size (default 5 MB).
#     """
#     return uploaded_file.size < max_mb * 1024 * 1024

# def analyze_image_entropy(image):
#     """
#     Calculate Shannon entropy of image grayscale histogram as an indicator
#     of randomness (low entropy might mean synthetic images, high entropy
#     might indicate noise or adversarial attempts).
#     """
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
#     hist_norm = hist.ravel() / hist.sum()
#     entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-7))
#     return entropy

# # -------------------------
# # Streamlit User Interface
# # -------------------------
# st.title("🛡️ CAPTCHA Solver with Integrated Security Analysis")

# uploaded_file = st.file_uploader("Upload a CAPTCHA image", type=["png", "jpg", "jpeg"])
# enable_security = st.checkbox("Enable Security Checks", value=True)

# if uploaded_file:
#     st.image(uploaded_file, caption="Uploaded CAPTCHA Image", use_column_width=True)

#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     file_buffer = io.BytesIO(file_bytes)

#     if not check_file_size(uploaded_file):
#         st.error("❌ Uploaded file exceeds size limit of 5 MB.")
#     elif enable_security and not is_valid_image(file_buffer):
#         st.error("❌ Uploaded file is not a valid or is corrupted image.")
#     else:
#         image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

#         if enable_security:
#             entropy_val = analyze_image_entropy(image)
#             st.info(f"🧠 Image Entropy: {entropy_val:.2f}")
#             if entropy_val < 3.0:
#                 st.warning("⚠️ Low entropy detected - image may be synthetic or adversarial.")
#             elif entropy_val > 7.5:
#                 st.warning("⚠️ High entropy detected - image may contain noise or attacks.")
#             else:
#                 st.success("✅ Image entropy is within expected range.")

#         if st.button("Solve CAPTCHA"):
#             with st.spinner("Processing..."):
#                 try:
#                     captcha_text = solve_captcha(image)
#                     st.success(f"✅ CAPTCHA Prediction: `{captcha_text}`")
#                 except Exception as err:
#                     st.error(f"❌ Error during CAPTCHA solving: {err}")

# ```python
# import streamlit as st
# import cv2
# import numpy as np
# import tensorflow as tf
# from ultralytics import YOLO
# import joblib
# from PIL import Image
# import io
# import time
# import logging
# from collections import deque
# from functools import wraps

# # ─────────────────────────────────────────────────────────────────────────────
# # 1) PAGE CONFIG & LOGGING
# # ─────────────────────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Enterprise CAPTCHA Solver",
#     page_icon="🔒",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )
# logging.basicConfig(
#     filename='app.log',
#     level=logging.INFO,
#     format='%(asctime)s %(levelname)s %(message)s'
# )

# # ─────────────────────────────────────────────────────────────────────────────
# # 2) AUTHENTICATION & RATE LIMITING
# # ─────────────────────────────────────────────────────────────────────────────
# RATE_LIMIT = 5  # requests per minute
# rate_registry = {}
# def require_api_key(fn):
#     @wraps(fn)
#     def wrapper(*args, **kwargs):
#         key = st.sidebar.text_input("API Key", type="password")
#         if not key or len(key) < 8:
#             st.sidebar.error("🔑 Enter a valid API key (min 8 chars)")
#             st.stop()
#         timestamps = rate_registry.setdefault(key, deque())
#         now = time.time()
#         while timestamps and now - timestamps[0] > 60:
#             timestamps.popleft()
#         if len(timestamps) >= RATE_LIMIT:
#             st.sidebar.error("⏱ Rate limit exceeded. Try again later.")
#             st.stop()
#         timestamps.append(now)
#         return fn(*args, **kwargs)
#     return wrapper

# # ─────────────────────────────────────────────────────────────────────────────
# # 3) MODEL LOADING & CACHING
# # ─────────────────────────────────────────────────────────────────────────────
# @st.cache_resource
# def load_models():
#     cnn = tf.keras.models.load_model("FINALMODELS/captchasolve.h5", compile=False)
#     le = joblib.load("trained_label_encoder.pkl")
#     yolo = YOLO("yolo_model2/best.pt")
#     return cnn, le, yolo
# model, label_encoder, yolo_model = load_models()

# # ─────────────────────────────────────────────────────────────────────────────
# # 4) SECURITY UTILITIES
# # ─────────────────────────────────────────────────────────────────────────────
# def is_valid_image(buffer: io.BytesIO) -> bool:
#     try:
#         img = Image.open(buffer); img.verify()
#         return True
#     except:
#         return False
# def check_file_size(file, max_mb=2):
#     return file.size <= max_mb * 1024 * 1024
# def image_entropy(img: np.ndarray) -> float:
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     hist = cv2.calcHist([gray],[0],None,[256],[0,256]).ravel()
#     p = hist / hist.sum()
#     return -np.sum(p * np.log2(p + 1e-7))
# entropy_history = deque(maxlen=20)
# latency_history = deque(maxlen=20)

# # ─────────────────────────────────────────────────────────────────────────────
# # 5) CAPTCHA SOLVING PIPELINE
# # ─────────────────────────────────────────────────────────────────────────────
# def extract_letters(img: np.ndarray):
#     results = yolo_model(img)
#     boxes = sorted(results[0].boxes, key=lambda b: b.xyxy[0][0])
#     letters, last_x = [], None
#     count = 0
#     for b in boxes:
#         x1,y1,x2,y2 = map(int,b.xyxy[0].tolist())
#         if last_x and x1 - last_x < 8: continue
#         crop = img[y1:y2, x1:x2]
#         small = cv2.resize(crop,(32,52))
#         gray = cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
#         _, binar = cv2.threshold(gray,128,255,cv2.THRESH_BINARY)
#         letters.append(binar)
#         last_x = x1; count += 1
#         if count>=8: break
#     return letters
# def predict_captcha(letters):
#     result = []
#     for l in letters:
#         inp = l.reshape(1,52,32,1)
#         p = model.predict(inp)
#         cls = label_encoder.inverse_transform([p.argmax()])[0]
#         result.append(cls)
#     return "".join(result)

# # ─────────────────────────────────────────────────────────────────────────────
# # 6) UI LAYOUT WITH TABS
# # ─────────────────────────────────────────────────────────────────────────────
# st.sidebar.title("Configuration")
# max_size = st.sidebar.slider("Max Upload Size (MB)", 1, 10, 2)
# enable_analysis = st.sidebar.checkbox("Enable Detailed Analysis", True)

# tab1, tab2 = st.tabs(["🔓 Solve CAPTCHA", "📊 Security Dashboard"])

# with tab1:
#     st.header("Solve CAPTCHA")
#     uploaded = st.file_uploader("Upload CAPTCHA Image", type=["png","jpg","jpeg"])
#     if uploaded:
#         buf = io.BytesIO(uploaded.read())
#         if not check_file_size(uploaded, max_size): st.error("❗ File exceeds size limit.")
#         elif enable_analysis and not is_valid_image(buf): st.error("❗ Invalid or corrupted image.")
#         else:
#             buf.seek(0)
#             arr = np.frombuffer(buf.read(), np.uint8)
#             img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
#             start = time.time()
#             letters = extract_letters(img)
#             text = predict_captcha(letters)
#             latency = (time.time()-start)*1000
#             latency_history.append(latency)
#             st.success(f"Predicted CAPTCHA: **{text}** (Latency: {latency:.1f} ms)")
#             st.image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB), caption="Input CAPTCHA")
#             if enable_analysis:
#                 ent = image_entropy(img)
#                 entropy_history.append(ent)
#                 st.metric("Image Entropy", f"{ent:.2f}")
#                 logging.info(f"CAPTCHA: {text}, Entropy: {ent:.2f}, Latency: {latency:.1f}ms")

# with tab2:
#     st.header("Security Dashboard")
#     cols = st.columns(2)
#     cols[0].line_chart(list(entropy_history), height=200)
#     cols[1].line_chart(list(latency_history), height=200)
#     st.subheader("Log Preview")
#     if st.button("Refresh Logs"):
#         try:
#             lines = open('app.log').read().splitlines()[-10:]
#             st.text("\n".join(lines))
#         except FileNotFoundError:
#             st.text("No logs found.")

# # ─────────────────────────────────────────────────────────────────────────────
# # 7) ADVERSARIAL HARDENING SCRIPT (Generator Side)
# # ─────────────────────────────────────────────────────────────────────────────
# # This script generates new CAPTCHA challenges by selecting those the solver fails.
# from ai_captcha_solver import solve_captcha, generate_captcha  # hypothetical API
# import os
# OUTPUT_DIR = "hardened_captchas"
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# for i in range(1000):
#     img, label = generate_captcha()
#     try:
#         pred = solve_captcha(img)
#     except:
#         continue
#     if pred != label:
#         cv2.imwrite(f"{OUTPUT_DIR}/adv_{label}_{i}.png", img)

# # ─────────────────────────────────────────────────────────────────────────────
# # 8) IP-BASED RATE LIMITER (FastAPI Example)
# # ─────────────────────────────────────────────────────────────────────────────
# from fastapi import FastAPI, Request, HTTPException
# from starlette.middleware.base import BaseHTTPMiddleware
# app_api = FastAPI()

# ip_registry = {}
# RATE_PER_MIN = 60
# class IPRateLimit(BaseHTTPMiddleware):
#     async def dispatch(self, request: Request, call_next):
#         ip = request.client.host
#         now = time.time()
#         entries = ip_registry.setdefault(ip, [])
#         # cleanup
#         ip_registry[ip] = [t for t in entries if now - t < 60]
#         if len(ip_registry[ip]) >= RATE_PER_MIN:
#             raise HTTPException(429, "Too Many Requests")
#         ip_registry[ip].append(now)
#         return await call_next(request)
# app_api.add_middleware(IPRateLimit)

# @app_api.get("/solve-captcha/")
# async def solve_endpoint(request: Request):
#     return {"captcha": "..."}

