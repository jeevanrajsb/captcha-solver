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
    # Load your trained CNN model for character recognition
    model = tf.keras.models.load_model("FINALMODELS/captchasolve.h5")
    # Load the label encoder used during training
    label_encoder = joblib.load('trained_label_encoder.pkl')
    # Load your YOLO model for character detection
    yolo_model = YOLO("yolo_model2/best.pt")
    return model, label_encoder, yolo_model

model, label_encoder, yolo_model = load_models()

# --- OCR Utility functions ---

def check_letter_tarakom(pic):
    # Binary thresholding and check for some pixel conditions for letter validation
    _, pic = cv2.threshold(pic, 127, 255, cv2.THRESH_BINARY)
    s = 90 - (np.sum(pic, axis=0, keepdims=True) / 255)
    total = len(s[0])
    howmanyblack = sum(1 for i in s[0] if np.sum(i) >= 175)
    return total - howmanyblack <= 22

def preprocess_letter(letter_crop):
    # Resize, grayscale and binarize each detected letter image before OCR
    resized_letter = cv2.resize(letter_crop, (32, 52))
    resized_letter = cv2.cvtColor(resized_letter, cv2.COLOR_BGR2GRAY)
    _, binarized = cv2.threshold(resized_letter, 128, 255, cv2.THRESH_BINARY)
    return binarized

def solve_captcha(image):
    # Use YOLO to detect letters in the image
    results = yolo_model(image)
    # Sort detected bounding boxes from left to right (based on x1)
    detections = sorted(results[0].boxes, key=lambda box: box.xyxy[0][0])

    letters = []
    last_one = []
    howmany = 0

    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        letter_crop = image[y1:y2, x1:x2]

        if last_one:
            # Only add letter if sufficiently far from previous
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
    # Calculate Shannon entropy of grayscale image histogram
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-7))
    return entropy

def detect_blur(image):
    # Use variance of Laplacian to estimate blur (low variance = blurry)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < 100  # Threshold, tune if needed

def detect_noise(image):
    # Check noise level by binarizing and counting black pixels ratio
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    noise_ratio = np.sum(binary == 0) / binary.size
    return noise_ratio > 0.6  # Threshold, tune if needed

# Placeholder for known bad images' hashes
BLACKLISTED_HASHES = {
    "d41d8cd98f00b204e9800998ecf8427e"  # Example md5 hash
}

def get_image_hash(image_bytes):
    # Compute MD5 hash of image bytes for blacklist checking
    return hashlib.md5(image_bytes).hexdigest()

def is_blacklisted(image_bytes):
    return get_image_hash(image_bytes) in BLACKLISTED_HASHES

def check_model_consistency(image, repeat=2):
    # Run OCR multiple times to check if predictions are consistent
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
        # Read image and convert to numpy RGB array
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
