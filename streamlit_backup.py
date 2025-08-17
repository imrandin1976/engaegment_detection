import streamlit as st
import cv2
import time
import os
from datetime import timedelta
import torch
import torchvision.transforms as T
import torchvision.models as models

# === Replace these with your actual implementations! ===
def load_model(model_path, num_classes=2):
    model = models.squeezenet1_1(pretrained=False)
    model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1))
    model.num_classes = num_classes
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict_engagement(model, img, device, class_names):
    # Assumes input img is a numpy array (HWC, BGR)
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(img).unsqueeze(0)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = int(torch.argmax(output, dim=1).cpu().item())
    return pred_idx  # 0 = engaged, 1 = unengaged
# =======================================================

st.set_page_config(page_title="Engagement Tracker", layout="centered")
st.title("🎥 Real-Time Engagement Detection")

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["engaged", "unengaged"]  # index 0: engaged, index 1: unengaged
MODEL_PATH = "models/squeezenet1_1_2stage.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model once
@st.cache_resource
def cached_model():
    return load_model(MODEL_PATH, num_classes=len(CLASS_NAMES))
model = cached_model()

# --- Camera Selection ---
st.sidebar.header("Camera")
available_indices = []
for index in range(5):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.read()[0]:
        available_indices.append(index)
    cap.release()

selected_cam = st.sidebar.selectbox("Select Camera Index", available_indices if available_indices else [0])

# --- Video Selection ---
video_file = st.sidebar.file_uploader("Upload a lecture video", type=["mp4"])
if video_file:
    st.video(video_file)

# --- State management ---
if "detection_running" not in st.session_state:
    st.session_state.detection_running = False
if "last_capture_time" not in st.session_state:
    st.session_state.last_capture_time = None
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "logged_frames" not in st.session_state:
    st.session_state.logged_frames = []

# --- Detection Control ---
col1, col2 = st.columns(2)
if col1.button("▶ Start Detection", key="start_detect_btn") and not st.session_state.detection_running:
    st.session_state.detection_running = True
    st.session_state.start_time = time.time()
    st.session_state.logged_frames = []
    st.success("Detection started.")

if col2.button("⏹ Stop Detection", key="stop_detect_btn") and st.session_state.detection_running:
    st.session_state.detection_running = False
    st.session_state.last_capture_time = None
    st.success("Detection stopped.")

# --- Detection Loop ---
if st.session_state.detection_running:
    current_time = time.time()
    if (
        st.session_state.last_capture_time is None or
        current_time - st.session_state.last_capture_time >= 1
    ):
        st.session_state.last_capture_time = current_time
        cap = cv2.VideoCapture(selected_cam, cv2.CAP_DSHOW)
        # Discard first 10 frames for camera warmup
        for _ in range(10):
            cap.read()
        ret, frame = cap.read()
        cap.release()
        if ret:
            video_timestamp = str(
                timedelta(seconds=int(current_time - st.session_state.start_time))
            ).replace(":", "-")
            resized = cv2.resize(frame, IMAGE_SIZE)
            pred_idx = predict_engagement(model, resized, DEVICE, CLASS_NAMES)
            if pred_idx == 1:  # Only save/display if unengaged
                os.makedirs("logs", exist_ok=True)
                frame_path = os.path.join("logs", f"{video_timestamp}.jpg")
                log_path = os.path.join("logs", "unengaged_log.txt")
                cv2.imwrite(frame_path, frame)
                with open(log_path, "a") as log_file:
                    log_file.write(f"{video_timestamp}\n")
                st.session_state.logged_frames.append((frame_path, video_timestamp))
        else:
            st.warning("Could not capture frame.")
    st.experimental_rerun()

# --- Display Logged Unengaged Frames ---
if st.session_state.logged_frames:
    st.markdown("### Tracked Unengaged Frames")
    for frame_path, video_timestamp in st.session_state.logged_frames[-10:][::-1]:  # show last 10, newest first
        st.image(frame_path, caption=f"Unengaged at {video_timestamp}")

st.info("Only unengaged frames are logged and displayed, with timestamp as filename. App is optimized for fast, robust research.")

