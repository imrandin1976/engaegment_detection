import streamlit as st
import cv2
import time
import os
import shutil
from datetime import timedelta
import torch
import torchvision.transforms as T
import torchvision.models as models

# ===================== Model utils =====================
def load_model(model_path, num_classes=2, device="cpu"):
    model = models.squeezenet1_1(pretrained=False)
    model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1))
    model.num_classes = num_classes
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model

@torch.inference_mode()
def predict_idx(model, img_bgr, device):
    """img_bgr: numpy HWC (BGR). Returns class index (0=engaged, 1=unengaged)."""
    # Convert to RGB before PIL
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = transform(img_rgb).unsqueeze(0).to(device)
    logits = model(x)
    return int(torch.argmax(logits, dim=1).item())

# ===================== App config =====================
st.set_page_config(page_title="Engagement Tracker", layout="centered")
st.title("🎥 Real-Time Engagement Detection (Buffered)")

IMAGE_SIZE = (224, 224)  # used only for display/preview if you want
CLASS_NAMES = ["engaged", "unengaged"]
MODEL_PATH = "models/squeezenet1_1_2stage.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#TMP_DIR = "/tmp/engagement_cache"       # buffer during capture
TMP_DIR = os.path.join(os.getcwd(), "tmp", "engagement_cache")
LOG_DIR = "logs"                        # final output for unengaged frames
UNENGAGED_LIST = os.path.join(LOG_DIR, "unengaged_log.txt")

os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Cache model once (used only during batch classification)
@st.cache_resource
def cached_model(device):
    return load_model(MODEL_PATH, num_classes=len(CLASS_NAMES), device=device)

model = cached_model(DEVICE)

# ===================== Sidebar: camera discovery =====================
st.sidebar.header("Camera")
available_indices = []
for index in range(5):
    cap_probe = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    ok, _ = cap_probe.read()
    if ok:
        available_indices.append(index)
    cap_probe.release()

selected_cam = st.sidebar.selectbox("Select Camera Index",
                                    available_indices if available_indices else [0])

# Optional video upload (not tied to classification in this buffered version)
video_file = st.sidebar.file_uploader("Upload a lecture video", type=["mp4"])
if video_file:
    st.video(video_file)

# ===================== Session state =====================
ss = st.session_state
if "detection_running" not in ss:
    ss.detection_running = False
if "last_capture_time" not in ss:
    ss.last_capture_time = None
if "start_time" not in ss:
    ss.start_time = None
if "cap" not in ss:
    ss.cap = None  # persistent cv2.VideoCapture
if "saved_count" not in ss:
    ss.saved_count = 0  # snapshots taken in this session

# ===================== Helpers =====================
def open_camera(idx):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    # Optional warmup
    for _ in range(5):
        cap.read()
    return cap

def close_camera():
    if ss.cap is not None:
        try:
            ss.cap.release()
        except Exception:
            pass
        ss.cap = None

def seconds_since_start(now=None):
    if ss.start_time is None:
        return 0
    if now is None:
        now = time.time()
    return int(now - ss.start_time)

def timestamp_str(total_seconds):
    # Use HH-MM-SS style, zero-padded
    return str(timedelta(seconds=total_seconds)).split(".")[0].replace(":", "-")

def save_snapshot(frame_bgr, ts_str):
    # Save raw captured frame with timestamp as filename
    fname = f"{ts_str}.jpg"
    path = os.path.join(TMP_DIR, fname)
    cv2.imwrite(path, frame_bgr)
    return path

def list_tmp_images_sorted():
    files = [f for f in os.listdir(TMP_DIR) if f.lower().endswith(".jpg")]
    # Sort by timestamp embedded in filename if possible
    files.sort()
    return [os.path.join(TMP_DIR, f) for f in files]

def append_unengaged_timestamp(ts_str):
    with open(UNENGAGED_LIST, "a") as f:
        f.write(f"{ts_str}\n")

# ===================== Controls =====================
col1, col2 = st.columns(2)
if col1.button("▶ Start Detection", disabled=ss.detection_running):
    # Reset state
    ss.detection_running = True
    ss.last_capture_time = None
    ss.start_time = time.time()
    ss.saved_count = 0
    # Clear any leftovers in TMP_DIR for a clean run
    for f in list_tmp_images_sorted():
        try: os.remove(f)
        except Exception: pass
    # Open persistent camera
    close_camera()
    ss.cap = open_camera(selected_cam)
    st.success("Capture started: buffering frames to /tmp/engagement_cache")

if col2.button("⏹ Stop & Classify", disabled=not ss.detection_running):
    # Stop capture
    ss.detection_running = False
    ss.last_capture_time = None
    close_camera()
    st.success("Capture stopped. Starting batch classification...")

    # === Batch classification phase with progress bar ===
    tmp_images = list_tmp_images_sorted()
    total = len(tmp_images)
    if total == 0:
        st.info("No images captured to classify.")
    else:
        progress = st.progress(0, text="Classifying buffered images...")
        kept = 0
        deleted = 0

        for i, img_path in enumerate(tmp_images, start=1):
            img = cv2.imread(img_path)  # BGR
            if img is None:
                # Corrupt or unreadable file; remove it
                try: os.remove(img_path)
                except Exception: pass
                progress.progress(i / total, text=f"Skipping unreadable image ({i}/{total})")
                continue

            pred = predict_idx(model, img, DEVICE)  # 0=engaged, 1=unengaged

            # Extract timestamp string (filename without extension)
            ts_str = os.path.splitext(os.path.basename(img_path))[0]

            if pred == 0:
                # engaged -> delete
                try:
                    os.remove(img_path)
                    deleted += 1
                except Exception:
                    pass
            else:
                # unengaged -> move to logs and record timestamp
                dest_path = os.path.join(LOG_DIR, os.path.basename(img_path))
                try:
                    shutil.move(img_path, dest_path)
                except Exception:
                    # If move fails (e.g., across devices), fallback to copy+delete
                    shutil.copy2(img_path, dest_path)
                    os.remove(img_path)
                kept += 1
                append_unengaged_timestamp(ts_str)

            progress.progress(i / total, text=f"Processed {i}/{total} images")

        st.success(f"Batch done. Kept (unengaged): {kept}  |  Deleted (engaged): {deleted}")
        st.write(f"Unengaged timestamps appended to `{UNENGAGED_LIST}`.")
        if kept > 0:
            st.markdown("### Recent Unengaged Frames")
            # Show last few kept images from LOG_DIR
            kept_files = [f for f in os.listdir(LOG_DIR) if f.lower().endswith(".jpg")]
            kept_files.sort(reverse=True)
            for f in kept_files[:10]:
                st.image(os.path.join(LOG_DIR, f), caption=f"Unengaged at {os.path.splitext(f)[0]}")

# ===================== Capture loop (buffer only) =====================
if ss.detection_running:
    now = time.time()
    if (ss.last_capture_time is None) or (now - ss.last_capture_time >= 1):
        ss.last_capture_time = now

        if ss.cap is None:
            # Try reopening if something closed it
            ss.cap = open_camera(selected_cam)

        ok, frame = (False, None)
        if ss.cap is not None:
            ok, frame = ss.cap.read()

        if not ok or frame is None:
            st.warning("Could not capture frame from camera.")
            # Try to reopen once next tick
            close_camera()
        else:
            # Save to /tmp every 1 second with HH-MM-SS timestamp since start
            ts = timestamp_str(seconds_since_start(now))
            save_snapshot(frame, ts)
            ss.saved_count += 1
            st.write(f"Buffered snapshot #{ss.saved_count} at {ts}")

    # Keep the loop ticking
    st.rerun()

# ===================== Footer info =====================
st.info(
    "This mode buffers 1 frame/sec to /tmp/engagement_cache while running. "
    "When you press 'Stop & Classify', the app classifies all buffered frames: "
    "deleting engaged ones and keeping unengaged ones in ./logs with a timestamp list."
)
