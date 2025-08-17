import streamlit as st
import cv2
import time
import os
import shutil
import json
from io import BytesIO
from datetime import timedelta, datetime
import torch
import torchvision.transforms as T
import torchvision.models as models

# Optional: Google Sheets
try:
    import gspread
    GSPREAD_AVAILABLE = True
except Exception:
    GSPREAD_AVAILABLE = False

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
st.title("🎥 Real-Time Engagement Detection (Buffered + Google Sheets)")

CLASS_NAMES = ["engaged", "unengaged"]
MODEL_PATH = "models/squeezenet1_1_2stage.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths relative to app folder
TMP_DIR = os.path.join(os.getcwd(), "tmp", "engagement_cache")  # buffered snapshots
LOG_DIR = os.path.join(os.getcwd(), "logs")                     # kept unengaged
UNENGAGED_LIST = os.path.join(LOG_DIR, "unengaged_log.txt")

os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Cache model once (used only during batch classification)
@st.cache_resource
def cached_model(device):
    return load_model(MODEL_PATH, num_classes=len(CLASS_NAMES), device=device)

model = cached_model(DEVICE)

# ===================== Sidebar: camera & Sheets =====================
st.sidebar.header("Camera")
available_indices = []
for index in range(5):
    cap_probe = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    ok, _ = cap_probe.read()
    if ok:
        available_indices.append(index)
    cap_probe.release()

selected_cam = st.sidebar.selectbox(
    "Select Camera Index",
    available_indices if available_indices else [0]
)

video_file = st.sidebar.file_uploader("Upload a lecture video (optional)", type=["mp4"])
if video_file:
    st.video(video_file)

st.sidebar.header("Google Sheets Sync")
sync_to_sheets = st.sidebar.checkbox("Enable Google Sheets Sync", value=False)
sheet_id = st.sidebar.text_input("Google Sheet ID (the long ID in the URL)", "")
worksheet_name = st.sidebar.text_input("Worksheet name", "unengaged")
service_json_file = st.sidebar.file_uploader("Service Account JSON", type=["json"])

if sync_to_sheets and not GSPREAD_AVAILABLE:
    st.sidebar.error("Install gspread: pip install gspread google-auth")

# ===================== Session state =====================
ss = st.session_state
if "detection_running" not in ss:
    ss.detection_running = False
if "last_capture_time" not in ss:
    ss.last_capture_time = None
if "start_time" not in ss:
    ss.start_time = None
if "cap" not in ss:
    ss.cap = None
if "saved_count" not in ss:
    ss.saved_count = 0

# ===================== Helpers =====================
def open_camera(idx):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
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
    return str(timedelta(seconds=total_seconds)).split(".")[0].replace(":", "-")

def save_snapshot(frame_bgr, ts_str):
    fname = f"{ts_str}.jpg"
    path = os.path.join(TMP_DIR, fname)
    cv2.imwrite(path, frame_bgr)
    return path

def list_tmp_images_sorted():
    files = [f for f in os.listdir(TMP_DIR) if f.lower().endswith(".jpg")]
    files.sort()
    return [os.path.join(TMP_DIR, f) for f in files]

def append_unengaged_timestamp(ts_str):
    # Ensure the file exists even if it's the first write
    if not os.path.exists(UNENGAGED_LIST):
        with open(UNENGAGED_LIST, "w") as f:
            f.write("")  # create empty
    with open(UNENGAGED_LIST, "a") as f:
        f.write(f"{ts_str}\n")

def ensure_logfile_exists_with_note_if_empty():
    # If the run produced zero unengaged frames, still ensure there's a file
    if not os.path.exists(UNENGAGED_LIST):
        with open(UNENGAGED_LIST, "w") as f:
            f.write(f"No unengaged frames detected at {datetime.now().isoformat(timespec='seconds')}\n")

def gsheets_append_rows(timestamps, sheet_id, worksheet_name, sa_json_bytes):
    """
    Append each timestamp as a new row, first column.
    - timestamps: list[str]
    - sheet_id: str
    - worksheet_name: str
    - sa_json_bytes: uploaded JSON file bytes
    """
    if not timestamps:
        return {"ok": True, "appended": 0, "msg": "No unengaged timestamps to sync."}

    sa_dict = json.loads(sa_json_bytes.decode("utf-8"))
    gc = gspread.service_account_from_dict(sa_dict)
    sh = gc.open_by_key(sheet_id)

    try:
        ws = sh.worksheet(worksheet_name)
    except gspread.exceptions.WorksheetNotFound:
        # Create worksheet with a single column if not exists
        ws = sh.add_worksheet(title=worksheet_name, rows="1000", cols="1")

    # Prepare rows: each timestamp in its own row, first column
    values = [[ts] for ts in timestamps]
    ws.append_rows(values, value_input_option="RAW")
    return {"ok": True, "appended": len(values), "msg": f"Appended {len(values)} rows to {worksheet_name}"}

# ===================== Controls =====================
col1, col2 = st.columns(2)
if col1.button("▶ Start Detection", disabled=ss.detection_running):
    ss.detection_running = True
    ss.last_capture_time = None
    ss.start_time = time.time()
    ss.saved_count = 0
    # Clean buffer for a fresh run
    for p in list_tmp_images_sorted():
        try: os.remove(p)
        except Exception: pass
    close_camera()
    ss.cap = open_camera(selected_cam)
    st.success(f"Capture started → buffering frames to {TMP_DIR}")

if col2.button("⏹ Stop & Classify", disabled=not ss.detection_running):
    ss.detection_running = False
    ss.last_capture_time = None
    close_camera()
    st.success("Capture stopped. Starting batch classification...")

    tmp_images = list_tmp_images_sorted()
    total = len(tmp_images)
    new_unengaged_timestamps = []

    if total == 0:
        ensure_logfile_exists_with_note_if_empty()
        st.info("No images captured to classify.")
    else:
        progress = st.progress(0, text="Classifying buffered images...")
        kept = 0
        deleted = 0

        for i, img_path in enumerate(tmp_images, start=1):
            img = cv2.imread(img_path)
            if img is None:
                try: os.remove(img_path)
                except Exception: pass
                progress.progress(i / total, text=f"Skipping unreadable image ({i}/{total})")
                continue

            pred = predict_idx(model, img, DEVICE)  # 0=engaged, 1=unengaged
            ts_str = os.path.splitext(os.path.basename(img_path))[0]

            if pred == 0:
                try:
                    os.remove(img_path)
                    deleted += 1
                except Exception:
                    pass
            else:
                dest_path = os.path.join(LOG_DIR, os.path.basename(img_path))
                try:
                    shutil.move(img_path, dest_path)
                except Exception:
                    shutil.copy2(img_path, dest_path)
                    os.remove(img_path)
                kept += 1
                append_unengaged_timestamp(ts_str)
                new_unengaged_timestamps.append(ts_str)

            progress.progress(i / total, text=f"Processed {i}/{total} images")

        # Always ensure the log file exists even if none kept
        if kept == 0:
            ensure_logfile_exists_with_note_if_empty()

        st.success(f"Batch done. Kept (unengaged): {kept}  |  Deleted (engaged): {deleted}")
        st.write(f"Unengaged timestamps written to `{UNENGAGED_LIST}`.")

        # ===== Google Sheets sync (optional) =====
        if sync_to_sheets:
            if not GSPREAD_AVAILABLE:
                st.error("Google Sheets sync requested, but gspread is not installed.")
            elif not sheet_id:
                st.error("Please provide a Google Sheet ID.")
            elif service_json_file is None:
                st.error("Please upload a Service Account JSON.")
            else:
                with st.spinner("Syncing unengaged timestamps to Google Sheets..."):
                    try:
                        result = gsheets_append_rows(
                            new_unengaged_timestamps,
                            sheet_id,
                            worksheet_name,
                            service_json_file.read()
                        )
                        if result["ok"]:
                            st.success(result["msg"])
                        else:
                            st.error(result.get("msg", "Google Sheets sync failed."))
                    except Exception as e:
                        st.error(f"Google Sheets sync error: {e}")

        # Show recent unengaged thumbnails
        if kept > 0:
            st.markdown("### Recent Unengaged Frames")
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
            ss.cap = open_camera(selected_cam)

        ok, frame = (False, None)
        if ss.cap is not None:
            ok, frame = ss.cap.read()

        if not ok or frame is None:
            st.warning("Could not capture frame from camera.")
            close_camera()
        else:
            ts = timestamp_str(seconds_since_start(now))
            save_snapshot(frame, ts)
            ss.saved_count += 1
            st.write(f"Buffered snapshot #{ss.saved_count} at {ts}")

    st.rerun()

# ===================== Footer info =====================
st.info(
    "Buffers 1 frame/sec to ./tmp/engagement_cache while running. "
    "On 'Stop & Classify', engaged frames are deleted; unengaged frames are moved to ./logs "
    "and timestamps are appended to logs/unengaged_log.txt. "
    "If Google Sheets Sync is enabled, new unengaged timestamps are appended to the selected sheet."
)
