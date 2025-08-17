import streamlit as st
import torch
import torchvision.transforms as T
import torchvision.models as models
import cv2

st.title("Single Frame Inference Test")
MODEL_PATH = "models/squeezenet1_1_2stage.pth"
CLASS_NAMES = ['Engagement', 'Unengaged']  # <<-- FIXED TO MATCH TEST FOLDER ORDER
IMAGE_SIZE = (224, 224)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model(model_path, num_classes=2):
    model = models.squeezenet1_1(weights=None)
    model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1))
    model.num_classes = num_classes
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model(MODEL_PATH, num_classes=len(CLASS_NAMES))

available_indices = []
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.read()[0]:
        available_indices.append(i)
    cap.release()
selected_cam = st.selectbox("Select Camera Index", available_indices if available_indices else [0])

if st.button("Capture & Predict"):
    cap = cv2.VideoCapture(selected_cam, cv2.CAP_DSHOW)
    for _ in range(10):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption="Captured Frame")
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = transform(frame_rgb).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor)
            st.write("Raw logits:", output.cpu().numpy())
            pred_idx = int(torch.argmax(output, dim=1).cpu().item())
            pred_label = CLASS_NAMES[pred_idx]
        st.success(f"Model Output: Class Index = {pred_idx}, Label = '{pred_label}'")
    else:
        st.warning("Could not capture frame.")
