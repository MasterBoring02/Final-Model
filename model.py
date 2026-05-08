import streamlit as st
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from PIL import Image
from pathlib import Path

st.set_page_config(page_title="Emotion Detection", layout="wide")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTIONS = ['happy', 'sad', 'angry']

BASE_DIR = Path(__file__).resolve().parent

RESNET_ENCODER_PATH = BASE_DIR / "resnet50_encoder_best.pth"
DCNN_FUSED_PATH = BASE_DIR / "dcnn_fused_best.pth"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

class DCNN(nn.Module):

    def __init__(self, input_dim=265, n_classes=3):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.3)

        self.out = nn.Linear(128, n_classes)

    def forward(self, x):

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)

        x = F.relu(self.bn3(self.fc3(x)))
        x = self.drop3(x)

        return self.out(x)

def make_resnet_encoder(embed_size=128):

    model = models.resnet50(weights=None)

    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(num_ftrs, embed_size)

    return model

@st.cache_resource
def load_models():

    resnet = make_resnet_encoder().to(DEVICE)

    resnet.load_state_dict(
        torch.load(
            str(RESNET_ENCODER_PATH),
            map_location=DEVICE
        )
    )

    dcnn = DCNN().to(DEVICE)

    dcnn.load_state_dict(
        torch.load(
            str(DCNN_FUSED_PATH),
            map_location=DEVICE
        )
    )

    resnet.eval()
    dcnn.eval()

    return resnet, dcnn

resnet, dcnn = load_models()

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

profile_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_profileface.xml'
)

st.title("Emotion Detection")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

camera_image = st.camera_input("Capture Image")

image_source = None

if uploaded_file is not None:
    image_source = uploaded_file

elif camera_image is not None:
    image_source = camera_image

if image_source is not None:

    image = Image.open(image_source).convert("RGB")

    frame = np.array(image)

    display = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces = profile_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80,80)
    )

    profile_side = "right"

    if len(faces) == 0:

        flipped = cv2.flip(gray, 1)

        flipped_faces = profile_detector.detectMultiScale(
            flipped,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80,80)
        )

        if len(flipped_faces) > 0:

            profile_side = "left"

            for (x,y,w,h) in flipped_faces:

                x = gray.shape[1] - x - w

                faces = [(x,y,w,h)]

                break

    if len(faces) == 0:

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80,80)
        )

        profile_side = "front"

    if len(faces) > 0:

        x,y,w,h = faces[0]

        face = frame[y:y+h, x:x+w]

        pil = Image.fromarray(face).resize((224,224))

        x_t = transform(pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            emb = resnet(x_t)

        emb = emb.cpu().numpy().squeeze()

        cheek_width = w / (h + 1e-6)

        pseudo_landmarks = np.array([
            x, y,
            x+w, y,
            x, y+h,
            x+w, y+h
        ])

        padding = np.zeros(136 - len(pseudo_landmarks))

        lm_flat = np.concatenate([
            pseudo_landmarks,
            padding
        ])

        fused = np.concatenate([
            emb,
            lm_flat,
            [cheek_width]
        ])

        x_tensor = torch.tensor(
            fused,
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            probs = torch.softmax(
                dcnn(x_tensor),
                dim=1
            ).cpu().numpy().squeeze()

        idx = np.argmax(probs)

        emotion = EMOTIONS[idx]
        confidence = probs[idx] * 100

        cv2.rectangle(
            display,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )

        if profile_side == "left":

            for i in range(9):

                px = int(x + w*0.18 + i*4)
                py = int(y + h*0.45 + np.sin(i)*6)

                cv2.circle(
                    display,
                    (px, py),
                    4,
                    (0,255,255),
                    -1
                )

        elif profile_side == "right":

            for i in range(9):

                px = int(x + w*0.78 - i*4)
                py = int(y + h*0.45 + np.sin(i)*6)

                cv2.circle(
                    display,
                    (px, py),
                    4,
                    (255,255,0),
                    -1
                )

        cv2.putText(
            display,
            f"{emotion} ({confidence:.1f}%)",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )

        st.image(display)

        st.metric("Detected Emotion", emotion)
        st.metric("Confidence", f"{confidence:.2f}%")
        st.metric("Profile", profile_side)

    else:

        st.error("No face detected")
