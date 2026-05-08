```python
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
from mediapipe.python.solutions import face_mesh

st.set_page_config(page_title="Emotion Detection", layout="wide")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTIONS = ['happy', 'sad', 'angry']

BASE_DIR = Path(__file__).resolve().parent

RESNET_ENCODER_PATH = BASE_DIR / "resnet50_encoder_best.pth"
DCNN_FUSED_PATH = BASE_DIR / "dcnn_fused_best.pth"

LEFT_CHEEK_ARCH = [205, 50, 187, 147, 116, 117, 118, 119, 120]
RIGHT_CHEEK_ARCH = [425, 280, 411, 376, 345, 346, 347, 348, 349]

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

mesh = face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5
)

st.title("Side Face Emotion Detection")

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

    small = cv2.resize(frame, (256,192))

    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    results = mesh.process(rgb)

    if results.multi_face_landmarks:

        lm = results.multi_face_landmarks[0].landmark

        h, w, _ = small.shape

        pts = np.array([
            [int(p.x*w), int(p.y*h)]
            for p in lm
        ])

        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)

        face = small[y1:y2, x1:x2]

        if face.size != 0:

            pil = Image.fromarray(face).resize((224,224))

            x_t = transform(pil).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                emb = resnet(x_t)

            emb = emb.cpu().numpy().squeeze()

            if pts.shape[0] >= 68:
                lm_flat = pts[:68].flatten()
            else:
                lm_flat = np.zeros(136)

            cheek_raw = np.linalg.norm(
                pts[234] - pts[454]
            ) / (
                np.linalg.norm(
                    pts[152] - pts[10]
                ) + 1e-6
            )

            fused = np.concatenate([
                emb,
                lm_flat,
                [cheek_raw]
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

            sx = frame.shape[1] / 256
            sy = frame.shape[0] / 192

            bx1 = int(x1 * sx)
            by1 = int(y1 * sy)
            bx2 = int(x2 * sx)
            by2 = int(y2 * sy)

            cv2.rectangle(
                frame,
                (bx1, by1),
                (bx2, by2),
                (0,255,0),
                2
            )

            left_outer = lm[33]
            left_inner = lm[133]

            right_inner = lm[362]
            right_outer = lm[263]

            left_w = abs(left_outer.x - left_inner.x)
            right_w = abs(right_outer.x - right_inner.x)

            ratio = left_w / (right_w + 1e-6)

            show_left = False
            show_right = False

            if ratio < 0.65:
                show_right = True

            elif ratio > 1.5:
                show_left = True

            if show_left:

                for idx_pt in LEFT_CHEEK_ARCH:

                    px = int(pts[idx_pt][0] * sx)
                    py = int(pts[idx_pt][1] * sy)

                    cv2.circle(
                        frame,
                        (px, py),
                        4,
                        (0,255,255),
                        -1
                    )

            elif show_right:

                for idx_pt in RIGHT_CHEEK_ARCH:

                    px = int(pts[idx_pt][0] * sx)
                    py = int(pts[idx_pt][1] * sy)

                    cv2.circle(
                        frame,
                        (px, py),
                        4,
                        (255,255,0),
                        -1
                    )

            cv2.putText(
                frame,
                f"{emotion} ({confidence:.1f}%)",
                (bx1, by1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,255,255),
                2
            )

            st.image(frame, channels="BGR")

            st.metric("Detected Emotion", emotion)
            st.metric("Confidence", f"{confidence:.2f}%")
            st.metric("Cheek Score", f"{cheek_raw:.3f}")

    else:

        st.error("No face detected")
```

requirements.txt

```text
streamlit==1.37.1
opencv-python-headless==4.10.0.84
torch==2.2.2
torchvision==0.17.2
numpy==1.26.4
Pillow==10.4.0
mediapipe==0.10.11
protobuf==4.25.3
```

# runtime.txt

```text
python-3.11
```

# Folder Structure

```text
project/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── resnet50_encoder_best.pth
└── dcnn_fused_best.pth
```

# Run Locally

```bash
streamlit run app.py
```

# Deploy

1. Push the project to GitHub
2. Open Streamlit Cloud
3. Connect GitHub repo
4. Select app.py
5. Deploy
