import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.set_page_config(page_title="Emotion Detection")

class EmotionProcessor(VideoProcessorBase):

    def recv(self, frame):

        global tracker
        global tracking
        global locked_label
        global locked_conf
        global locked_cheek
        global cheek_smooth
        global change_counter

        frame = frame.to_ndarray(format="bgr24")

        frame = cv2.flip(frame, 1)

        small = cv2.resize(frame, (256,192))

        rgb = cv2.cvtColor(
            small,
            cv2.COLOR_BGR2RGB
        )

        res = mp_face_mesh.process(rgb)

        if res.multi_face_landmarks:

            lm = res.multi_face_landmarks[0].landmark

            h, w, _ = small.shape

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

            else:

                tracking = False
                tracker = None

            if show_left or show_right:

                if not tracking:

                    pts = np.array([
                        [int(p.x*w), int(p.y*h)]
                        for p in lm
                    ])

                    x1, y1 = pts.min(axis=0)
                    x2, y2 = pts.max(axis=0)

                    box = (
                        x1,
                        y1,
                        x2 - x1,
                        y2 - y1
                    )

                    tracker = cv2.TrackerKCF_create()

                    tracker.init(
                        small,
                        box
                    )

                    tracking = True

                ok, box = tracker.update(small)

                if ok:

                    x, y, bw, bh = map(int, box)

                    face = small[y:y+bh, x:x+bw]

                    if face.size != 0:

                        pil = Image.fromarray(
                            cv2.cvtColor(
                                face,
                                cv2.COLOR_BGR2RGB
                            )
                        ).resize((224,224))

                        x_t = eval_tf(pil).unsqueeze(0).to(DEVICE)

                        with torch.no_grad():

                            emb = resnet(x_t)

                        emb = emb.cpu().numpy().squeeze()

                        pts = np.array([
                            [int(p.x*w), int(p.y*h)]
                            for p in lm
                        ])

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

                        if cheek_smooth is None:
                            cheek_smooth = cheek_raw
                        else:
                            cheek_smooth = (
                                0.8 * cheek_smooth +
                                0.2 * cheek_raw
                            )

                        fused = np.concatenate([
                            emb,
                            lm_flat,
                            [cheek_smooth]
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

                        sad_idx = EMOTIONS.index('sad')
                        angry_idx = EMOTIONS.index('angry')

                        if probs[sad_idx] > probs[angry_idx]:
                            probs[sad_idx] *= 1.08
                        else:
                            probs[angry_idx] *= 1.08

                        probs = probs / np.sum(probs)

                        idx = np.argmax(probs)

                        conf = float(probs[idx])

                        label = EMOTIONS[idx]

                        if locked_label is None:

                            locked_label = label
                            locked_conf = conf
                            locked_cheek = cheek_smooth

                        else:

                            if label != locked_label:

                                if conf > 0.6:
                                    change_counter += 1

                            else:

                                change_counter = max(
                                    change_counter - 1,
                                    0
                                )

                            if (
                                change_counter >= SWITCH_THRESHOLD
                                or
                                conf > locked_conf + CONF_BOOST_THRESHOLD
                            ):

                                locked_label = label
                                locked_conf = conf
                                locked_cheek = cheek_smooth

                                change_counter = 0

                        final_label = locked_label
                        final_conf = locked_conf
                        final_cheek = locked_cheek

                        sx = frame.shape[1] / 256
                        sy = frame.shape[0] / 192

                        x1 = int(x * sx)
                        y1 = int(y * sy)

                        x2 = int((x+bw) * sx)
                        y2 = int((y+bh) * sy)

                        cv2.rectangle(
                            frame,
                            (x1,y1),
                            (x2,y2),
                            (0,255,0),
                            2
                        )

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
                            f"{final_label} ({final_conf*100:.1f}%)",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255,255,255),
                            2
                        )

        return av.VideoFrame.from_ndarray(
            frame,
            format="bgr24"
        )

st.title("Realtime Emotion Detection")

webrtc_streamer(
    key="emotion",
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)
