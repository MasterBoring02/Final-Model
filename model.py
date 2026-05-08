import streamlit as st
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
