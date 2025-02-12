import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import time

# Load YOLO model
model = YOLO("best.pt")
class_names = model.names

# Streamlit UI
st.set_page_config(layout="wide")
st.title("Pothole Detection System")

# Sidebar Configuration
st.sidebar.header("Select Task")
task = st.sidebar.radio("Pothole Detection", ["Image", "Video", "Webcam"])

uploaded_file = st.sidebar.file_uploader("Upload File", type=["jpg", "jpeg", "png", "mp4"])


def detect_potholes(img):
    img_resized = cv2.resize(img, (1020, 500))
    results = model.predict(img_resized)
    pothole_detected = False

    for r in results:
        boxes = r.boxes
        masks = r.masks

        if masks is not None:
            masks = masks.data.cpu()
            for seg, box in zip(masks.data.cpu().numpy(), boxes):
                seg = cv2.resize(seg, (1020, 500))
                contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    d = int(box.cls)
                    c = class_names[d]
                    x, y, x1, y1 = cv2.boundingRect(contour)
                    cv2.polylines(img_resized, [contour], True, (0, 0, 255), 2)
                    cv2.putText(img_resized, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    pothole_detected = True

    return img_resized, pothole_detected


if task == "Image":
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_np = np.array(image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Detect Potholes"):
            detected_img, pothole_detected = detect_potholes(img_np)
            st.image(detected_img, caption="Detection Result", use_column_width=True)

            if pothole_detected:
                st.markdown("""<script>document.body.style.animation = "blink 0.5s 1"</script>""",
                            unsafe_allow_html=True)
                st.error("Pothole Detected 🚨")
            else:
                st.success("No Potholes Found ✅")

elif task == "Video":
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detected_frame, pothole_detected = detect_potholes(frame)
            stframe.image(detected_frame, channels="BGR")
            time.sleep(0.03)
        cap.release()
        st.success("Video Processing Complete")

elif task == "Webcam":
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detected_frame, pothole_detected = detect_potholes(frame)
        stframe.image(detected_frame, channels="BGR")
        time.sleep(0.03)

        if pothole_detected:
            st.error("Pothole Detected 🚨")
        else:
            st.success("No Potholes Found ✅")

    cap.release()
    st.success("Webcam Stream Stopped")
