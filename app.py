import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import os
import gdown

# Kiểm tra tệp YOLO
weights_file = "yolov3.weights"
config_file = "yolov3.cfg"
classes_file = "yolov3.txt"

if not os.path.exists(weights_file):
    drive_url = "https://drive.google.com/uc?id=11rE4um7BB12mtsgiq-D774qprMaRhjpm"
    st.write("Downloading yolov3.weights from Google Drive...")
    gdown.download(drive_url, weights_file, quiet=False)

if not os.path.exists(config_file) or not os.path.exists(classes_file):
    st.error("Tệp cấu hình hoặc classes không tồn tại!")
    st.stop()

with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet(weights_file, config_file)

def detect_objects(frame, object_names):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward([net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers()])
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id].lower() in object_names:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = COLORS[class_ids[i]]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return frame


class VideoTransformer(VideoTransformerBase):
    def __init__(self, object_names):
        self.object_names = object_names

    def transform(self, frame):
        frame = frame.to_ndarray(format="bgr24")
        frame = detect_objects(frame, self.object_names)
        return frame


st.title("Object Detection with YOLO")
object_names_input = st.sidebar.text_input('Object Names (comma-separated)', 'person,cell phone,laptop')
object_names = [name.strip().lower() for name in object_names_input.split(",")]

webrtc_streamer(
    key="object-detection",
    video_processor_factory=lambda: VideoTransformer(object_names),
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
)
