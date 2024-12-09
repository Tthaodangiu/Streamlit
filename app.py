import cv2
import numpy as np
import streamlit as st
import os
import gdown
from datetime import timedelta

# Tải YOLO weights và config nếu chưa có
weights_file = "yolov3.weights"
config_file = "yolov3.cfg"
classes_file = "yolov3.txt"

if not os.path.exists(weights_file):
    gdown.download("https://drive.google.com/uc?id=11rE4um7BB12mtsgiq-D774qprMaRhjpm", weights_file, quiet=False)

if not os.path.exists(config_file) or not os.path.exists(classes_file):
    st.error("Missing YOLO config or classes file.")
    st.stop()

# Đọc danh sách các lớp
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Tải mô hình YOLO
net = cv2.dnn.readNet(weights_file, config_file)

# Hàm lấy layer đầu ra
def get_output_layers(net):
    layer_names = net.getLayerNames()
    return [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Giao diện Streamlit
st.title("Object Detection with YOLO")

# Thanh bên trái để nhập thông tin
st.sidebar.header("Settings")
object_names_input = st.sidebar.text_input("Enter Object Names (comma separated)", "cell phone,laptop,umbrella")
object_names = [obj.strip().lower() for obj in object_names_input.split(',')]
monitor_counts = {}
lost_objects_frame = {}  # Đếm frame mất
alerted_objects = set()

for obj in object_names:
    monitor_counts[obj] = st.sidebar.number_input(f"Enter number of {obj} to monitor", min_value=0, value=1, step=1)

frame_limit_seconds = st.sidebar.slider("Set Frame Limit for Alarm (seconds)", 1, 10, 3)

# Chọn nguồn video
video_source = st.radio("Choose Video Source", ["Upload File"])
temp_video_path = "temp_video.mp4"

start_button = st.button("Start Detection")
stop_button = st.button("Stop and Delete Video")

cap = None

# Đọc file âm thanh cảnh báo
def play_alert_sound():
    alert_audio_file = '/mnt/data/police.wav'
    if os.path.exists(alert_audio_file):
        with open(alert_audio_file, 'rb') as f:
            audio_bytes = f.read()
            st.audio(audio_bytes, format='audio/wav')

# Xử lý video từ nguồn
if video_source == "Upload File":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success("Video uploaded successfully!")
        cap = cv2.VideoCapture(temp_video_path)

if cap is not None and start_button:
    stframe = st.empty()
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_limit = int(fps * frame_limit_seconds)  # Tính ngưỡng frame mất

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Video ended or no frames available.")
            break

        height, width, _ = frame.shape
        detected_objects = {}

        # Phát hiện vật thể
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        boxes = []
        class_ids = []
        confidences = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x, center_y, w, h = int(detection[0] * width), int(detection[1] * height), \
                                               int(detection[2] * width), int(detection[3] * height)
                    x, y = center_x - w // 2, center_y - h // 2
                    boxes.append([x, y, w, h])
                    class_ids.append(class_id)
                    confidences.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]].lower()
                color = COLORS[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if label in detected_objects:
                    detected_objects[label] += 1
                else:
                    detected_objects[label] = 1

        # Kiểm tra vật thể mất
        for obj in object_names:
            required_count = monitor_counts[obj]
            current_count = detected_objects.get(obj, 0)

            if current_count < required_count:
                lost_objects_frame[obj] = lost_objects_frame.get(obj, 0) + 1
                if lost_objects_frame[obj] >= frame_limit and obj not in alerted_objects:
                    alerted_objects.add(obj)
                    lost_time_str = str(timedelta(seconds=int(lost_objects_frame[obj] / fps)))
                    st.warning(f"⚠️ ALERT: '{obj}' is missing for {lost_time_str}!")
                    play_alert_sound()
            else:
                lost_objects_frame.pop(obj, None)
                alerted_objects.discard(obj)

        stframe.image(frame, channels="BGR", use_container_width=True)

if stop_button:
    if cap:
        cap.release()
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    st.success("Video stopped and temporary file deleted.")
