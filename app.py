import cv2
import numpy as np
import streamlit as st
import os
import gdown
import time

# Kiểm tra và tải tệp yolov3.weights từ Google Drive nếu chưa tồn tại
weights_file = "yolov3.weights"
if not os.path.exists(weights_file):
    drive_url = "https://drive.google.com/uc?id=11rE4um7BB12mtsgiq-D774qprMaRhjpm"
    st.write("Downloading yolov3.weights from Google Drive...")
    gdown.download(drive_url, weights_file, quiet=False)

# Kiểm tra và tải các tệp cấu hình
config_file = "yolov3.cfg"
classes_file = "yolov3.txt"

if not os.path.exists(config_file) or not os.path.exists(classes_file):
    st.error("Tệp cấu hình hoặc tệp classes không tồn tại. Vui lòng kiểm tra lại!")
    st.stop()

# Đọc các lớp từ tệp
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Tạo màu sắc ngẫu nhiên cho các lớp đối tượng
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Tải mô hình YOLO
try:
    net = cv2.dnn.readNet(weights_file, config_file)
except Exception as e:
    st.error(f"Lỗi khi tải mô hình YOLO: {e}")
    st.stop()

# Hàm lấy các lớp đầu ra từ YOLO
def get_output_layers(net):
    layer_names = net.getLayerNames()
    return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Hàm xử lý YOLO
def detect_objects(frame, object_names):
    if frame is None:
        st.warning("Không nhận được khung hình, bỏ qua xử lý!")
        return frame  # Bỏ qua nếu khung hình là None
    
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

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

    # Áp dụng Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in indices.flatten():
        box = boxes[i]
        x, y, w, h = box
        color = COLORS[class_ids[i]]
        label = str(classes[class_ids[i]])
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return frame

# Streamlit UI
st.title("Object Detection with YOLO")
object_names_input = st.sidebar.text_input('Enter Object Names (comma separated)', 'cell phone,laptop,umbrella')
object_names = [obj.strip().lower() for obj in object_names_input.split(',')]

# Nút bật/tắt camera
camera_index = st.sidebar.selectbox("Select Camera", [0, 1, 2])  # Select camera index
start_camera = st.button("Start Camera")
stop_camera = st.button("Stop Camera")

cap = None  # Declare camera variable outside the if block

if start_camera:
    st.write("Camera is starting...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        st.error("Không thể mở camera. Vui lòng kiểm tra lại thiết bị.")
        st.stop()

    frame_window = st.image([])

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Không nhận được khung hình từ camera.")
            break

        # Xử lý khung hình
        processed_frame = detect_objects(frame, object_names)

        # Hiển thị khung hình đã xử lý
        frame_window.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Kiểm tra xem người dùng đã nhấn Stop Camera chưa
        if stop_camera:
            st.write("Camera is stopping...")
            break

        time.sleep(0.1)  # Pause briefly to simulate frame rate and allow UI updates

    cap.release()
    cv2.destroyAllWindows()
else:
    st.write("Camera is not running.")
