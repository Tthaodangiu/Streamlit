import cv2
import numpy as np
import streamlit as st
import os
import gdown

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

# Nhận đầu vào từ camera của trình duyệt
camera_input = st.camera_input("Capture Image")

if camera_input:
    # Đọc ảnh từ camera
    image = cv2.imdecode(np.frombuffer(camera_input.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    
    # Xử lý ảnh với YOLO
    processed_frame = detect_objects(image, object_names)

    # Hiển thị ảnh đã xử lý
    st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB")
else:
    st.write("Vui lòng cấp phép sử dụng camera hoặc tải ảnh lên.")
