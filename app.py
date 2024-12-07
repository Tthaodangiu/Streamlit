import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
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
def detect_objects(frame, object_names, frame_limit, object_counts_input):
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
    detected_objects = {obj: 0 for obj in object_names}

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
        detected_objects[classes[class_ids[i]].lower()] += 1

    return frame

# Xác định lớp xử lý video
class VideoTransformer(VideoTransformerBase):
    def __init__(self, object_names, frame_limit, object_counts_input):
        self.object_names = object_names
        self.frame_limit = frame_limit
        self.object_counts_input = object_counts_input

    def transform(self, frame):
        if frame is None:
            st.error("Không nhận được khung hình từ camera.")
            return None

        try:
            # Chuyển khung hình từ WebRTC về dạng numpy array (BGR)
            frame = frame.to_ndarray(format="bgr24")

            # Xử lý khung hình với YOLO
            processed_frame = detect_objects(frame, self.object_names, self.frame_limit, self.object_counts_input)

            # Trả về khung hình đã xử lý
            return processed_frame
        except Exception as e:
            st.error(f"Lỗi trong quá trình xử lý khung hình: {e}")
            return frame

# Streamlit UI
st.title("Object Detection with YOLO")
object_names_input = st.sidebar.text_input('Enter Object Names (comma separated)', 'cell phone,laptop,umbrella')
object_names = [obj.strip().lower() for obj in object_names_input.split(',')]
frame_limit = st.sidebar.slider('Set Frame Limit for Alarm', 1, 10, 3)

# Nhập số lượng vật thể cần giám sát
object_counts_input = {}
for obj in object_names:
    object_counts_input[obj] = st.sidebar.number_input(f'Enter number of {obj} to monitor', min_value=0, value=0, step=1)

# Thêm TURN Server cho môi trường đám mây
TURN_SERVER = {
    "urls": "turn:your_turn_server_url",  # Thay thế bằng TURN server của bạn
    "username": "your_username",
    "credential": "your_password",
}

# Cấu hình ICE server
rtc_configuration = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        TURN_SERVER,  # Thêm TURN server
    ]
}

# Khởi chạy camera với streamlit-webrtc
webrtc_ctx = webrtc_streamer(
    key="object-detection",
    video_processor_factory=lambda: VideoTransformer(object_names, frame_limit, object_counts_input),
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"video": True, "audio": False},  # Chỉ bật video
)

# Kiểm tra trạng thái camera
if webrtc_ctx and webrtc_ctx.state.playing:
    st.success("Camera đang hoạt động.")
    # Kiểm tra trạng thái ICE nếu có thuộc tính
    if hasattr(webrtc_ctx, "ice_connection_state"):
        st.write("ICE Connection State: ", webrtc_ctx.ice_connection_state)
else:
    st.warning("Không thể hiển thị video. Vui lòng kiểm tra kết nối hoặc cấu hình TURN Server.")
    if webrtc_ctx:
        st.write("WebRTC State: ", webrtc_ctx.state)
    else:
        st.error("WebRTC context không được khởi tạo.")
