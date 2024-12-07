import os
import gdown  # Để tải file từ Google Drive
import numpy as np
import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, VideoFrame

# Kiểm tra và tải tệp yolov3.weights từ Google Drive nếu chưa tồn tại
weights_file = "yolov3.weights"
if not os.path.exists(weights_file):
    drive_url = "https://drive.google.com/uc?id=11rE4um7BB12mtsgiq-D774qprMaRhjpm"
    st.write("Downloading yolov3.weights from Google Drive...")
    gdown.download(drive_url, weights_file, quiet=False)

# Đường dẫn đến tệp âm thanh cảnh báo
alarm_sound = "police.wav"  # Đảm bảo file nằm trong cùng thư mục với mã Python

# Đọc các file cấu hình và class
config_file = "yolov3.cfg"  # Thay bằng đường dẫn phù hợp
classes_file = "yolov3.txt"

# Đọc các lớp từ tệp
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Tạo màu sắc ngẫu nhiên cho các lớp đối tượng
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Tải mô hình YOLO
net = cv2.dnn.readNet(weights_file, config_file)

# Lấy các layer output
def get_output_layers(net):
    layer_names = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()
    if isinstance(unconnected_out_layers, np.ndarray) and unconnected_out_layers.ndim == 1:
        return [layer_names[i - 1] for i in unconnected_out_layers]
    else:
        return [layer_names[i[0] - 1] for i in unconnected_out_layers]

# Streamlit UI
st.title("Object Detection with YOLO")
object_names_input = st.sidebar.text_input('Enter Object Names (comma separated)', 'cell phone,laptop,umbrella')
object_names = [obj.strip().lower() for obj in object_names_input.split(',')]
frame_limit = st.sidebar.slider('Set Frame Limit for Alarm', 1, 10, 3)

# Nhập số lượng vật thể cần giám sát
object_counts_input = {}
for obj in object_names:
    object_counts_input[obj] = st.sidebar.number_input(f'Enter number of {obj} to monitor', min_value=0, value=0, step=1)

# VideoProcessor để xử lý video frame từ camera
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0

    def recv(self, frame: VideoFrame) -> VideoFrame:
        # Chuyển đổi video frame thành ndarray để xử lý
        image = frame.to_ndarray(format="bgr24")
        height, width, channels = image.shape
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        detected_objects = {obj: 0 for obj in object_names}  # Đếm các vật thể đã phát hiện

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

        # Áp dụng NMS (Non-Maximum Suppression) để loại bỏ các bounding boxes dư thừa
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        object_found = False
        if len(indices) > 0:
            object_found = True
            for i in indices.flatten():
                box = boxes[i]
                x, y, w, h = box
                color = COLORS[class_ids[i]]
                label = str(classes[class_ids[i]])
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                # Cập nhật số lượng vật thể đã phát hiện
                detected_objects[classes[class_ids[i]].lower()] += 1

        # Kiểm tra số lượng vật thể theo yêu cầu
        for obj in object_names:
            required_count = object_counts_input[obj]
            current_count = detected_objects[obj]

            # Nếu số lượng vật thể phát hiện ít hơn yêu cầu, hiển thị cảnh báo
            if required_count > 0 and current_count < required_count:
                missing_object = obj
                cv2.putText(image, f"Warning: {missing_object} Missing!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                st.audio(alarm_sound, format="audio/wav", start_time=0)

        return VideoFrame.from_ndarray(image, format="bgr24")

# Tạo WebRTC streamer để hiển thị camera trực tiếp
webrtc_streamer(key="example", video_processor_factory=VideoProcessor)
