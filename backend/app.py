import os
import io
import base64
import logging

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2 # OpenCV

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv # .env 파일 사용 시

from mobilenet import CustomMobileNetV3Small

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__, static_folder='../frontend', template_folder='../frontend')
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'super-secret-key!')
socketio = SocketIO(app, cors_allowed_origins="*") # 개발 중에는 모든 origin 허용

print(f"Current working directory: {os.getcwd()}")
print(f"App static folder: {app.static_folder}")
print(f"Resolved static folder path: {os.path.abspath(app.static_folder)}")
print(f"Does static folder exist? {os.path.exists(os.path.abspath(app.static_folder))}")
if os.path.exists(os.path.abspath(app.static_folder)):
    print(f"Files in static folder: {os.listdir(os.path.abspath(app.static_folder))}")

# --- PyTorch 모델 로딩 설정 ---
NUM_CLASSES = 5
# 클래스 이름: 0~3번 인덱스는 Target, 4번 인덱스는 Non-target
CLASS_NAMES = ["Class 1", "Class 2", "Class 3", "Class 4", "Non-target"]
MODEL_PATH = "./CustomMobileNetV3Small_checkpoint_epoch_15.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CustomMobileNetV3Small(num_classes=NUM_CLASSES) # 모델 클래스 인스턴스 생성

# 모델 가중치 로드
try:
    # GPU에서 학습된 모델을 CPU에서 로드하거나 그 반대의 경우 map_location 사용
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    logger.error(f"Model file not found at {MODEL_PATH}. Please check the path.")
    # 적절한 에러 처리 또는 기본 모델 사용 로직 추가 가능
    exit() # 모델이 없으면 실행 중단
except Exception as e:
    logger.error(f"Error loading model state_dict: {e}")
    logger.warning("This might be due to a mismatch between the model architecture and the state_dict.")
    logger.warning("Ensure YourCustomModel (or the model defined here) matches the one used for training.")
    exit()

model.to(DEVICE)
model.eval() # 추론 모드로 설정

preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])

def process_image_and_infer(base64_image_string):
    try:
        # Base64 헤더 제거 (e.g., "data:image/jpeg;base64,")
        if ',' in base64_image_string:
            base64_data = base64_image_string.split(',', 1)[1]
        else:
            base64_data = base64_image_string

        image_bytes = base64.b64decode(base64_data)

        # Bytes -> OpenCV (numpy array) -> RGB
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # BGR
        if img_cv is None:
            logger.error("Failed to decode image with OpenCV.")
            return "Decode Error", -1
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        # Numpy array (RGB) -> PIL Image
        image_pil = Image.fromarray(img_rgb)

        # 전처리 및 배치 차원 추가
        input_tensor = preprocess(image_pil)
        input_batch = input_tensor.unsqueeze(0).to(DEVICE) # (1, C, H, W)

        with torch.no_grad(): # 기울기 계산 비활성화
            output = model(input_batch)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class_idx = predicted_idx.item()
        predicted_class_name = CLASS_NAMES[predicted_class_idx]
        confidence_score = confidence.item()

        logger.info(f"Prediction: {predicted_class_name} (ID: {predicted_class_idx}), Confidence: {confidence_score:.4f}")
        return predicted_class_name, predicted_class_idx

    except Exception as e:
        logger.error(f"Error during image processing or inference: {e}", exc_info=True)
        return "Processing Error", -1


@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    logger.info(f'Client connected: {request.sid}')

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f'Client disconnected: {request.sid}')

@socketio.on('video_frame')
def handle_video_frame(data):
    # logger.debug(f"Received video_frame from {request.sid}")
    image_data_url = data.get('frame')
    if not image_data_url:
        emit('inference_result', {'class_name': 'No Frame Data', 'class_id': -1})
        return

    predicted_class_name, predicted_class_idx = process_image_and_infer(image_data_url)

    emit('inference_result', {
        'class_name': predicted_class_name,
        'class_id': predicted_class_idx  # 0-3은 타겟, 4는 non-target
    })

if __name__ == '__main__':
    logger.info(f"Starting server on http://0.0.0.0:5000 with device: {DEVICE}")
    # 개발 시: debug=True, use_reloader=True (단, reloader는 가끔 문제를 일으킬 수 있음)
    # 프로덕션에서는 Gunicorn + eventlet/gevent 사용 권장
    # 예: gunicorn -k eventlet -w 1 app:app -b 0.0.0.0:5000
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)