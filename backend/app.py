# backend/app.py
import os
import io
import base64
import logging

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv

# mobilenet.py 파일이 app.py와 같은 디렉토리에 있다고 가정
try:
    from mobilenet import CustomMobileNetV3Small # 사용자 정의 모델 클래스
except ImportError:
    logging.error("Failed to import CustomMobileNetV3Small from mobilenet.py. Ensure the file exists and is in the correct path.")
    exit()


# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(threadName)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__, static_folder='../frontend', template_folder='../frontend')
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'change_this_in_production_!@#$')
socketio = SocketIO(app, cors_allowed_origins="*") # 개발 편의상 모든 origin 허용

logger.info(f"--- Application Startup ---")
# ... (이전 디버깅 로그는 간결하게 하거나 필요시 주석 해제) ...
logger.info(f"CWD: {os.getcwd()}")
resolved_static_folder = os.path.abspath(os.path.join(app.root_path, app.static_folder))
logger.info(f"Static folder: {resolved_static_folder} (Exists: {os.path.exists(resolved_static_folder)})")
if os.path.exists(resolved_static_folder): logger.info(f"Static files: {os.listdir(resolved_static_folder)}")
logger.info(f"--- End Startup Info ---")


# --- PyTorch 모델 로딩 및 설정 ---
NUM_CLASSES = 5
CLASS_NAMES = ["Class 1", "Class 2", "Class 3", "Class 4", "Non-target"]
# Non-target 클래스의 인덱스를 명확히 정의
try:
    NON_TARGET_CLASS_ID = CLASS_NAMES.index("Non-target")
except ValueError:
    logger.error("'Non-target' class not found in CLASS_NAMES. Please define it.")
    # 기본값으로 마지막 인덱스 사용 또는 에러 처리
    NON_TARGET_CLASS_ID = NUM_CLASSES - 1 if NUM_CLASSES > 0 else 0


CONFIDENCE_THRESHOLD = 0.9  # 예시: 신뢰도 임계값을 70% (0.7)로 설정
logger.info(f"Confidence threshold set to: {CONFIDENCE_THRESHOLD}")

MODEL_PATH = "./CustomMobileNetV3Small_checkpoint_epoch_15.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

model = CustomMobileNetV3Small(num_classes=NUM_CLASSES) # pretrained 기본값 사용 가정

try:
    model_abs_path = os.path.abspath(MODEL_PATH)
    logger.info(f"Attempting to load model from: {model_abs_path}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    logger.error(f"Model file not found at {model_abs_path}.")
    exit()
except Exception as e:
    logger.error(f"Error loading model state_dict: {e}", exc_info=True)
    exit()

model.to(DEVICE)
model.eval()

# 전처리 파이프라인: 학습 시 사용한 것과 '정확히' 동일해야 합니다.
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
logger.info("Preprocessing pipeline initialized.")

def process_image_and_infer(base64_image_string):
    # logger.debug("Starting image processing and inference...") # 필요시 상세 로그 활성화
    try:
        if ',' in base64_image_string:
            base64_data = base64_image_string.split(',', 1)[1]
        else:
            base64_data = base64_image_string

        image_bytes = base64.b64decode(base64_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_cv is None:
            logger.warning("Failed to decode image with OpenCV from byte array.")
            return {"class_name": "Decode Error (OpenCV)", "class_id": -1, "confidence": 0.0}

        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(img_rgb)
        input_tensor = preprocess(image_pil)
        input_batch = input_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_batch)
            probabilities = torch.softmax(output, dim=1)
            confidence_tensor, predicted_idx_tensor = torch.max(probabilities, 1)

        predicted_class_idx_raw = predicted_idx_tensor.item()
        confidence_score_raw = confidence_tensor.item()

        # logger.debug(f"Raw Prediction: {CLASS_NAMES[predicted_class_idx_raw]} (ID: {predicted_class_idx_raw}), Conf: {confidence_score_raw:.4f}")

        final_predicted_class_idx = predicted_class_idx_raw
        final_predicted_class_name = CLASS_NAMES[final_predicted_class_idx]
        final_confidence_score = confidence_score_raw

        # 신뢰도 임계값 적용
        if confidence_score_raw < CONFIDENCE_THRESHOLD:
            # 특정 클래스(예: Class 1~4)가 아니면서 신뢰도가 낮은 경우 Non-target으로 분류할 수 있음
            # 또는 모든 예측에 대해 신뢰도 미달 시 Non-target으로 분류
            if final_predicted_class_idx != NON_TARGET_CLASS_ID: # 이미 Non-target으로 예측된 경우는 그대로 둠 (선택적 로직)
                logger.info(f"Confidence ({confidence_score_raw:.4f}) for '{CLASS_NAMES[predicted_class_idx_raw]}' is below threshold ({CONFIDENCE_THRESHOLD}). Classifying as 'Non-target'.")
                final_predicted_class_idx = NON_TARGET_CLASS_ID
                final_predicted_class_name = CLASS_NAMES[final_predicted_class_idx]
                # final_confidence_score = confidence_score_raw # 원래 confidence를 보낼 수도 있고
                # final_confidence_score = 0.0 # 아니면 0으로 설정하여 UI에서 명확히 구분
            else:
                 logger.info(f"Predicted as 'Non-target' with confidence {confidence_score_raw:.4f} (Threshold: {CONFIDENCE_THRESHOLD})")
        else:
            logger.info(f"Prediction: {final_predicted_class_name} (ID: {final_predicted_class_idx}), Conf: {final_confidence_score:.4f} (Threshold Met)")


        return {
            "class_name": final_predicted_class_name,
            "class_id": final_predicted_class_idx,
            "confidence": final_confidence_score
        }

    except Exception as e:
        logger.error(f"Error in process_image_and_infer: {e}", exc_info=True)
        return {"class_name": "Processing Error", "class_id": -1, "confidence": 0.0}


@app.route('/')
def index_route():
    logger.debug(f"Serving index.html for request from {request.remote_addr}")
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    logger.info(f'Client connected: {request.sid} from IP: {request.remote_addr}')

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f'Client disconnected: {request.sid}')

@socketio.on('video_frame')
def handle_video_frame(data):
    # logger.debug(f"Received 'video_frame' event from client: {request.sid}") # 너무 자주 찍히므로 INFO로 변경 또는 주석
    image_data_url = data.get('frame')
    if not image_data_url:
        logger.warning(f"Client {request.sid} sent empty frame data.")
        emit('inference_result', {"class_name": "No Frame Data", "class_id": -1, "confidence": 0.0})
        return

    result_dict = process_image_and_infer(image_data_url)
    emit('inference_result', result_dict)
    # logger.debug(f"Emitted 'inference_result' to client {request.sid}: {result_dict}") # 너무 자주 찍히므로 INFO로 변경 또는 주석

if __name__ == '__main__':
    logger.info(f"Attempting to start server...") # HTTP 또는 HTTPS 여부는 run_simple 로그가 알려줌
    cert_path = 'cert.pem'
    key_path = 'key.pem'
    use_https = os.path.exists(cert_path) and os.path.exists(key_path)

    if use_https:
        logger.info(f"SSL certificate and key found. Starting HTTPS server on https://0.0.0.0:5000")
    else:
        logger.warning(f"SSL certificate or key not found. Starting HTTP server on http://0.0.0.0:5000")
        logger.warning(f"Camera functionality might not work over HTTP on mobile browsers.")

    try:
        socketio.run(app,
                     host='0.0.0.0',
                     port=5000,
                     debug=True, # 프로덕션에서는 False
                     use_reloader=False,
                     keyfile=key_path if use_https else None,
                     certfile=cert_path if use_https else None
                    )
    except Exception as e:
        logger.error(f"Failed to start the server: {e}", exc_info=True)
if __name__ == '__main__':
    logger.info(f"Attempting to start HTTPS server on https://0.0.0.0:5000")

    try:
        # eventlet을 사용하도록 명시적으로 지정하고, 로깅 레벨을 올려서 eventlet/socketio 내부 동작도 확인 가능
        # socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False,
        #              keyfile=key_path, certfile=cert_path, engineio_logger=True, log_output=True)
        socketio.run(app,
                     host='0.0.0.0',
                     port=5000,
                     debug=True, # 개발 중에는 True, 프로덕션에서는 False
                     use_reloader=False
                    )
    except Exception as e:
        logger.error(f"Failed to start HTTPS server: {e}", exc_info=True)