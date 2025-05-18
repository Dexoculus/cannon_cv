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
    exit(1) # 오류 발생 시 종료 코드 명시


# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(threadName)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__, static_folder='../frontend', template_folder='../frontend')
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'a_very_strong_secret_key_!@#$_default') # 기본값도 좀 더 강력하게
# Cloudflare Tunnel을 통해 외부에서 접속하므로, 터널의 도메인을 명시하거나, 개발시는 '*'
# 프로덕션에서는 실제 터널 도메인으로 제한하는 것이 좋습니다.
CORS_ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "*") # 환경 변수에서 읽어오도록 변경
socketio = SocketIO(app, cors_allowed_origins=CORS_ALLOWED_ORIGINS, async_mode='threading') # async_mode 명시 (threading, eventlet, gevent 등)

logger.info(f"--- Application Startup ---")
logger.info(f"CWD: {os.getcwd()}")
resolved_static_folder = os.path.abspath(os.path.join(app.root_path, app.static_folder))
logger.info(f"Static folder: {resolved_static_folder} (Exists: {os.path.exists(resolved_static_folder)})")
if os.path.exists(resolved_static_folder):
    logger.info(f"Static files: {os.listdir(resolved_static_folder)}")
logger.info(f"CORS allowed origins: {CORS_ALLOWED_ORIGINS}")
logger.info(f"--- End Startup Info ---")


# --- PyTorch 모델 로딩 및 설정 ---
NUM_CLASSES = 5
CLASS_NAMES = ["Class 1", "Class 2", "Class 3", "Class 4", "Non-target"]
try:
    NON_TARGET_CLASS_ID = CLASS_NAMES.index("Non-target")
except ValueError:
    logger.error("'Non-target' class not found in CLASS_NAMES. Defaulting to last index or exiting.")
    # NON_TARGET_CLASS_ID = NUM_CLASSES - 1 if NUM_CLASSES > 0 else 0 # 기본값 설정
    exit(1) # Non-target 클래스는 중요하므로 정의되지 않으면 종료

CONFIDENCE_THRESHOLD = 0.8
logger.info(f"Confidence threshold set to: {CONFIDENCE_THRESHOLD}")

MODEL_PATH = "./CustomMobileNetV3Small_checkpoint_epoch_12.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

model = CustomMobileNetV3Small(num_classes=NUM_CLASSES)

try:
    model_abs_path = os.path.abspath(MODEL_PATH)
    logger.info(f"Attempting to load model from: {model_abs_path}")
    if not os.path.exists(model_abs_path):
        logger.error(f"Model file not found at {model_abs_path}. Please check the MODEL_PATH environment variable or the file path.")
        exit(1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading model state_dict: {e}", exc_info=True)
    exit(1)

model.to(DEVICE)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
logger.info("Preprocessing pipeline initialized.")

def process_image_and_infer(base64_image_string):
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

        final_predicted_class_idx = predicted_class_idx_raw
        final_predicted_class_name = CLASS_NAMES[final_predicted_class_idx] # Indexing before check for safety
        final_confidence_score = confidence_score_raw

        if confidence_score_raw < CONFIDENCE_THRESHOLD:
            if final_predicted_class_idx != NON_TARGET_CLASS_ID:
                logger.info(f"Confidence ({confidence_score_raw:.4f}) for '{CLASS_NAMES[predicted_class_idx_raw]}' is below threshold ({CONFIDENCE_THRESHOLD}). Classifying as 'Non-target'.")
                final_predicted_class_idx = NON_TARGET_CLASS_ID
                final_predicted_class_name = CLASS_NAMES[final_predicted_class_idx]
            else:
                logger.info(f"Predicted as 'Non-target' with confidence {confidence_score_raw:.4f} (Threshold: {CONFIDENCE_THRESHOLD})")
        else:
            logger.info(f"Prediction: {final_predicted_class_name} (ID: {final_predicted_class_idx}), Conf: {final_confidence_score:.4f} (Threshold Met)")

        return {
            "class_name": final_predicted_class_name,
            "class_id": final_predicted_class_idx,
            "confidence": final_confidence_score
        }

    except IndexError: # CLASS_NAMES 접근 시 발생 가능
        logger.error(f"IndexError during prediction. Predicted index {predicted_class_idx_raw} might be out of bounds for CLASS_NAMES (len: {len(CLASS_NAMES)}).", exc_info=True)
        return {"class_name": "Prediction Index Error", "class_id": -1, "confidence": 0.0}
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
    image_data_url = data.get('frame')
    if not image_data_url:
        logger.warning(f"Client {request.sid} sent empty frame data.")
        emit('inference_result', {"class_name": "No Frame Data", "class_id": -1, "confidence": 0.0})
        return

    result_dict = process_image_and_infer(image_data_url)
    emit('inference_result', result_dict)


if __name__ == '__main__':
    # Cloudflare Tunnel 사용 시 Flask 앱은 HTTP로 실행
    # HTTPS는 Cloudflare에서 처리
    logger.info(f"Starting HTTP server on http://0.0.0.0:5000 for Cloudflare Tunnel")
    # 프로덕션 환경에서는 debug=False로 설정
    # DEBUG_MODE = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    DEBUG_MODE = True # 개발 중에는 True로 설정. 배포 시 False로 변경하거나 환경변수 사용.

    try:
        socketio.run(app,
                     host='0.0.0.0',
                     port=5000,
                     use_reloader=False # 프로덕션에서는 False 권장
                    )
    except Exception as e:
        logger.error(f"Failed to start the server: {e}", exc_info=True)
        exit(1)