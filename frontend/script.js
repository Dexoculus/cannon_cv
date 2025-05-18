document.addEventListener('DOMContentLoaded', () => {
    const videoElement = document.getElementById('liveVideo');
    const canvasElement = document.getElementById('captureCanvas');
    const context = canvasElement.getContext('2d', { willReadFrequently: true });
    const inferenceOverlay = document.getElementById('inferenceOverlay');
    const statusBar = document.getElementById('statusBar');
    const resetCheckboxesBtn = document.getElementById('resetCheckboxesBtn');

    const checkboxes = {
        cbClass1: document.getElementById('cbClass1'), // Corresponds to Class ID 0
        cbClass2: document.getElementById('cbClass2'), // Corresponds to Class ID 1
        cbClass3: document.getElementById('cbClass3'), // Corresponds to Class ID 2
        cbClass4: document.getElementById('cbClass4')  // Corresponds to Class ID 3
    };

    const SERVER_URL = window.location.origin; // Assumes backend is on the same host/port
    const socket = io(SERVER_URL, {
        reconnectionAttempts: 5,
        reconnectionDelay: 3000,
    });

    let streamActive = false;
    const FRAME_SEND_INTERVAL = 500; // ms, 0.5초마다 프레임 전송
    const VIDEO_CONSTRAINTS = {
        audio: false,
        video: {
            facingMode: 'environment',
            width: { ideal: 640 },
            height: { ideal: 480 },
            // frameRate: { ideal: 15 } // Optional: limit frame rate
        }
    };

    async function setupCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia(VIDEO_CONSTRAINTS);
            videoElement.srcObject = stream;
            await videoElement.play();
            streamActive = true;
            statusBar.textContent = '카메라 준비 완료';

            videoElement.onloadedmetadata = () => {
                canvasElement.width = videoElement.videoWidth;
                canvasElement.height = videoElement.videoHeight;
                console.log(`Camera resolution: ${videoElement.videoWidth}x${videoElement.videoHeight}`);
            };
        } catch (err) {
            console.error("카메라 접근 오류:", err);
            inferenceOverlay.textContent = "카메라 오류";
            statusBar.textContent = `카메라 접근 실패: ${err.name}`;
            alert(`카메라에 접근할 수 없습니다: ${err.message}\nHTTPS 환경인지, 카메라 권한이 허용되었는지 확인해주세요.`);
        }
    }

    function sendFrame() {
        if (streamActive && videoElement.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && videoElement.videoWidth > 0) {
            context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
            const frameDataURL = canvasElement.toDataURL('image/jpeg', 0.7); // JPEG, quality 0.7
            socket.emit('video_frame', { frame: frameDataURL });
        }
    }

    let frameIntervalId = null;

    socket.on('connect', () => {
        console.log('서버 연결 성공 (ID:', socket.id, ')');
        statusBar.textContent = '서버 연결됨';
        if (streamActive && !frameIntervalId) { // 스트림 활성화 상태고, 인터벌이 아직 없으면 시작
            frameIntervalId = setInterval(sendFrame, FRAME_SEND_INTERVAL);
        }
    });

    socket.on('disconnect', (reason) => {
        console.warn('서버 연결 끊김:', reason);
        statusBar.textContent = '서버 연결 끊김. 재시도 중...';
        if (frameIntervalId) {
            clearInterval(frameIntervalId);
            frameIntervalId = null;
        }
    });

    socket.on('connect_error', (err) => {
        console.error('서버 연결 오류:', err);
        statusBar.textContent = `서버 연결 오류: ${err.message}`;
        if (frameIntervalId) {
            clearInterval(frameIntervalId);
            frameIntervalId = null;
        }
    });

    socket.on('inference_result', (data) => {
        const { class_name, class_id } = data;

        // 1. 실시간 추론 결과 오버레이 업데이트
        inferenceOverlay.textContent = `결과: ${class_name || "N/A"}`;

        // 2. 검출 이력 누적 (체크박스)
        //    Class ID 0~3 (Class 1~4)에 대해서만 체크박스 업데이트
        if (class_id >= 0 && class_id <= 3) {
            const targetCheckboxKey = `cbClass${class_id + 1}`; // e.g., class_id 0 -> cbClass1
            if (checkboxes[targetCheckboxKey]) {
                checkboxes[targetCheckboxKey].checked = true; // 누적
            }
        }
        // Non-target (class_id 4) 또는 에러 (-1)는 체크박스 상태에 영향을 주지 않음.
    });

    // 3. 명시적 리셋 기능
    if (resetCheckboxesBtn) {
        resetCheckboxesBtn.addEventListener('click', () => {
            Object.values(checkboxes).forEach(cb => cb.checked = false);
            // inferenceOverlay는 현재 프레임 결과로 계속 업데이트되므로, 리셋 시 텍스트 변경은 선택사항.
            // inferenceOverlay.textContent = "결과: 이력 리셋됨";
            console.log("감지 이력(체크박스)이 사용자에 의해 리셋되었습니다.");
            // 만약 리셋 시 오버레이도 "대기 중..."으로 바꾸고 싶다면,
            // inferenceOverlay.textContent = "결과: 대기 중...";
            // 이 경우, 다음 inference_result가 오기 전까지 "대기 중"으로 보입니다.
        });
    }

    // 페이지 로드 시 카메라 시작
    setupCamera();

    // 페이지 unload 시 정리 (선택 사항이지만 좋은 습관)
    window.addEventListener('beforeunload', () => {
        if (socket.connected) {
            socket.disconnect();
        }
        if (videoElement.srcObject) {
            const tracks = videoElement.srcObject.getTracks();
            tracks.forEach(track => track.stop());
        }
        if (frameIntervalId) {
            clearInterval(frameIntervalId);
        }
    });
});