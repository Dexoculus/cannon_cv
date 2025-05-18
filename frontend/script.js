// frontend/script.js

console.log("script.js execution started!"); // 스크립트 파일 실행 시작 로그

document.addEventListener('DOMContentLoaded', () => {
    console.log("DOMContentLoaded event fired!"); // DOM 로드 완료 로그

    const videoElement = document.getElementById('liveVideo');
    const canvasElement = document.getElementById('captureCanvas');
    const context = canvasElement.getContext('2d', { willReadFrequently: true });
    const inferenceOverlay = document.getElementById('inferenceOverlay');
    const statusBar = document.getElementById('statusBar');
    const resetCheckboxesBtn = document.getElementById('resetCheckboxesBtn');

    const checkboxes = {
        cbClass1: document.getElementById('cbClass1'),
        cbClass2: document.getElementById('cbClass2'),
        cbClass3: document.getElementById('cbClass3'),
        cbClass4: document.getElementById('cbClass4')
    };

    console.log("Attempting to connect to Socket.IO server...");
    const SERVER_URL = window.location.origin;
    console.log("Server URL for Socket.IO:", SERVER_URL);
    let socket; // socket 변수를 try 블록 외부에서 선언

    try {
        socket = io(SERVER_URL, {
            // transports: ['websocket', 'polling'] // 필요시 명시적 지정
        });
        console.log("Socket.IO client initialized (io function called). Waiting for 'connect' event...");
    } catch (e) {
        console.error("Error initializing Socket.IO client:", e);
        statusBar.textContent = "Socket.IO 초기화 오류";
        // Socket 초기화 실패 시 이후 로직 진행 불가
        return;
    }


    let streamActive = false;
    const FRAME_SEND_INTERVAL = 500;
    const VIDEO_CONSTRAINTS = {
        audio: false,
        video: {
            facingMode: 'environment',
            width: { ideal: 640 },
            height: { ideal: 480 },
        }
    };

    async function setupCamera() {
        console.log("Attempting to setup camera...");
        try {
            const stream = await navigator.mediaDevices.getUserMedia(VIDEO_CONSTRAINTS);
            videoElement.srcObject = stream;
            await videoElement.play();
            streamActive = true;
            statusBar.textContent = '카메라 준비 완료';
            console.log("Camera setup successful, stream active.");

            videoElement.onloadedmetadata = () => {
                canvasElement.width = videoElement.videoWidth;
                canvasElement.height = videoElement.videoHeight;
                console.log(`Camera resolution: ${videoElement.videoWidth}x${videoElement.videoHeight}`);
                // 스트림 활성화 및 메타데이터 로드 후 프레임 전송 시작
                if (socket.connected && !frameIntervalId) { // Socket 연결 상태도 확인
                     console.log("Camera ready and socket connected, starting frame interval.");
                    frameIntervalId = setInterval(sendFrame, FRAME_SEND_INTERVAL);
                } else if (!socket.connected) {
                    console.warn("Camera ready, but socket not connected yet. Frame interval will start upon connection.");
                }
            };
        } catch (err) {
            console.error("카메라 접근 오류:", err);
            inferenceOverlay.textContent = "카메라 오류";
            statusBar.textContent = `카메라 접근 실패: ${err.name}`;
            alert(`카메라에 접근할 수 없습니다: ${err.message}\nHTTPS 환경인지, 카메라 권한이 허용되었는지 확인해주세요.`);
            streamActive = false; // 스트림 활성화 실패 명시
        }
    }

    function sendFrame() {
        if (streamActive && videoElement.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && videoElement.videoWidth > 0) {
            // console.log("Attempting to send frame..."); // 너무 자주 찍히므로 필요시 주석 해제
            context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
            const frameDataURL = canvasElement.toDataURL('image/jpeg', 0.7);
            // console.log("Frame Data URL (first 100 chars):", frameDataURL.substring(0,100)); // 데이터 확인용
            socket.emit('video_frame', { frame: frameDataURL });
            // console.log("Frame sent via socket.emit('video_frame')"); // 너무 자주 찍히므로 필요시 주석 해제
        } else {
            // console.warn("Conditions not met for sending frame. streamActive:", streamActive, "readyState:", videoElement.readyState, "videoWidth:", videoElement.videoWidth);
        }
    }

    let frameIntervalId = null;

    socket.on('connect', () => {
        console.log('서버 연결 성공 (ID:', socket.id, ')');
        statusBar.textContent = '서버 연결됨';
        // 카메라가 이미 준비되었고, 인터벌이 아직 시작되지 않았다면 프레임 전송 시작
        if (streamActive && videoElement.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && !frameIntervalId) {
            console.log("Socket connected, camera already active, starting frame interval.");
            frameIntervalId = setInterval(sendFrame, FRAME_SEND_INTERVAL);
        } else if (!streamActive) {
            console.warn("Socket connected, but camera stream not active yet.");
        }
    });

    socket.on('disconnect', (reason) => {
        console.warn('서버 연결 끊김:', reason);
        statusBar.textContent = '서버 연결 끊김. 재시도 중...';
        if (frameIntervalId) {
            clearInterval(frameIntervalId);
            frameIntervalId = null;
            console.log("Frame interval cleared due to disconnect.");
        }
    });

    socket.on('connect_error', (err) => {
        console.error('서버 연결 오류:', err);
        statusBar.textContent = `서버 연결 오류: ${err.message}`;
        if (frameIntervalId) {
            clearInterval(frameIntervalId);
            frameIntervalId = null;
            console.log("Frame interval cleared due to connection error.");
        }
    });

    socket.on('inference_result', (data) => {
        console.log("Received 'inference_result' from server:", data); // ★★★ 결과 수신 로그 ★★★
        const { class_name, class_id } = data;
        inferenceOverlay.textContent = `결과: ${class_name || "N/A"}`;

        if (class_id >= 0 && class_id <= 3) {
            const targetCheckboxKey = `cbClass${class_id + 1}`;
            if (checkboxes[targetCheckboxKey]) {
                checkboxes[targetCheckboxKey].checked = true;
            }
        }
    });

    if (resetCheckboxesBtn) {
        resetCheckboxesBtn.addEventListener('click', () => {
            console.log("Reset button clicked by user.");
            Object.values(checkboxes).forEach(cb => cb.checked = false);
            inferenceOverlay.textContent = "결과: 이력 리셋됨";
        });
    }

    // 페이지 로드 시 카메라 시작
    setupCamera();

    window.addEventListener('beforeunload', () => {
        console.log("beforeunload event triggered. Disconnecting socket and stopping tracks.");
        if (socket && socket.connected) {
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