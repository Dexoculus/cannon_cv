// frontend/script.js
console.log("script.js execution started! Version: 20250520_1000_UNIQUE_TEST");

document.addEventListener('DOMContentLoaded', () => {
    console.log("DOMContentLoaded event fired!");

    const videoElement = document.getElementById('liveVideo');
    const canvasElement = document.getElementById('captureCanvas');
    const context = canvasElement.getContext('2d', { willReadFrequently: true });
    const inferenceOverlay = document.getElementById('inferenceOverlay');
    const statusBar = document.getElementById('statusBar');
    const resetCheckboxesBtn = document.getElementById('resetCheckboxesBtn');
    const startCameraButton = document.getElementById('startCameraButton'); // 카메라 시작 버튼

    const checkboxes = {
        cbClass1: document.getElementById('cbClass1'),
        cbClass2: document.getElementById('cbClass2'),
        cbClass3: document.getElementById('cbClass3'),
        cbClass4: document.getElementById('cbClass4')
    };

    let socket;
    let stream = null; // 미디어 스트림 객체
    let streamActive = false;
    let isCameraReady = false; // 비디오가 실제로 재생 준비되고 재생 중일 때 true
    let isSocketConnected = false;
    let frameIntervalId = null;

    const FRAME_SEND_INTERVAL = 500; // ms
    const VIDEO_CONSTRAINTS = {
        audio: false,
        video: {
            facingMode: 'environment',
            width: { ideal: 640 },
            height: { ideal: 480 },
        }
    };

    // --- Socket.IO 초기화 ---
    try {
        const SERVER_URL = window.location.origin;
        console.log("Attempting to connect to Socket.IO server at:", SERVER_URL);
        socket = io(SERVER_URL, {
            transports: ['websocket', 'polling']
        });
        statusBar.textContent = "서버 연결 시도 중...";
        console.log("Socket.IO client initialized. Waiting for 'connect' event...");
    } catch (e) {
        console.error("Fatal Error: Initializing Socket.IO client failed:", e);
        statusBar.textContent = "Socket.IO 초기화 치명적 오류: " + e.message;
        inferenceOverlay.textContent = "오류: 새로고침 해주세요.";
        if(startCameraButton) startCameraButton.disabled = true; // 버튼 비활성화
        return;
    }

    // --- 프레임 전송 시작/중지 함수 ---
    function tryStartFrameSending() {
        console.log(`[tryStartFrameSending] Conditions: isCameraReady=${isCameraReady}, isSocketConnected=${isSocketConnected}, frameIntervalId=${frameIntervalId}`);
        if (isCameraReady && isSocketConnected && !frameIntervalId) {
            console.log("[tryStartFrameSending] All conditions met. Starting frame interval.");
            statusBar.textContent = "카메라 및 서버 준비 완료. 프레임 전송 시작.";
            frameIntervalId = setInterval(sendFrame, FRAME_SEND_INTERVAL);
        } else if (frameIntervalId) {
            console.log("[tryStartFrameSending] Frame interval already active.");
        } else {
            let statusMsg = "대기 중: ";
            if (!isCameraReady) statusMsg += "카메라 미준비, ";
            if (!isSocketConnected) statusMsg += "서버 미연결, ";
            statusBar.textContent = statusMsg.slice(0, -2) + "."; // 마지막 쉼표와 공백 제거
            console.log("[tryStartFrameSending] Conditions not yet fully met to start frame interval.");
        }
    }

    function stopFrameSending(reason = "Unknown reason") {
        if (frameIntervalId) {
            clearInterval(frameIntervalId);
            frameIntervalId = null;
            console.log(`[stopFrameSending] Frame interval cleared. Reason: ${reason}`);
        }
    }

    // --- 카메라 설정 함수 (스트림 얻기 및 이벤트 핸들러 등록) ---
    async function initializeCamera() {
        console.log("[initializeCamera] Attempting to get user media (camera stream)...");
        statusBar.textContent = "카메라 접근 시도 중...";
        try {
            stream = await navigator.mediaDevices.getUserMedia(VIDEO_CONSTRAINTS);
            console.log("[initializeCamera] getUserMedia successful. Stream obtained.");
            videoElement.srcObject = stream;

            videoElement.onloadedmetadata = () => {
                console.log("[videoElement.onloadedmetadata] Event fired.");
                canvasElement.width = videoElement.videoWidth;
                canvasElement.height = videoElement.videoHeight;
                console.log(`[videoElement.onloadedmetadata] Canvas resolution set to: ${videoElement.videoWidth}x${videoElement.videoHeight}`);
                if (videoElement.videoWidth === 0 || videoElement.videoHeight === 0) {
                    console.warn("[videoElement.onloadedmetadata] Warning: Video dimensions are zero. Stream might not be fully ready.");
                    statusBar.textContent = "카메라 해상도 인식 중...";
                    isCameraReady = false; // 아직 준비 안됨
                    return;
                }
                console.log("[videoElement.onloadedmetadata] Metadata loaded. Ready for videoElement.play().");
                statusBar.textContent = "카메라 메타데이터 로드 완료. 재생 가능.";
                // isCameraReady는 videoElement.play() 성공 후 onplaying에서 설정
            };

            videoElement.onplaying = () => {
                console.log("[videoElement.onplaying] Video has started playing.");
                streamActive = true;
                isCameraReady = true; // 실제 재생이 시작되면 카메라가 준비된 것으로 간주
                console.log("[videoElement.onplaying] Camera is ready and playing.");
                statusBar.textContent = '카메라 재생 중.';
                tryStartFrameSending(); // 여기서 프레임 전송 시도
            };

            videoElement.onpause = () => {
                console.log("[videoElement.onpause] Video paused.");
                streamActive = false; // 스트림은 활성 상태가 아님
                // isCameraReady는 false로 하지 않음. 다시 play하면 되므로.
                stopFrameSending("Video paused by user or system");
                statusBar.textContent = "카메라 일시 정지됨.";
            };

            videoElement.onstalled = () => {
                console.warn("[videoElement.onstalled] Media data is not available or download has stalled.");
                statusBar.textContent = "카메라 데이터 수신 지연...";
            };

            videoElement.onerror = (e) => {
                console.error("[videoElement.onerror] Video element error:", e);
                const err = videoElement.error;
                statusBar.textContent = `비디오 요소 오류: ${err ? err.message : '알 수 없는 오류'}`;
                streamActive = false;
                isCameraReady = false;
                stopFrameSending("Video element error");
            };
            console.log("[initializeCamera] Camera stream assigned to video element. Waiting for videoElement.play().");
            return true; // 초기화 성공

        } catch (err) {
            console.error("[initializeCamera] Error accessing camera:", err);
            let userMessage = `카메라 접근 실패: ${err.name}. `;
            if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") userMessage += "카메라 장치를 찾을 수 없습니다.";
            else if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") userMessage += "카메라 접근 권한이 거부되었습니다.";
            else if (err.name === "NotReadableError" || err.name === "TrackStartError") userMessage += "카메라를 사용할 수 없습니다 (다른 앱 사용 중?).";
            else userMessage += "알 수 없는 오류입니다.";
            
            inferenceOverlay.textContent = "카메라 오류";
            statusBar.textContent = userMessage;
            streamActive = false;
            isCameraReady = false;
            return false; // 초기화 실패
        }
    }

    // --- 프레임 전송 함수 ---
    function sendFrame() {
        if (streamActive && !videoElement.paused && videoElement.readyState >= HTMLMediaElement.HAVE_ENOUGH_DATA && videoElement.videoWidth > 0 && videoElement.videoHeight > 0) {
            context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
            const frameDataURL = canvasElement.toDataURL('image/jpeg', 0.7);

            if (socket && socket.connected) {
                socket.emit('video_frame', { frame: frameDataURL });
                // console.log("[sendFrame] Frame sent.");
            } else {
                console.warn("[sendFrame] Socket not connected, frame not sent. Stopping frame sending.");
                statusBar.textContent = "서버 연결 끊김. 프레임 전송 중단.";
                stopFrameSending("Socket not connected during sendFrame");
            }
        } else {
            // console.warn(`[sendFrame] Conditions NOT met. streamActive=${streamActive}, paused=${videoElement.paused}, readyState=${videoElement.readyState}, videoWidth=${videoElement.videoWidth}, videoHeight=${videoElement.videoHeight}`);
            if (videoElement.videoWidth === 0 || videoElement.videoHeight === 0 && isCameraReady) {
                 console.warn("[sendFrame] Video dimensions are zero but camera was ready. Possible issue. Frame not sent.");
            }
        }
    }

    // --- Socket.IO 이벤트 핸들러 ---
    socket.on('connect', () => {
        console.log(`[socket.on('connect')] Successfully connected to server. SID: ${socket.id}`);
        statusBar.textContent = '서버 연결 성공!';
        isSocketConnected = true;
        tryStartFrameSending(); // 카메라가 이미 준비되었다면 프레임 전송 시작
    });

    socket.on('disconnect', (reason) => {
        console.warn(`[socket.on('disconnect')] Server connection lost. Reason: ${reason}`);
        statusBar.textContent = `서버 연결 끊김: ${reason}. 재연결 시도 중...`;
        isSocketConnected = false;
        stopFrameSending(`Socket disconnected: ${reason}`);
    });

    socket.on('connect_error', (err) => {
        console.error(`[socket.on('connect_error')] Server connection error: ${err.message}`, err);
        statusBar.textContent = `서버 연결 오류: ${err.message}`;
        isSocketConnected = false;
        stopFrameSending(`Socket connection error: ${err.message}`);
    });

    socket.on('inference_result', (data) => {
        const { class_name, class_id, confidence } = data;
        inferenceOverlay.textContent = `결과: ${class_name || "N/A"} (신뢰도: ${confidence !== undefined ? (confidence * 100).toFixed(1) + '%' : 'N/A'})`;
        if (class_id !== undefined && class_id >= 0 && class_id < (Object.keys(checkboxes).length)) {
            const targetCheckboxKey = `cbClass${class_id + 1}`;
            if (checkboxes[targetCheckboxKey]) checkboxes[targetCheckboxKey].checked = true;
        }
    });

    // --- 버튼 이벤트 핸들러 ---
    if (startCameraButton) {
        startCameraButton.addEventListener('click', async () => {
            console.log("[startCameraButton] Clicked by user.");
            startCameraButton.disabled = true; // 중복 클릭 방지
            statusBar.textContent = "카메라 초기화 중...";

            if (!videoElement.srcObject) { // 스트림이 아직 할당되지 않았다면 (최초 클릭 또는 이전 실패)
                const cameraInitialized = await initializeCamera(); // 스트림을 얻고 srcObject에 할당, 핸들러 등록
                if (!cameraInitialized) {
                    statusBar.textContent = "카메라 초기화 실패. 권한 등을 확인하세요.";
                    startCameraButton.disabled = false; // 다시 시도할 수 있도록 버튼 활성화
                    return;
                }
            }

            // 스트림이 할당되어 있고, 비디오가 일시정지 상태이거나 아직 재생 시작 전이라면
            if (videoElement.srcObject && (videoElement.paused || videoElement.readyState < HTMLMediaElement.HAVE_ENOUGH_DATA)) {
                try {
                    console.log("[startCameraButton] Attempting to play video...");
                    statusBar.textContent = "카메라 스트림 재생 시도...";
                    await videoElement.play(); // 사용자 인터랙션 내에서 호출
                    console.log("[startCameraButton] videoElement.play() call successful (onplaying event will confirm).");
                    // onplaying 이벤트 핸들러에서 isCameraReady=true 및 tryStartFrameSending() 호출됨
                } catch (playErr) {
                    console.error("[startCameraButton] Error calling videoElement.play():", playErr);
                    statusBar.textContent = `비디오 재생 오류: ${playErr.name}`;
                    streamActive = false;
                    isCameraReady = false;
                    startCameraButton.disabled = false; // 다시 시도할 수 있도록 버튼 활성화
                }
            } else if (videoElement.srcObject && !videoElement.paused) {
                console.log("[startCameraButton] Video is already playing.");
                statusBar.textContent = "카메라가 이미 재생 중입니다.";
                // 이미 재생 중이면 버튼을 다시 활성화할 필요는 없을 수 있음 (상황에 따라)
            } else if (!videoElement.srcObject) {
                console.warn("[startCameraButton] No video stream available after setup. Check camera permissions or initializeCamera logic.");
                statusBar.textContent = "카메라 스트림을 얻을 수 없습니다. 권한을 확인하거나 페이지를 새로고침 해주세요.";
                startCameraButton.disabled = false;
            }
            // play()가 성공적으로 호출되면 onplaying에서 버튼 상태를 관리하거나, 여기서 바로 비활성화 유지
            // startCameraButton.disabled = false; // play() 성공 여부와 관계없이 일단 다시 활성화 (선택적)
        });
    } else {
        console.warn("Start camera button (startCameraButton) not found in DOM.");
        statusBar.textContent = "오류: '카메라 시작' 버튼을 찾을 수 없습니다.";
    }

    if (resetCheckboxesBtn) {
        resetCheckboxesBtn.addEventListener('click', () => {
            console.log("[resetCheckboxesBtn] Clicked by user.");
            Object.values(checkboxes).forEach(cb => cb.checked = false);
            inferenceOverlay.textContent = "결과: 이력 리셋됨";
        });
    }

    // --- 페이지 언로드 시 리소스 정리 ---
    window.addEventListener('beforeunload', () => {
        console.log("[window.beforeunload] Page is closing. Cleaning up resources.");
        if (socket && socket.connected) {
            socket.disconnect();
        }
        if (stream) { // stream 객체가 존재하면 트랙 중지
            stream.getTracks().forEach(track => track.stop());
            console.log("[window.beforeunload] All media tracks stopped.");
        }
        stopFrameSending("Page closing");
        streamActive = false;
        isCameraReady = false;
        isSocketConnected = false;
    });

    // 페이지 로드 시 초기 상태 메시지
    statusBar.textContent = "준비 완료. '카메라 시작' 버튼을 누르세요.";
});