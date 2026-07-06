// CYBERMATTE STUDIO - GAME ASSET ANIMATION & SPRITE SHEET CONTROLLER

document.addEventListener('DOMContentLoaded', () => {
    let currentFileId = null;
    let currentVideoFile = null;
    let currentBgImageElement = null;
    let animFramesList = [];
    let animIntervalId = null;
    let isPlayingAnim = false;
    let currentFrameIndex = 0;
    let animFps = 12;

    // DOM Elements
    const dropzone = document.getElementById('dropzone');
    const videoInput = document.getElementById('videoInput');
    const uploadCard = document.getElementById('uploadCard');
    const videoInfoCard = document.getElementById('videoInfoCard');
    const dropzoneContent = document.getElementById('dropzoneContent');
    const removeVideoBtn = document.getElementById('removeVideoBtn');
    const sourceVideoElement = document.getElementById('sourceVideoElement');

    // Video Meta DOM Elements
    const videoThumb = document.getElementById('videoThumb');
    const videoFileName = document.getElementById('videoFileName');
    const metaRes = document.getElementById('metaRes');
    const metaFps = document.getElementById('metaFps');
    const metaDuration = document.getElementById('metaDuration');
    const metaFrames = document.getElementById('metaFrames');
    const metaSize = document.getElementById('metaSize');

    // Controls DOM Elements
    const controlsCard = document.getElementById('controlsCard');
    const numFramesSelect = document.getElementById('numFramesSelect');
    const targetSizeSelect = document.getElementById('targetSizeSelect');
    const upscaleSelect = document.getElementById('upscaleSelect');
    const modelSelect = document.getElementById('modelSelect');
    const startAnimBtn = document.getElementById('startAnimBtn');

    // Progress DOM Elements
    const progressCard = document.getElementById('progressCard');
    const progressBarFill = document.getElementById('progressBarFill');
    const percentVal = document.getElementById('percentVal');
    const frameVal = document.getElementById('frameVal');
    const fpsVal = document.getElementById('fpsVal');
    const terminalBox = document.getElementById('terminalBox');

    // Result DOM Elements
    const resultCard = document.getElementById('resultCard');
    const spritesheetImg = document.getElementById('spritesheetImg');
    const animCanvas = document.getElementById('animCanvas');
    const playPauseBtn = document.getElementById('playPauseBtn');
    const fpsSlider = document.getElementById('fpsSlider');
    const fpsDisplay = document.getElementById('fpsDisplay');
    const downloadBtn = document.getElementById('downloadBtn');
    const restartBtn = document.getElementById('restartBtn');

    // --- UPLOAD HANDLERS ---
    dropzone.addEventListener('click', () => videoInput.click());

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    videoInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    removeVideoBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetVideoSelection();
    });

    async function handleFileUpload(file) {
        currentVideoFile = file;
        const fileUrl = URL.createObjectURL(file);
        sourceVideoElement.src = fileUrl;

        const formData = new FormData();
        formData.append('file', file);

        dropzoneContent.innerHTML = `
            <div class="dropzone-icon"><div class="fa-solid fa-spinner fa-spin"></div></div>
            <h3>Đang đọc tập tin video animation...</h3>
        `;

        let isServerAvailable = false;
        try {
            const resp = await fetch('/api/upload', { method: 'POST', body: formData });
            if (resp.ok) {
                const data = await resp.json();
                currentFileId = data.file_id;
                renderMeta(data.filename, data.width, data.height, data.fps, data.duration, data.total_frames, data.size_mb, data.thumb_url || fileUrl);
                isServerAvailable = true;
            }
        } catch (e) {
            console.log("Server API not reachable, running Client-side mode.");
        }

        if (!isServerAvailable) {
            sourceVideoElement.onloadedmetadata = () => {
                const width = sourceVideoElement.videoWidth || 1280;
                const height = sourceVideoElement.videoHeight || 720;
                const duration = Math.round(sourceVideoElement.duration * 10) / 10 || 10.0;
                const fps = 30;
                const totalFrames = Math.floor(duration * fps);
                const sizeMb = Math.round((file.size / (1024 * 1024)) * 10) / 10;

                renderMeta(file.name, width, height, fps, duration, totalFrames, sizeMb, null);
            };
        }

        dropzone.classList.add('hidden');
        videoInfoCard.classList.remove('hidden');
        controlsCard.classList.remove('disabled-state');
        startAnimBtn.disabled = false;
    }

    function renderMeta(filename, width, height, fps, duration, totalFrames, sizeMb, thumbUrl) {
        videoFileName.textContent = filename;
        metaRes.textContent = `${width} x ${height}`;
        metaFps.textContent = `${fps} FPS`;
        metaDuration.textContent = `${duration}s`;
        metaFrames.textContent = `${totalFrames} frames`;
        metaSize.textContent = `${sizeMb} MB`;
        if (thumbUrl) {
            videoThumb.src = thumbUrl;
            videoThumb.style.display = 'block';
        }
    }

    function resetVideoSelection() {
        currentFileId = null;
        currentVideoFile = null;
        videoInput.value = '';
        dropzone.classList.remove('hidden');
        videoInfoCard.classList.add('hidden');
        dropzoneContent.innerHTML = `
            <div class="dropzone-icon"><i class="fa-solid fa-film"></i></div>
            <h3>Kéo thả Video Animation vào đây để <span>Tạo Sprite Sheet</span></h3>
            <p class="dropzone-hint">Hỗ trợ MP4, MOV, WEBM, AVI (Bóc tách chuẩn sắc nét Game Asset VIP)</p>
        `;
        controlsCard.classList.add('disabled-state');
        startAnimBtn.disabled = true;
        stopFlipbookAnimation();
    }

    // --- START ANIMATION GENERATION ---
    startAnimBtn.addEventListener('click', async () => {
        if (!currentVideoFile && !currentFileId) return;

        controlsCard.classList.add('disabled-state');
        startAnimBtn.disabled = true;
        progressCard.classList.remove('hidden');
        progressCard.scrollIntoView({ behavior: 'smooth' });

        if (terminalBox) terminalBox.innerHTML = '<div>[SYSTEM LOG] Phân tích video và nạp mô hình IS-Net DIS Engine...</div>';

        const numFrames = parseInt(numFramesSelect.value);
        const targetSize = parseInt(targetSizeSelect.value);
        const upscaleFactor = parseFloat(upscaleSelect.value);
        const selectedModel = modelSelect.value;

        // On Local Server (`http://localhost:8000`), run Python worker
        if (currentFileId && window.location.hostname === 'localhost') {
            const formData = new FormData();
            formData.append('file_id', currentFileId);
            formData.append('num_frames', numFrames);
            formData.append('target_size', targetSize);
            formData.append('upscale_factor', upscaleFactor);
            formData.append('model_name', selectedModel);

            try {
                const resp = await fetch('/api/process-animation', { method: 'POST', body: formData });
                const data = await resp.json();
                if (data.task_id) {
                    connectWebSocket(data.task_id);
                    return;
                }
            } catch (err) {
                appendLog(`Error: ${err.message}`);
            }
        }

        // --- CLIENT-SIDE BROWSER PROCESSING (VERCEL MODE) ---
        runClientSideProcessor(numFrames, targetSize);
    });

    async function runClientSideProcessor(numFrames, targetSize) {
        try {
            const res = await window.clientVideoProcessor.extractTestFrames({
                videoElement: sourceVideoElement,
                numFrames: numFrames,
                targetSize: targetSize,
                progressCallback: updateProgressUI,
                logCallback: appendLog
            });
            showAnimationResult(res);
        } catch (err) {
            appendLog(`Lỗi bóc tách frame: ${err.message}`);
            progressCard.classList.add('hidden');
            controlsCard.classList.remove('disabled-state');
            startAnimBtn.disabled = false;
        }
    }

    function appendLog(text) {
        if (!terminalBox) return;
        const line = document.createElement('div');
        line.textContent = text;
        terminalBox.appendChild(line);
        terminalBox.scrollTop = terminalBox.scrollHeight;
    }

    function connectWebSocket(taskId) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/progress/${taskId}`;

        websocket = new WebSocket(wsUrl);
        websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updateProgressUI(data);

            if (data.log) {
                appendLog(data.log);
            }

            if (data.status === 'completed') {
                websocket.close();
                showAnimationResult(data);
            } else if (data.status === 'failed') {
                websocket.close();
                appendLog(`[ERROR] ${data.error || 'Unknown error'}`);
            }
        };
    }

    function updateProgressUI(data) {
        const percent = data.percent || 0;
        progressBarFill.style.width = `${percent}%`;
        percentVal.textContent = `${percent}%`;
        frameVal.textContent = `${data.current_frame || 0} / ${data.total_frames || 0}`;
        fpsVal.textContent = `${data.fps || 0} FPS`;
    }

    // --- DISPLAY RESULT & INTERACTIVE ANIMATION FLIPBOOK PLAYER ---
    function showAnimationResult(data) {
        progressCard.classList.add('hidden');
        resultCard.classList.remove('hidden');
        resultCard.scrollIntoView({ behavior: 'smooth' });

        if (data.spritesheet_url) {
            spritesheetImg.src = data.spritesheet_url;
        } else if (data.extracted_images && data.extracted_images.length > 0) {
            spritesheetImg.src = data.extracted_images[0];
        }

        if (data.download_url) {
            downloadBtn.href = data.download_url;
            downloadBtn.setAttribute('download', 'game_asset_spritesheet.zip');
        }

        // Load frames for interactive Flipbook animation preview
        if (data.extracted_images && data.extracted_images.length > 0) {
            loadFramesAndStartAnimation(data.extracted_images);
        }
    }

    function loadFramesAndStartAnimation(imageUrls) {
        stopFlipbookAnimation();
        animFramesList = [];
        let loadedCount = 0;

        imageUrls.forEach(url => {
            const img = new Image();
            img.src = url;
            img.onload = () => {
                loadedCount++;
                if (loadedCount === imageUrls.length) {
                    startFlipbookAnimation();
                }
            };
            animFramesList.push(img);
        });
    }

    function startFlipbookAnimation() {
        if (animFramesList.length === 0) return;
        isPlayingAnim = true;
        currentFrameIndex = 0;
        playPauseBtn.innerHTML = '<i class="fa-solid fa-pause"></i> Tạm Dừng';

        const ctx = animCanvas.getContext('2d');

        function renderNextFrame() {
            if (!isPlayingAnim || animFramesList.length === 0) return;
            const img = animFramesList[currentFrameIndex];
            if (img && img.complete) {
                animCanvas.width = img.width || 256;
                animCanvas.height = img.height || 256;
                ctx.clearRect(0, 0, animCanvas.width, animCanvas.height);
                ctx.drawImage(img, 0, 0);
            }
            currentFrameIndex = (currentFrameIndex + 1) % animFramesList.length;
        }

        renderNextFrame();
        clearInterval(animIntervalId);
        animIntervalId = setInterval(renderNextFrame, Math.floor(1000 / animFps));
    }

    function stopFlipbookAnimation() {
        isPlayingAnim = false;
        clearInterval(animIntervalId);
        if (playPauseBtn) playPauseBtn.innerHTML = '<i class="fa-solid fa-play"></i> Phát Animation';
    }

    if (playPauseBtn) {
        playPauseBtn.addEventListener('click', () => {
            if (isPlayingAnim) {
                stopFlipbookAnimation();
            } else {
                startFlipbookAnimation();
            }
        });
    }

    if (fpsSlider) {
        fpsSlider.addEventListener('input', (e) => {
            animFps = parseInt(e.target.value);
            fpsDisplay.textContent = `${animFps} FPS`;
            if (isPlayingAnim) {
                clearInterval(animIntervalId);
                animIntervalId = setInterval(() => {
                    const ctx = animCanvas.getContext('2d');
                    if (animFramesList.length > 0) {
                        const img = animFramesList[currentFrameIndex];
                        if (img && img.complete) {
                            ctx.clearRect(0, 0, animCanvas.width, animCanvas.height);
                            ctx.drawImage(img, 0, 0);
                        }
                        currentFrameIndex = (currentFrameIndex + 1) % animFramesList.length;
                    }
                }, Math.floor(1000 / animFps));
            }
        });
    }

    restartBtn.addEventListener('click', () => {
        resultCard.classList.add('hidden');
        resetVideoSelection();
        uploadCard.scrollIntoView({ behavior: 'smooth' });
    });
});
