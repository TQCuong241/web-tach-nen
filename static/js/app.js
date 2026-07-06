// CYBERMATTE STUDIO - FRONTEND JS CONTROLLER (SERVER & VERCEL BROWSER HYBRID)

document.addEventListener('DOMContentLoaded', () => {
    // State variables
    let currentFileId = null;
    let currentVideoFile = null;
    let currentBgImageId = null;
    let currentBgImageElement = null;
    let selectedBgMode = 'greenscreen';
    let websocket = null;

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
    const bgModeBtns = document.querySelectorAll('.bg-mode-btn');
    const colorPickerBox = document.getElementById('colorPickerBox');
    const imageUploadBox = document.getElementById('imageUploadBox');
    const blurOptionsBox = document.getElementById('blurOptionsBox');
    const bgColorInput = document.getElementById('bgColorInput');
    const colorPresets = document.querySelectorAll('.color-preset');
    const colorModePreview = document.getElementById('colorModePreview');
    const bgImageInput = document.getElementById('bgImageInput');
    const bgImageBtn = document.getElementById('bgImageBtn');
    const bgImageName = document.getElementById('bgImageName');
    const blurRadiusInput = document.getElementById('blurRadiusInput');
    const blurValDisplay = document.getElementById('blurValDisplay');

    const modelSelect = document.getElementById('modelSelect');
    const outputFormatSelect = document.getElementById('outputFormatSelect');
    const downsampleSelect = document.getElementById('downsampleSelect');
    const startProcessBtn = document.getElementById('startProcessBtn');

    // Progress DOM Elements
    const progressCard = document.getElementById('progressCard');
    const progressBarFill = document.getElementById('progressBarFill');
    const percentVal = document.getElementById('percentVal');
    const frameVal = document.getElementById('frameVal');
    const fpsVal = document.getElementById('fpsVal');
    const etaVal = document.getElementById('etaVal');

    // Result DOM Elements
    const resultCard = document.getElementById('resultCard');
    const resultVideo = document.getElementById('resultVideo');
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
            handleVideoUpload(files[0]);
        }
    });

    videoInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleVideoUpload(e.target.files[0]);
        }
    });

    removeVideoBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetVideoSelection();
    });

    async function handleVideoUpload(file) {
        currentVideoFile = file;
        const videoUrl = URL.createObjectURL(file);
        sourceVideoElement.src = videoUrl;

        // Try server upload if API available, else compute meta locally
        const formData = new FormData();
        formData.append('file', file);

        dropzoneContent.innerHTML = `
            <div class="dropzone-icon"><div class="fa-solid fa-spinner fa-spin"></div></div>
            <h3>Đang đọc thông tin video...</h3>
        `;

        let isServerAvailable = false;
        try {
            const resp = await fetch('/api/upload', { method: 'POST', body: formData });
            if (resp.ok) {
                const data = await resp.json();
                currentFileId = data.file_id;
                renderMeta(data.filename, data.width, data.height, data.fps, data.duration, data.total_frames, data.size_mb, data.thumb_url);
                isServerAvailable = true;
            }
        } catch (e) {
            console.log("Server API not reachable, running Client-side local mode for Vercel.");
        }

        if (!isServerAvailable) {
            // Read metadata locally using HTML5 Video Element
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
        startProcessBtn.disabled = false;
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
        } else {
            videoThumb.style.display = 'none';
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
            <h3>Kéo thả Video vào đây hoặc <span>Duyệt Tập Tin</span></h3>
            <p class="dropzone-hint">Khuyên dùng video có chủ thể người hoặc vật thể chuyển động rõ nét</p>
        `;
        controlsCard.classList.add('disabled-state');
        startProcessBtn.disabled = true;
    }

    // --- BACKGROUND MODE CONTROLS ---
    bgModeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            bgModeBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            selectedBgMode = btn.dataset.mode;

            colorPickerBox.classList.add('hidden');
            imageUploadBox.classList.add('hidden');
            blurOptionsBox.classList.add('hidden');

            if (selectedBgMode === 'color') {
                colorPickerBox.classList.remove('hidden');
            } else if (selectedBgMode === 'image') {
                imageUploadBox.classList.remove('hidden');
            } else if (selectedBgMode === 'blur') {
                blurOptionsBox.classList.remove('hidden');
            }
        });
    });

    bgColorInput.addEventListener('input', (e) => {
        colorModePreview.style.backgroundColor = e.target.value;
    });

    colorPresets.forEach(preset => {
        preset.addEventListener('click', () => {
            const color = preset.dataset.color;
            bgColorInput.value = color;
            colorModePreview.style.backgroundColor = color;
        });
    });

    bgImageBtn.addEventListener('click', () => bgImageInput.click());
    bgImageInput.addEventListener('change', async (e) => {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            bgImageName.textContent = `Đã chọn: ${file.name}`;

            const img = new Image();
            img.src = URL.createObjectURL(file);
            currentBgImageElement = img;

            if (currentFileId) {
                const formData = new FormData();
                formData.append('file', file);
                try {
                    const resp = await fetch('/api/upload-bg', { method: 'POST', body: formData });
                    const data = await resp.json();
                    currentBgImageId = data.bg_id;
                } catch (err) {}
            }
        }
    });

    blurRadiusInput.addEventListener('input', (e) => {
        blurValDisplay.textContent = `${e.target.value}px`;
    });

    // --- START PROCESSING HANDLER ---
    startProcessBtn.addEventListener('click', async () => {
        if (!currentVideoFile && !currentFileId) return;

        // UI Transition to Progress
        controlsCard.classList.add('disabled-state');
        startProcessBtn.disabled = true;
        progressCard.classList.remove('hidden');
        progressCard.scrollIntoView({ behavior: 'smooth' });

        const selectedModel = modelSelect.value;

        // Check if server processing is available & requested
        if (currentFileId && selectedModel !== 'auto') {
            const formData = new FormData();
            formData.append('file_id', currentFileId);
            formData.append('bg_type', selectedBgMode);
            formData.append('bg_color', bgColorInput.value);
            if (currentBgImageId) formData.append('bg_image_id', currentBgImageId);
            formData.append('blur_radius', blurRadiusInput.value);
            formData.append('output_format', outputFormatSelect.value);
            formData.append('downsample_ratio', downsampleSelect.value);
            formData.append('model_name', selectedModel);

            try {
                const resp = await fetch('/api/process', { method: 'POST', body: formData });
                const data = await resp.json();
                if (data.task_id) {
                    connectWebSocket(data.task_id);
                    return;
                }
            } catch (err) {
                console.log("Falling back to Client-side browser frame matting...");
            }
        }

        // --- CLIENT-SIDE BROWSER FRAME PROCESSING (VERCEL MODE) ---
        try {
            const res = await window.clientVideoProcessor.processVideo({
                videoElement: sourceVideoElement,
                bgType: selectedBgMode,
                bgColor: bgColorInput.value,
                bgImageElement: currentBgImageElement,
                blurRadius: parseInt(blurRadiusInput.value),
                outputFormat: outputFormatSelect.value,
                progressCallback: updateProgressUI
            });

            showResult({
                output_media_url: res.blobUrl,
                download_url: res.blobUrl,
                output_filename: `nobg_video.${res.extension}`
            });
        } catch (err) {
            alert(`Lỗi xử lý frame trên trình duyệt: ${err.message}`);
            progressCard.classList.add('hidden');
            controlsCard.classList.remove('disabled-state');
            startProcessBtn.disabled = false;
        }
    });

    function connectWebSocket(taskId) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/progress/${taskId}`;

        websocket = new WebSocket(wsUrl);
        websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            updateProgressUI(data);

            if (data.status === 'completed') {
                websocket.close();
                showResult(data);
            } else if (data.status === 'failed') {
                websocket.close();
                alert(`Tách nền thất bại: ${data.error || 'Unknown error'}`);
                progressCard.classList.add('hidden');
                controlsCard.classList.remove('disabled-state');
                startProcessBtn.disabled = false;
            }
        };
    }

    function updateProgressUI(data) {
        const percent = data.percent || 0;
        progressBarFill.style.width = `${percent}%`;
        percentVal.textContent = `${percent}%`;
        frameVal.textContent = `${data.current_frame || 0} / ${data.total_frames || 0}`;
        fpsVal.textContent = `${data.fps || 0} FPS`;

        if (data.eta_seconds !== undefined) {
            const mins = Math.floor(data.eta_seconds / 60);
            const secs = data.eta_seconds % 60;
            etaVal.textContent = `${mins > 0 ? mins + 'm ' : ''}${secs}s`;
        }
    }

    function showResult(data) {
        progressCard.classList.add('hidden');
        resultCard.classList.remove('hidden');
        resultCard.scrollIntoView({ behavior: 'smooth' });

        if (data.output_media_url) {
            resultVideo.src = data.output_media_url;
            resultVideo.load();
            resultVideo.play().catch(() => {});
        }

        if (data.download_url) {
            downloadBtn.href = data.download_url;
            downloadBtn.setAttribute('download', data.output_filename || 'nobg_video.mp4');
        }
    }

    restartBtn.addEventListener('click', () => {
        resultCard.classList.add('hidden');
        resetVideoSelection();
        uploadCard.scrollIntoView({ behavior: 'smooth' });
    });
});
