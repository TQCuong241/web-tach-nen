// CYBERMATTE STUDIO - FRONTEND JS CONTROLLER (VIDEO & IMAGE HYBRID ENGINE)

document.addEventListener('DOMContentLoaded', () => {
    let currentFileId = null;
    let currentVideoFile = null;
    let isImageFile = false;
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
        const fileExt = file.name.split('.').pop().toLowerCase();
        isImageFile = ['png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff'].includes(fileExt);

        const fileUrl = URL.createObjectURL(file);
        if (isImageFile) {
            videoThumb.src = fileUrl;
            videoThumb.style.display = 'block';
        } else {
            sourceVideoElement.src = fileUrl;
        }

        const formData = new FormData();
        formData.append('file', file);

        dropzoneContent.innerHTML = `
            <div class="dropzone-icon"><i class="fa-solid fa-spinner fa-spin"></i></div>
            <h3>Đang đọc tập tin ${isImageFile ? 'ảnh' : 'video'}...</h3>
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
            if (isImageFile) {
                const img = new Image();
                img.src = fileUrl;
                img.onload = () => {
                    const sizeMb = Math.round((file.size / (1024 * 1024)) * 10) / 10;
                    renderMeta(file.name, img.width, img.height, 0, 0, 1, sizeMb, fileUrl);
                };
            } else {
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
        }

        dropzone.classList.add('hidden');
        videoInfoCard.classList.remove('hidden');
        controlsCard.classList.remove('disabled-state');
        startProcessBtn.disabled = false;
    }

    function renderMeta(filename, width, height, fps, duration, totalFrames, sizeMb, thumbUrl) {
        videoFileName.textContent = filename;
        metaRes.textContent = `${width} x ${height}`;
        metaFps.textContent = isImageFile ? 'Ảnh tĩnh' : `${fps} FPS`;
        metaDuration.textContent = isImageFile ? '1 khung hình' : `${duration}s`;
        metaFrames.textContent = isImageFile ? '1 frame' : `${totalFrames} frames`;
        metaSize.textContent = `${sizeMb} MB`;
        if (thumbUrl) {
            videoThumb.src = thumbUrl;
            videoThumb.style.display = 'block';
        }
    }

    function resetVideoSelection() {
        currentFileId = null;
        currentVideoFile = null;
        isImageFile = false;
        videoInput.value = '';
        dropzone.classList.remove('hidden');
        videoInfoCard.classList.add('hidden');
        dropzoneContent.innerHTML = `
            <div class="dropzone-icon"><i class="fa-solid fa-photo-film"></i></div>
            <h3>Kéo thả Video hoặc Ảnh vào đây để <span>Bóc Tách Nền</span></h3>
            <p class="dropzone-hint">Hỗ trợ MP4, MOV, WEBM, PNG, JPG, WEBP (Bóc tách chuẩn sắc nét VIP)</p>
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

        controlsCard.classList.add('disabled-state');
        startProcessBtn.disabled = true;
        progressCard.classList.remove('hidden');
        progressCard.scrollIntoView({ behavior: 'smooth' });

        const selectedModel = modelSelect.value;

        // On Local Server (`http://localhost:8000`), run Python worker
        if (currentFileId && selectedModel !== 'auto' && window.location.hostname === 'localhost') {
            const formData = new FormData();
            formData.append('file_id', currentFileId);
            formData.append('bg_type', selectedBgMode);
            formData.append('bg_color', bgColorInput.value);
            if (currentBgImageId) formData.append('bg_image_id', currentBgImageId);
            formData.append('blur_radius', blurRadiusInput.value);
            formData.append('output_format', outputFormatSelect.value);
            formData.append('downsample_ratio', downsampleSelect.value);
            formData.append('model_name', selectedModel);

            if (isImageFile) {
                try {
                    const resp = await fetch('/api/process-image', { method: 'POST', body: formData });
                    const data = await resp.json();
                    if (data.status === 'completed') {
                        showResult(data);
                        return;
                    }
                } catch (err) {}
            } else {
                try {
                    const resp = await fetch('/api/process', { method: 'POST', body: formData });
                    const data = await resp.json();
                    if (data.task_id) {
                        connectWebSocket(data.task_id);
                        return;
                    }
                } catch (err) {}
            }
        }

        // --- CLIENT-SIDE BROWSER PROCESSING (VERCEL / CLIENT MODE) ---
        runClientSideProcessor();
    });

    async function runClientSideProcessor() {
        try {
            if (isImageFile) {
                // Single Image Matting
                const imgElement = new Image();
                imgElement.src = URL.createObjectURL(currentVideoFile);
                await new Promise(r => imgElement.onload = r);

                const dummyVideo = document.createElement('video');
                dummyVideo.width = imgElement.width;
                dummyVideo.height = imgElement.height;

                // Create a temporary canvas representing the image
                const canvas = document.createElement('canvas');
                canvas.width = imgElement.width;
                canvas.height = imgElement.height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(imgElement, 0, 0);

                const res = await window.clientVideoProcessor.processVideo({
                    videoElement: dummyVideo,
                    bgType: selectedBgMode,
                    bgColor: bgColorInput.value,
                    bgImageElement: currentBgImageElement,
                    blurRadius: parseInt(blurRadiusInput.value),
                    outputFormat: 'png',
                    progressCallback: updateProgressUI
                });

                showResult({
                    output_media_url: res.blobUrl,
                    download_url: res.blobUrl,
                    output_filename: `nobg_image.png`
                });
            } else {
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
            }
        } catch (err) {
            alert(`Lỗi xử lý bóc tách: ${err.message}`);
            progressCard.classList.add('hidden');
            controlsCard.classList.remove('disabled-state');
            startProcessBtn.disabled = false;
        }
    }

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
                runClientSideProcessor();
            }
        };
    }

    function updateProgressUI(data) {
        const percent = data.percent || 0;
        progressBarFill.style.width = `${percent}%`;
        percentVal.textContent = `${percent}%`;
        frameVal.textContent = isImageFile ? '1 / 1 frame' : `${data.current_frame || 0} / ${data.total_frames || 0}`;
        fpsVal.textContent = isImageFile ? 'Ảnh tĩnh' : `${data.fps || 0} FPS`;

        if (data.eta_seconds !== undefined && !isImageFile) {
            const mins = Math.floor(data.eta_seconds / 60);
            const secs = data.eta_seconds % 60;
            etaVal.textContent = `${mins > 0 ? mins + 'm ' : ''}${secs}s`;
        } else {
            etaVal.textContent = '0s';
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
            downloadBtn.setAttribute('download', data.output_filename || 'nobg_result.png');
        }
    }

    restartBtn.addEventListener('click', () => {
        resultCard.classList.add('hidden');
        resetVideoSelection();
        uploadCard.scrollIntoView({ behavior: 'smooth' });
    });
});
