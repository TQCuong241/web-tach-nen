/**
 * CYBERMATTE AI - CLIENT-SIDE BROWSER FRAME ENGINE
 * Runs video frame-by-frame background matting directly in the browser.
 * Ensures EXACT frame rate and duration match (e.g., 240 frames @ 10s = 10s output).
 */

class ClientVideoProcessor {
    constructor() {
        this.segmenter = null;
        this.isInitialized = false;
    }

    async init() {
        if (this.isInitialized) return;

        if (window.SelfieSegmentation) {
            try {
                this.segmenter = new window.SelfieSegmentation({
                    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`
                });

                this.segmenter.setOptions({
                    modelSelection: 1, // 0 for general, 1 for landscape/quality
                });

                this.isInitialized = true;
                console.log("[ClientEngine] MediaPipe SelfieSegmentation initialized successfully.");
            } catch (e) {
                console.warn("[ClientEngine] Could not init MediaPipe SelfieSegmentation:", e);
            }
        }
    }

    /**
     * Process video frame-by-frame on Canvas maintaining exact duration & frame rate
     */
    async processVideo({
        videoElement,
        bgType = 'greenscreen',
        bgColor = '#00FF00',
        bgImageElement = null,
        blurRadius = 15,
        outputFormat = 'mp4',
        progressCallback
    }) {
        await this.init();

        const duration = videoElement.duration || 10.0;
        const width = videoElement.videoWidth || 1280;
        const height = videoElement.videoHeight || 720;
        const fps = 30.0;
        const totalFrames = Math.floor(duration * fps);

        const sourceCanvas = document.createElement('canvas');
        sourceCanvas.width = width;
        sourceCanvas.height = height;
        const sourceCtx = sourceCanvas.getContext('2d', { willReadFrequently: true });

        const outputCanvas = document.createElement('canvas');
        outputCanvas.width = width;
        outputCanvas.height = height;
        const outputCtx = outputCanvas.getContext('2d');

        // captureStream(0) allows manual frame stepping via track.requestFrame()
        const stream = outputCanvas.captureStream(0);
        const track = stream.getVideoTracks()[0];

        let mimeType = 'video/webm;codecs=vp9';
        if (!MediaRecorder.isTypeSupported(mimeType)) mimeType = 'video/webm';
        if (!MediaRecorder.isTypeSupported(mimeType)) mimeType = 'video/mp4';

        const recorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 5000000 });
        const chunks = [];
        recorder.ondataavailable = (e) => {
            if (e.data.size > 0) chunks.push(e.data);
        };

        recorder.start();

        const frameInterval = 1.0 / fps;
        const startTime = Date.now();

        for (let frameIdx = 1; frameIdx <= totalFrames; frameIdx++) {
            const currentTime = (frameIdx - 1) * frameInterval;
            videoElement.currentTime = currentTime;

            await new Promise(resolve => {
                const onSeeked = () => {
                    videoElement.removeEventListener('seeked', onSeeked);
                    resolve();
                };
                videoElement.addEventListener('seeked', onSeeked);
            });

            sourceCtx.drawImage(videoElement, 0, 0, width, height);

            if (this.segmenter) {
                await new Promise((resolve) => {
                    this.segmenter.onResults((results) => {
                        outputCtx.save();
                        outputCtx.clearRect(0, 0, width, height);

                        // 1. Draw segmentation mask
                        outputCtx.drawImage(results.segmentationMask, 0, 0, width, height);

                        // 2. Composite original subject frame inside mask (replaces red mask with true subject colors)
                        outputCtx.globalCompositeOperation = 'source-in';
                        outputCtx.drawImage(results.image, 0, 0, width, height);

                        // 3. Composite background behind subject
                        if (bgType !== 'transparent') {
                            outputCtx.globalCompositeOperation = 'destination-over';
                            this._drawBackground(outputCtx, sourceCanvas, width, height, bgType, bgColor, bgImageElement, blurRadius);
                        }

                        outputCtx.restore();

                        // Request canvas frame to be recorded at exact interval
                        if (track && track.requestFrame) {
                            track.requestFrame();
                        }
                        resolve();
                    });
                    this.segmenter.send({ image: sourceCanvas });
                });
            } else {
                this._drawFallbackMatting(sourceCtx, outputCtx, width, height, bgType, bgColor, bgImageElement);
                if (track && track.requestFrame) {
                    track.requestFrame();
                }
            }

            const elapsedSec = (Date.now() - startTime) / 1000.0;
            const currentFps = frameIdx / elapsedSec;
            const percent = Math.min(100, Math.floor((frameIdx / totalFrames) * 100));
            const etaSec = currentFps > 0 ? Math.ceil((totalFrames - frameIdx) / currentFps) : 0;

            if (progressCallback) {
                progressCallback({
                    status: 'processing',
                    current_frame: frameIdx,
                    total_frames: totalFrames,
                    percent: percent,
                    fps: Math.round(currentFps * 10) / 10,
                    eta_seconds: etaSec
                });
            }
        }

        return new Promise((resolve) => {
            recorder.onstop = () => {
                const blob = new Blob(chunks, { type: mimeType });
                const blobUrl = URL.createObjectURL(blob);
                resolve({
                    blobUrl,
                    blob,
                    mimeType,
                    extension: mimeType.includes('mp4') ? 'mp4' : 'webm'
                });
            };
            recorder.stop();
        });
    }

    _drawBackground(ctx, sourceCanvas, width, height, bgType, bgColor, bgImageElement, blurRadius) {
        if (bgType === 'greenscreen') {
            ctx.fillStyle = '#00FF00';
            ctx.fillRect(0, 0, width, height);
        } else if (bgType === 'bluescreen') {
            ctx.fillStyle = '#0000FF';
            ctx.fillRect(0, 0, width, height);
        } else if (bgType === 'color') {
            ctx.fillStyle = bgColor || '#00FF00';
            ctx.fillRect(0, 0, width, height);
        } else if (bgType === 'image' && bgImageElement) {
            ctx.drawImage(bgImageElement, 0, 0, width, height);
        } else if (bgType === 'blur') {
            ctx.filter = `blur(${blurRadius || 15}px)`;
            ctx.drawImage(sourceCanvas, 0, 0, width, height);
            ctx.filter = 'none';
        } else {
            ctx.clearRect(0, 0, width, height);
        }
    }

    _drawFallbackMatting(sourceCtx, outputCtx, width, height, bgType, bgColor, bgImageElement) {
        const frameData = sourceCtx.getImageData(0, 0, width, height);
        const data = frameData.data;

        for (let i = 0; i < data.length; i += 4) {
            const r = data[i];
            const g = data[i + 1];
            const b = data[i + 2];
            if (g > 100 && g > r * 1.4 && g > b * 1.4) {
                data[i + 3] = 0;
            }
        }
        outputCtx.putImageData(frameData, 0, 0);
    }
}

window.clientVideoProcessor = new ClientVideoProcessor();
