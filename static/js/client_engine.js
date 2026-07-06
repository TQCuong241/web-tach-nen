/**
 * CYBERMATTE AI - CLIENT-SIDE BROWSER FRAME ENGINE
 * Features AI Segmentation + White De-Fringing & Edge Matte Refinement.
 * Includes 10-Frame Test Extractor with real-time log output.
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
                    modelSelection: 1,
                });

                this.isInitialized = true;
                console.log("[ClientEngine] MediaPipe SelfieSegmentation initialized successfully.");
            } catch (e) {
                console.warn("[ClientEngine] Could not init MediaPipe SelfieSegmentation:", e);
            }
        }
    }

    /**
     * Extract and process 10 test frames directly in browser with real-time log output
     */
    async extractTestFrames({
        videoElement,
        numFrames = 10,
        targetSize = 1000,
        progressCallback,
        logCallback
    }) {
        await this.init();

        const duration = videoElement.duration || 10.0;
        const width = videoElement.videoWidth || 1280;
        const height = videoElement.videoHeight || 720;
        const outSize = targetSize > 0 ? targetSize : width;

        const sourceCanvas = document.createElement('canvas');
        sourceCanvas.width = width;
        sourceCanvas.height = height;
        const sourceCtx = sourceCanvas.getContext('2d', { willReadFrequently: true });

        const outputCanvas = document.createElement('canvas');
        outputCanvas.width = outSize;
        outputCanvas.height = outSize;
        const outputCtx = outputCanvas.getContext('2d', { willReadFrequently: true });

        const extractedImages = [];
        const logMessages = [];
        const frameInterval = duration / numFrames;

        for (let i = 0; i < numFrames; i++) {
            const frameStart = Date.now();
            const seekTime = i * frameInterval;
            videoElement.currentTime = seekTime;

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
                        outputCtx.clearRect(0, 0, outSize, outSize);

                        const aspect = width / height;
                        let drawW = outSize;
                        let drawH = outSize;
                        let offsetX = 0;
                        let offsetY = 0;
                        if (aspect > 1) {
                            drawH = outSize / aspect;
                            offsetY = (outSize - drawH) / 2;
                        } else {
                            drawW = outSize * aspect;
                            offsetX = (outSize - drawW) / 2;
                        }

                        outputCtx.drawImage(results.segmentationMask, offsetX, offsetY, drawW, drawH);
                        outputCtx.globalCompositeOperation = 'source-in';
                        outputCtx.drawImage(results.image, offsetX, offsetY, drawW, drawH);
                        outputCtx.restore();

                        // White De-fringing Pass
                        const imgData = outputCtx.getImageData(0, 0, outSize, outSize);
                        const pixels = imgData.data;
                        for (let k = 0; k < pixels.length; k += 4) {
                            const r = pixels[k];
                            const g = pixels[k + 1];
                            const b = pixels[k + 2];
                            const a = pixels[k + 3];
                            if (a > 0) {
                                if (r > 230 && g > 230 && b > 230) {
                                    pixels[k + 3] = 0;
                                } else if (r > 195 && g > 195 && b > 195) {
                                    const brightness = (r + g + b) / 3.0;
                                    const factor = (230.0 - brightness) / 35.0;
                                    pixels[k + 3] = Math.floor(a * Math.max(0, factor));
                                }
                            }
                        }
                        outputCtx.putImageData(imgData, 0, 0);
                        resolve();
                    });
                    this.segmenter.send({ image: sourceCanvas });
                });
            } else {
                this._drawFallbackMatting(sourceCtx, outputCtx, width, height, 'transparent', '#00FF00', null);
            }

            const frameBlob = await new Promise(res => outputCanvas.toBlob(res, 'image/png'));
            const blobUrl = URL.createObjectURL(frameBlob);
            extractedImages.push(blobUrl);

            const elapsedSec = (Date.now() - frameStart) / 1000.0;
            const logLine = `[${i + 1}/${numFrames}] Frame ${Math.floor(seekTime * 30)} -> frame_${String(i + 1).padStart(3, '0')}.png (${elapsedSec.toFixed(2)}s)`;
            logMessages.push(logLine);

            if (logCallback) logCallback(logLine);
            if (progressCallback) {
                progressCallback({
                    status: 'processing',
                    current_frame: i + 1,
                    total_frames: numFrames,
                    percent: Math.floor(((i + 1) / numFrames) * 100)
                });
            }
        }

        return {
            extracted_images: extractedImages,
            log_messages: logMessages,
            download_url: extractedImages[0]
        };
    }

    /**
     * Process video frame-by-frame with White De-Fringing & Edge Refinement
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
        const outputCtx = outputCanvas.getContext('2d', { willReadFrequently: true });

        const stream = outputCanvas.captureStream(fps);
        let mimeType = 'video/webm;codecs=vp9';
        if (!MediaRecorder.isTypeSupported(mimeType)) mimeType = 'video/webm';
        if (!MediaRecorder.isTypeSupported(mimeType)) mimeType = 'video/mp4';

        const recorder = new MediaRecorder(stream, { mimeType, videoBitsPerSecond: 8000000 });
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

                        outputCtx.drawImage(results.segmentationMask, 0, 0, width, height);
                        outputCtx.globalCompositeOperation = 'source-in';
                        outputCtx.drawImage(results.image, 0, 0, width, height);

                        outputCtx.restore();

                        const imgData = outputCtx.getImageData(0, 0, width, height);
                        const pixels = imgData.data;
                        for (let i = 0; i < pixels.length; i += 4) {
                            const r = pixels[i];
                            const g = pixels[i + 1];
                            const b = pixels[i + 2];
                            const a = pixels[i + 3];

                            if (a > 0) {
                                if (r > 230 && g > 230 && b > 230) {
                                    pixels[i + 3] = 0;
                                } else if (r > 195 && g > 195 && b > 195) {
                                    const brightness = (r + g + b) / 3.0;
                                    const factor = (230.0 - brightness) / 35.0;
                                    pixels[i + 3] = Math.floor(a * Math.max(0, factor));
                                }
                            }
                        }
                        outputCtx.putImageData(imgData, 0, 0);

                        if (bgType !== 'transparent') {
                            outputCtx.save();
                            outputCtx.globalCompositeOperation = 'destination-over';
                            this._drawBackground(outputCtx, sourceCanvas, width, height, bgType, bgColor, bgImageElement, blurRadius);
                            outputCtx.restore();
                        }

                        resolve();
                    });
                    this.segmenter.send({ image: sourceCanvas });
                });
            } else {
                this._drawFallbackMatting(sourceCtx, outputCtx, width, height, bgType, bgColor, bgImageElement);
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

            if (r > 230 && g > 230 && b > 230) {
                data[i + 3] = 0;
            } else if (g > 100 && g > r * 1.4 && g > b * 1.4) {
                data[i + 3] = 0;
            }
        }
        outputCtx.putImageData(frameData, 0, 0);

        if (bgType !== 'transparent') {
            outputCtx.save();
            outputCtx.globalCompositeOperation = 'destination-over';
            this._drawBackground(outputCtx, sourceCanvas, width, height, bgType, bgColor, bgImageElement, 15);
            outputCtx.restore();
        }
    }
}

window.clientVideoProcessor = new ClientVideoProcessor();
