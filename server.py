import os
import sys
import uuid
import asyncio
import threading
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Safe imports for optional heavy libraries
try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import numpy as np
except ImportError:
    np = None

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from video_engine import process_video_task, process_video_frames_test, VideoMattingEngine, parse_hex_color

app = FastAPI(title="Professional Video & Image Background Removal Studio")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(BASE_DIR, "static")):
    parent_dir = os.path.dirname(BASE_DIR)
    if os.path.exists(os.path.join(parent_dir, "static")):
        BASE_DIR = parent_dir

# On Vercel or read-only environments, use /tmp
try:
    UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
except Exception:
    UPLOAD_DIR = "/tmp/uploads"
    os.makedirs(UPLOAD_DIR, exist_ok=True)

try:
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
except Exception:
    OUTPUT_DIR = "/tmp/outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Mount static files if directory exists
if os.path.exists(STATIC_DIR):
    try:
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    except Exception as e:
        print(f"[Static Mount Error] {e}")

# Task progress tracking dictionary
tasks_progress: Dict[str, Dict[str, Any]] = {}
active_websockets: Dict[str, list] = {}

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Server is running. index.html not found.</h1>")

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload input video or image and return metadata."""
    allowed_exts = ('.mp4', '.mov', '.avi', '.webm', '.mkv', '.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail="Invalid file format.")

    file_id = f"file_{uuid.uuid4().hex[:10]}"
    saved_filename = f"{file_id}{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_filename)

    with open(saved_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    is_image = ext in ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')
    width, height, fps, duration, total_frames = 1280, 720, 30.0, 10.0, 300
    thumb_filename = f"{file_id}_thumb.jpg"
    thumb_path = os.path.join(UPLOAD_DIR, thumb_filename)

    if is_image and Image is not None:
        try:
            im = Image.open(saved_path)
            width, height = im.size
            duration, total_frames, fps = 0.0, 1, 0.0
            im.thumbnail((320, 320))
            im.convert('RGB').save(thumb_path, 'JPEG')
        except Exception:
            pass

    elif not is_image and cv2 is not None:
        try:
            cap = cv2.VideoCapture(saved_path)
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0 or fps != fps:
                    fps = 30.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = total_frames / fps if fps > 0 else 0.0
                cap.set(cv2.CAP_PROP_POS_FRAMES, min(5, total_frames - 1))
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(thumb_path, cv2.resize(frame, (320, int(320 * height / width)) if width > 0 else (320, 180)))
                cap.release()
        except Exception:
            pass

    size_mb = round(os.path.getsize(saved_path) / (1024 * 1024), 2) if os.path.exists(saved_path) else 0.0

    return {
        "file_id": file_id,
        "filename": file.filename,
        "saved_filename": saved_filename,
        "is_image": is_image,
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "duration": round(duration, 2),
        "total_frames": total_frames,
        "size_mb": size_mb,
        "media_url": f"/api/media/{saved_filename}",
        "thumb_url": f"/api/media/{thumb_filename}" if os.path.exists(thumb_path) else None
    }

@app.post("/api/upload-bg")
async def upload_bg_image(file: UploadFile = File(...)):
    """Upload custom background image."""
    allowed_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail="Invalid image format.")

    bg_id = f"bg_{uuid.uuid4().hex[:10]}"
    saved_filename = f"{bg_id}{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_filename)

    with open(saved_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    return {
        "bg_id": bg_id,
        "bg_filename": saved_filename,
        "bg_url": f"/api/media/{saved_filename}"
    }

@app.post("/api/process-test-10")
async def start_10_frames_test(
    file_id: str = Form(...),
    num_frames: int = Form(10),
    target_size: int = Form(1000),
    upscale_factor: float = Form(3.0),
    model_name: str = Form("max")
):
    """Process 10-frame extraction & matting test with live console logs from main copy.py."""
    video_path = None
    for f in os.listdir(UPLOAD_DIR):
        if f.startswith(file_id) and not f.endswith("_thumb.jpg"):
            video_path = os.path.join(UPLOAD_DIR, f)
            break

    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found.")

    task_id = f"task_test_{uuid.uuid4().hex[:10]}"
    tasks_progress[task_id] = {
        "status": "queued",
        "percent": 0,
        "current_frame": 0,
        "total_frames": num_frames,
        "logs": [],
        "extracted_images": []
    }

    def run_worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        def progress_cb(data):
            if "log" in data:
                tasks_progress[task_id]["logs"].append(data["log"])
            tasks_progress[task_id].update(data)
            if task_id in active_websockets:
                for ws in active_websockets[task_id]:
                    try:
                        loop.run_until_complete(ws.send_json(tasks_progress[task_id]))
                    except Exception:
                        pass

        try:
            res = process_video_frames_test(
                video_path=video_path,
                output_dir=OUTPUT_DIR,
                num_frames_to_extract=num_frames,
                target_size=target_size,
                upscale_factor=upscale_factor,
                engine_type=model_name,
                progress_callback=progress_cb
            )
            tasks_progress[task_id]["status"] = "completed"
            tasks_progress[task_id]["percent"] = 100
            tasks_progress[task_id]["extracted_images"] = [f"/api/media-output/{f}" for f in res["extracted_files"]]
            tasks_progress[task_id]["download_url"] = f"/api/download/{res['zip_file']}"
        except Exception as e:
            print(f"[Test Worker Error] {e}")
            tasks_progress[task_id]["status"] = "failed"
            tasks_progress[task_id]["error"] = str(e)

    thread = threading.Thread(target=run_worker, daemon=True)
    thread.start()

    return {"task_id": task_id, "status": "started"}

@app.post("/api/process-image")
async def process_single_image_endpoint(
    file_id: str = Form(...),
    bg_type: str = Form("transparent"),
    bg_color: str = Form("#00FF00"),
    upscale_factor: float = Form(1.0),
    model_name: str = Form("max")
):
    """Process single image background matting (IS-Net High Precision DIS)."""
    image_path = None
    for f in os.listdir(UPLOAD_DIR):
        if f.startswith(file_id):
            image_path = os.path.join(UPLOAD_DIR, f)
            break

    if not image_path or not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image file not found.")

    engine = VideoMattingEngine(model_name=model_name, engine_type=model_name)
    
    try:
        pil_img = Image.open(image_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open image: {e}")

    no_bg_rgba = engine.process_frame(np.array(pil_img.convert('RGB'))[:, :, ::-1])
    
    output_filename = f"{file_id}_nobg.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    pil_out = Image.fromarray(no_bg_rgba)
    if upscale_factor > 1.0:
        new_w = int(pil_out.width * upscale_factor)
        new_h = int(pil_out.height * upscale_factor)
        pil_out = pil_out.resize((new_w, new_h), Image.Resampling.LANCZOS)

    pil_out.save(output_path, "PNG", optimize=True)

    return {
        "status": "completed",
        "output_media_url": f"/api/media-output/{output_filename}",
        "download_url": f"/api/download/{output_filename}",
        "output_filename": output_filename
    }

@app.post("/api/process")
async def start_processing(
    file_id: str = Form(...),
    bg_type: str = Form("greenscreen"),
    bg_color: str = Form("#00FF00"),
    bg_image_id: str = Form(None),
    blur_radius: int = Form(15),
    output_format: str = Form("mp4"),
    downsample_ratio: float = Form(1.0),
    model_name: str = Form("max"),
    upscale_factor: float = Form(1.0),
    target_size: int = Form(0)
):
    """Start background removal processing task."""
    video_path = None
    for f in os.listdir(UPLOAD_DIR):
        if f.startswith(file_id) and not f.endswith("_thumb.jpg"):
            video_path = os.path.join(UPLOAD_DIR, f)
            break

    if not video_path or not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found. Please upload video first.")

    bg_image_path = None
    if bg_image_id:
        for f in os.listdir(UPLOAD_DIR):
            if f.startswith(bg_image_id):
                bg_image_path = os.path.join(UPLOAD_DIR, f)
                break

    task_id = f"task_{uuid.uuid4().hex[:10]}"
    tasks_progress[task_id] = {
        "status": "queued",
        "percent": 0,
        "current_frame": 0,
        "total_frames": 0,
        "fps": 0,
        "eta_seconds": 0,
        "output_file": None
    }

    def run_worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        def progress_cb(data):
            tasks_progress[task_id].update(data)
            if task_id in active_websockets:
                for ws in active_websockets[task_id]:
                    try:
                        loop.run_until_complete(ws.send_json(tasks_progress[task_id]))
                    except Exception:
                        pass

        try:
            output_file = process_video_task(
                video_path=video_path,
                output_dir=OUTPUT_DIR,
                bg_type=bg_type,
                bg_color=bg_color,
                bg_image_path=bg_image_path,
                blur_radius=blur_radius,
                output_format=output_format,
                downsample_ratio=downsample_ratio,
                model_name=model_name,
                engine_type=model_name,
                upscale_factor=upscale_factor,
                target_size=target_size,
                progress_callback=progress_cb
            )
            tasks_progress[task_id]["status"] = "completed"
            tasks_progress[task_id]["percent"] = 100
            tasks_progress[task_id]["download_url"] = f"/api/download/{os.path.basename(output_file)}"
            tasks_progress[task_id]["output_media_url"] = f"/api/media-output/{os.path.basename(output_file)}"
        except Exception as e:
            print(f"[Worker Error] Task {task_id} failed: {e}")
            tasks_progress[task_id]["status"] = "failed"
            tasks_progress[task_id]["error"] = str(e)

    thread = threading.Thread(target=run_worker, daemon=True)
    thread.start()

    return {"task_id": task_id, "status": "started"}

@app.get("/api/status/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks_progress:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks_progress[task_id]

@app.websocket("/ws/progress/{task_id}")
async def websocket_progress(websocket: WebSocket, task_id: str):
    await websocket.accept()
    if task_id not in active_websockets:
        active_websockets[task_id] = []
    active_websockets[task_id].append(websocket)

    try:
        if task_id in tasks_progress:
            await websocket.send_json(tasks_progress[task_id])
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if task_id in active_websockets and websocket in active_websockets[task_id]:
            active_websockets[task_id].remove(websocket)

@app.get("/api/media/{filename}")
async def serve_uploaded_media(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/api/media-output/{filename}")
async def serve_output_media(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename, media_type="application/octet-stream")
    raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    import uvicorn
    print("=" * 80)
    print("  KHOI CHAY PROFESSIONAL VIDEO & IMAGE BACKGROUND REMOVER WEB STUDIO")
    print("  URL: http://localhost:8000")
    print("=" * 80)
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
