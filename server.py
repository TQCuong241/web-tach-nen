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

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from video_engine import process_video_task

app = FastAPI(title="Professional Video Background Removal Studio")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
    except Exception:
        pass

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
    """Upload input video and return metadata."""
    allowed_exts = ('.mp4', '.mov', '.avi', '.webm', '.mkv')
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail="Invalid video format. Supported: MP4, MOV, AVI, WEBM, MKV")

    file_id = f"vid_{uuid.uuid4().hex[:10]}"
    saved_filename = f"{file_id}{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_filename)

    with open(saved_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    width, height, fps, duration, total_frames = 1280, 720, 30.0, 10.0, 300
    thumb_filename = f"{file_id}_thumb.jpg"
    thumb_path = os.path.join(UPLOAD_DIR, thumb_filename)

    if cv2 is not None:
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

@app.post("/api/process")
async def start_processing(
    file_id: str = Form(...),
    bg_type: str = Form("greenscreen"),
    bg_color: str = Form("#00FF00"),
    bg_image_id: str = Form(None),
    blur_radius: int = Form(15),
    output_format: str = Form("mp4"),
    downsample_ratio: float = Form(1.0),
    model_name: str = Form("mobilenetv3")
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
    print("  KHOI CHAY PROFESSIONAL VIDEO BACKGROUND REMOVER WEB STUDIO")
    print("  URL: http://localhost:8000")
    print("=" * 80)
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
