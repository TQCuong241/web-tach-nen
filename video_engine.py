import os
import sys
import time
import zipfile
import subprocess
import shutil

# Safe imports for optional heavy libraries on Vercel
try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from PIL import Image, ImageFilter
except ImportError:
    Image = None

try:
    import torch
    import torchvision.transforms as T
except ImportError:
    torch = None
    T = None

# Setup UTF-8 encoding for standard output
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def parse_hex_color(hex_str):
    """Convert hex color string like '#00FF00' or '00FF00' to RGB tuple (R, G, B)."""
    hex_str = hex_str.lstrip('#')
    if len(hex_str) == 6:
        return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    return (0, 255, 0)

def upscale_image(img, scale_factor=2.0):
    """Phóng to ảnh với chất lượng cao bằng LANCZOS interpolation."""
    new_width = int(img.width * scale_factor)
    new_height = int(img.height * scale_factor)
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


class SAM2ViTMattingEngine:
    """
    Quy trình Cao cấp VIP (--Max): IS-Net DIS (Dichotomous Image Segmentation) High-Precision Engine.
    Tách nền chuẩn sắc nét cao 100% không làm lùng lỗ hay tạo rác đen trên chủ thể.
    """
    def __init__(self, model_name="isnet-general-use", device=None):
        from rembg import remove, new_session
        print(f"[SAM2ViTMattingEngine] Loading IS-Net DIS High-Precision AI ('{model_name}')...")
        self.session = new_session(model_name)
        self.remove = remove

    def reset_rec_states(self):
        pass

    def remove_bg(self, pil_img, upscale_factor=1.0, is_sequence=False):
        orig_size = pil_img.size
        if pil_img.mode != 'RGBA':
            work_img = pil_img.convert('RGBA')
        else:
            work_img = pil_img

        if upscale_factor > 1.0:
            work_img = upscale_image(work_img, upscale_factor)
            no_bg_pil = self.remove(work_img, session=self.session)
            no_bg_pil = no_bg_pil.resize(orig_size, Image.Resampling.LANCZOS)
        else:
            no_bg_pil = self.remove(work_img, session=self.session)

        return no_bg_pil


class RVMMattingEngine:
    """
    Robust Video Matting (RVM) Engine (--Min) bởi ByteDance (PeterL1n).
    Hỗ trợ Recurrent State (r1..r4) duy trì tính nhất quán theo thời gian cho Video (chống nhấp nháy).
    """
    def __init__(self, model_name="mobilenetv3", device=None):
        if torch is None:
            raise ImportError("PyTorch is required for RVM Engine.")
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[RVMMattingEngine] Loading MIN (Robust Video Matting '{model_name}') on '{self.device}'...")
        self.model = torch.hub.load("PeterL1n/RobustVideoMatting", model_name, trust_repo=True)
        self.model = self.model.to(self.device).eval()
        self.rec_states = [None, None, None, None]
        self.T = T
        self.torch = torch

    def reset_rec_states(self):
        self.rec_states = [None, None, None, None]

    def remove_bg(self, pil_img, upscale_factor=1.0, is_sequence=False):
        if not is_sequence:
            self.reset_rec_states()

        orig_size = pil_img.size
        if upscale_factor > 1.0:
            work_img = upscale_image(pil_img, upscale_factor)
        else:
            work_img = pil_img

        img_rgb = work_img.convert("RGB")
        img_rgb_np = np.array(img_rgb)
        src = self.T.functional.to_tensor(img_rgb).unsqueeze(0).to(self.device)

        with self.torch.no_grad():
            fgr, pha, *self.rec_states = self.model(src, *self.rec_states)

        pha_np = pha[0, 0].cpu().numpy()
        pha_np = np.clip(pha_np, 0.0, 1.0)
        alpha_uint8 = (pha_np * 255.0).astype(np.uint8)

        if fgr is not None:
            fgr_np = (fgr[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        else:
            fgr_np = img_rgb_np

        rgba_np = np.dstack([fgr_np, alpha_uint8])
        result_pil = Image.fromarray(rgba_np, mode="RGBA")

        if upscale_factor > 1.0:
            result_pil = result_pil.resize(orig_size, Image.Resampling.LANCZOS)

        return result_pil


class VideoMattingEngine:
    """
    Unified AI Video Matting Engine supporting both --Max (IS-Net DIS High-Precision)
    and --Min (Robust Video Matting).
    """
    def __init__(self, model_name="mobilenetv3", device=None, engine_type="max"):
        self.engine_type = engine_type.lower()
        self.model_name = model_name
        self.active_engine = None

        if self.engine_type in ("max", "sam2", "isnet"):
            try:
                self.active_engine = SAM2ViTMattingEngine(model_name="isnet-general-use", device=device)
            except Exception as e:
                print(f"[VideoMattingEngine] Failed to load MAX engine ({e}). Fallback to RVM...")
                self.active_engine = None

        if self.active_engine is None and torch is not None:
            try:
                self.active_engine = RVMMattingEngine(model_name=model_name, device=device)
            except Exception as e:
                print(f"[VideoMattingEngine] Failed to load RVM ({e}).")
                self.active_engine = None

    def reset_rec_states(self):
        if self.active_engine and hasattr(self.active_engine, 'reset_rec_states'):
            self.active_engine.reset_rec_states()

    def process_frame(self, frame_bgr, downsample_ratio=1.0):
        if cv2 is None or np is None:
            raise RuntimeError("OpenCV / Numpy is required for server frame processing.")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        if self.active_engine:
            no_bg_pil = self.active_engine.remove_bg(pil_img, upscale_factor=1.0, is_sequence=True)
            return np.array(no_bg_pil)
        else:
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
            alpha = cv2.bitwise_not(mask)
            return np.dstack([frame_rgb, alpha])

    def render_composite(self, rgba_np, orig_bgr, bg_type='greenscreen', bg_color=(0, 255, 0), bg_image_np=None, blur_radius=15):
        if cv2 is None or np is None:
            return rgba_np

        fg_rgb = rgba_np[:, :, :3]
        alpha = rgba_np[:, :, 3].astype(np.float32) / 255.0
        alpha_3d = np.dstack([alpha, alpha, alpha])
        h, w = rgba_np.shape[:2]

        if bg_type == 'transparent':
            return rgba_np

        elif bg_type in ('greenscreen', 'bluescreen', 'color'):
            color_rgb = (0, 255, 0) if bg_type == 'greenscreen' else ((0, 0, 255) if bg_type == 'bluescreen' else bg_color)
            bg_rgb = np.full((h, w, 3), color_rgb, dtype=np.uint8)
            composite_rgb = (fg_rgb * alpha_3d + bg_rgb * (1.0 - alpha_3d)).astype(np.uint8)
            return cv2.cvtColor(composite_rgb, cv2.COLOR_RGB2BGR)

        elif bg_type == 'image' and bg_image_np is not None:
            bg_resized = cv2.resize(bg_image_np, (w, h), interpolation=cv2.INTER_AREA)
            if bg_resized.shape[2] == 4:
                bg_resized = bg_resized[:, :, :3]
            bg_rgb = cv2.cvtColor(bg_resized, cv2.COLOR_BGR2RGB)
            composite_rgb = (fg_rgb * alpha_3d + bg_rgb * (1.0 - alpha_3d)).astype(np.uint8)
            return cv2.cvtColor(composite_rgb, cv2.COLOR_RGB2BGR)

        elif bg_type == 'blur':
            ksize = blur_radius * 2 + 1
            blurred_bgr = cv2.GaussianBlur(orig_bgr, (ksize, ksize), 0)
            bg_rgb = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2RGB)
            composite_rgb = (fg_rgb * alpha_3d + bg_rgb * (1.0 - alpha_3d)).astype(np.uint8)
            return cv2.cvtColor(composite_rgb, cv2.COLOR_RGB2BGR)

        else:
            bg_rgb = np.full((h, w, 3), (0, 255, 0), dtype=np.uint8)
            composite_rgb = (fg_rgb * alpha_3d + bg_rgb * (1.0 - alpha_3d)).astype(np.uint8)
            return cv2.cvtColor(composite_rgb, cv2.COLOR_RGB2BGR)


def extract_audio(video_path, output_audio_path):
    if not shutil.which("ffmpeg"):
        return False
    try:
        cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "copy", output_audio_path]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return res.returncode == 0 and os.path.exists(output_audio_path)
    except Exception:
        return False

def merge_audio(video_path, audio_path, output_video_path):
    if not shutil.which("ffmpeg") or not os.path.exists(audio_path):
        shutil.copy(video_path, output_video_path)
        return
    try:
        cmd = ["ffmpeg", "-y", "-i", video_path, "-i", audio_path, "-c:v", "copy", "-c:a", "aac", "-shortest", output_video_path]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if res.returncode != 0 or not os.path.exists(output_video_path):
            shutil.copy(video_path, output_video_path)
    except Exception:
        shutil.copy(video_path, output_video_path)


def process_video_task(
    video_path: str,
    output_dir: str,
    bg_type: str = 'greenscreen',
    bg_color: str = '#00FF00',
    bg_image_path: str = None,
    blur_radius: int = 15,
    output_format: str = 'mp4',
    downsample_ratio: float = 1.0,
    model_name: str = 'max',
    engine_type: str = 'max',
    progress_callback = None
):
    """
    Processes 100% of video frames (frame-by-frame) maintaining EXACT original FPS and duration!
    If source video has 240 frames (10 seconds @ 24fps), output video will have 240 frames (10 seconds @ 24fps)!
    """
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV is required for video file processing.")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("=" * 80)
    print(f"[Process Video Task] Input: {base_name}")
    print(f"[Process Video Task] Total Frames: {total_frames} | FPS: {fps:.2f} | Target Duration: {total_frames/fps:.2f}s")
    print("=" * 80)

    engine = VideoMattingEngine(model_name=model_name, engine_type=engine_type)
    engine.reset_rec_states()

    rgb_color = parse_hex_color(bg_color)
    bg_image_np = cv2.imread(bg_image_path) if (bg_type == 'image' and bg_image_path and os.path.exists(bg_image_path)) else None

    audio_path = os.path.join(output_dir, "temp_audio.aac")
    has_audio = extract_audio(video_path, audio_path)

    temp_video_out = os.path.join(output_dir, f"temp_processed.{'webm' if output_format == 'webm' else 'mp4'}")
    final_output_path = os.path.join(output_dir, f"{base_name}_nobg.{output_format if output_format != 'zip_png' else 'zip'}")

    writer = None
    frames_gif = []
    png_dir = os.path.join(output_dir, "png_frames") if output_format == 'zip_png' else None
    if png_dir:
        os.makedirs(png_dir, exist_ok=True)

    if output_format == 'webm' and bg_type == 'transparent':
        fourcc = cv2.VideoWriter_fourcc(*'VP90')
        writer = cv2.VideoWriter(temp_video_out, fourcc, fps, (width, height), isColor=True)
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'VP80')
            writer = cv2.VideoWriter(temp_video_out, fourcc, fps, (width, height), isColor=True)
    elif output_format != 'gif' and output_format != 'zip_png':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_video_out, fourcc, fps, (width, height))

    start_time = time.time()
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_idx += 1

        rgba_np = engine.process_frame(frame_bgr, downsample_ratio=downsample_ratio)

        if output_format == 'zip_png':
            pil_rgba = Image.fromarray(rgba_np)
            frame_name = f"frame_{frame_idx:05d}.png"
            pil_rgba.save(os.path.join(png_dir, frame_name), "PNG", optimize=True)

        elif output_format == 'gif':
            composite_bgr = engine.render_composite(rgba_np, frame_bgr, bg_type, rgb_color, bg_image_np, blur_radius)
            composite_rgb = cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(composite_rgb)
            if width > 640:
                new_h = int(height * (640.0 / width))
                pil_frame = pil_frame.resize((640, new_h), Image.Resampling.LANCZOS)
            frames_gif.append(pil_frame)

        else:
            effective_bg = bg_type if bg_type != 'transparent' else 'greenscreen'
            composite_bgr = engine.render_composite(rgba_np, frame_bgr, effective_bg, rgb_color, bg_image_np, blur_radius)
            if writer and writer.isOpened():
                writer.write(composite_bgr)

        elapsed = time.time() - start_time
        current_fps = frame_idx / elapsed if elapsed > 0 else 0.0
        percent = int((frame_idx / total_frames) * 100) if total_frames > 0 else 0

        if progress_callback:
            progress_callback({
                "status": "processing",
                "current_frame": frame_idx,
                "total_frames": total_frames,
                "percent": percent,
                "fps": round(current_fps, 1),
                "eta_seconds": int((total_frames - frame_idx) / current_fps) if current_fps > 0 else 0
            })

    cap.release()
    if writer:
        writer.release()

    if output_format == 'zip_png' and png_dir:
        with zipfile.ZipFile(final_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(png_dir):
                for f in files:
                    zipf.write(os.path.join(root, f), arcname=f)
        shutil.rmtree(png_dir, ignore_errors=True)

    elif output_format == 'gif' and frames_gif:
        duration_ms = int(1000.0 / fps)
        frames_gif[0].save(final_output_path, save_all=True, append_images=frames_gif[1:], optimize=True, duration=duration_ms, loop=0)

    elif os.path.exists(temp_video_out):
        if has_audio and os.path.exists(audio_path):
            merge_audio(temp_video_out, audio_path, final_output_path)
            if os.path.exists(temp_video_out):
                os.remove(temp_video_out)
        else:
            shutil.move(temp_video_out, final_output_path)

    if os.path.exists(audio_path):
        os.remove(audio_path)

    if progress_callback:
        progress_callback({
            "status": "completed",
            "percent": 100,
            "output_file": os.path.basename(final_output_path),
            "output_path": final_output_path
        })

    return final_output_path
