import os
import sys
import time
import zipfile
import subprocess
import shutil

# Safe imports for optional heavy libraries
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
    Quy trình Cao cấp VIP (--Max): IS-Net DIS (Dichotomous Image Segmentation) High-Precision Engine
    từ main copy.py của bạn. Bóc tách từng khung hình ảnh tĩnh (Frame-by-Frame Matting) sắc nét 100%.
    """
    def __init__(self, model_name="isnet-general-use", device=None):
        from rembg import remove, new_session
        print(f"[SAM2ViTMattingEngine] Loading MAX High-Precision AI ('{model_name}')...")
        self.session = new_session(model_name)
        self.remove = remove

    def reset_rec_states(self):
        pass

    def remove_bg(self, pil_img, upscale_factor=1.0, is_sequence=False):
        orig_size = pil_img.size
        if upscale_factor > 1.0:
            work_img = upscale_image(pil_img, upscale_factor)
        else:
            work_img = pil_img

        if work_img.mode != 'RGBA':
            work_img = work_img.convert('RGBA')

        # Thực hiện tách nền bằng IS-Net High Precision Model từ main copy.py
        no_bg_pil = self.remove(work_img, session=self.session)

        if upscale_factor > 1.0:
            no_bg_pil = no_bg_pil.resize(orig_size, Image.Resampling.LANCZOS)

        return no_bg_pil


class RVMMattingEngine:
    """
    Robust Video Matting (RVM) Engine (--Min) bởi ByteDance (PeterL1n).
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
    Unified AI Video Matting Engine supporting --Max (main copy.py IS-Net DIS) and --Min.
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

    def process_frame(self, frame_bgr, downsample_ratio=1.0, upscale_factor=1.0):
        if cv2 is None or np is None:
            raise RuntimeError("OpenCV / Numpy is required for server frame processing.")

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        if self.active_engine:
            no_bg_pil = self.active_engine.remove_bg(pil_img, upscale_factor=upscale_factor, is_sequence=True)
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


def process_video_frames_test(
    video_path: str,
    output_dir: str,
    num_frames_to_extract: int = 10,
    target_size: int = 1000,
    upscale_factor: float = 3.0,
    engine_type: str = 'max',
    progress_callback = None
):
    """
    TÁCH VÀ XUẤT 10 BỨC ẢNH TEST TỪ MAIN COPY.PY:
    Phân tích video -> Trích xuất & Tách nền đúng num_frames_to_extract bức ảnh PNG trong suốt -> Xuất log từng frame!
    """
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV is required for video frame processing.")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("=" * 80)
    print(f"Mo video thanh cong: {base_name}")
    print(f"Tong so frame: {frame_count}")
    print(f"Trich xuat & Tach nen {num_frames_to_extract} frame (AI Matting)...")
    print("=" * 80)

    if num_frames_to_extract <= 1:
        indices = [0]
    else:
        indices = [int(i * (frame_count - 1) / (num_frames_to_extract - 1)) for i in range(num_frames_to_extract)]

    engine = VideoMattingEngine(model_name="isnet-general-use", engine_type=engine_type)
    engine.reset_rec_states()

    extracted_files = []
    log_messages = []

    start_time = time.time()
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_start_time = time.time()
        rgba_np = engine.process_frame(frame, upscale_factor=upscale_factor)

        # Căn vuông sprite nếu target_size > 0
        if target_size > 0:
            h_f, w_f, _ = rgba_np.shape
            if w_f > h_f:
                diff = w_f - h_f
                top_pad, bottom_pad = diff // 2, diff - diff // 2
                left_pad, right_pad = 0, 0
            else:
                diff = h_f - w_f
                left_pad, right_pad = diff // 2, diff - diff // 2
                top_pad, bottom_pad = 0, 0

            padded = cv2.copyMakeBorder(rgba_np, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
            rgba_np = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_AREA)

        output_name = f"frame_{i+1:03d}.png"
        output_path = os.path.join(output_dir, output_name)

        out_pil = Image.fromarray(rgba_np)
        out_pil.save(output_path, "PNG", optimize=True)

        elapsed = time.time() - frame_start_time
        log_line = f"[{i+1}/{num_frames_to_extract}] Frame {idx} -> {output_name} ({elapsed:.2f}s)"
        print(log_line)
        log_messages.append(log_line)
        extracted_files.append(output_name)

        if progress_callback:
            progress_callback({
                "status": "processing",
                "current_frame": i + 1,
                "total_frames": num_frames_to_extract,
                "percent": int(((i + 1) / num_frames_to_extract) * 100),
                "log": log_line,
                "fps": round(1.0 / elapsed, 1) if elapsed > 0 else 0.0
            })

    cap.release()
    print("\nDA HOAN THANH TACH NEN 10 KHUNG HINH TEST!")

    zip_filename = f"{base_name}_10_frames.zip"
    zip_path = os.path.join(output_dir, zip_filename)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for fname in extracted_files:
            zipf.write(os.path.join(output_dir, fname), arcname=fname)

    return {
        "output_dir": output_dir,
        "extracted_files": extracted_files,
        "zip_file": zip_filename,
        "log_messages": log_messages
    }


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
    upscale_factor: float = 1.0,
    target_size: int = 0,
    progress_callback = None
):
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

    out_w, out_h = (target_size, target_size) if target_size > 0 else (width, height)

    writer = None
    frames_gif = []
    png_dir = os.path.join(output_dir, "png_frames") if output_format == 'zip_png' else None
    if png_dir:
        os.makedirs(png_dir, exist_ok=True)

    if output_format == 'webm' and bg_type == 'transparent':
        fourcc = cv2.VideoWriter_fourcc(*'VP90')
        writer = cv2.VideoWriter(temp_video_out, fourcc, fps, (out_w, out_h), isColor=True)
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'VP80')
            writer = cv2.VideoWriter(temp_video_out, fourcc, fps, (out_w, out_h), isColor=True)
    elif output_format != 'gif' and output_format != 'zip_png':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_video_out, fourcc, fps, (out_w, out_h))

    start_time = time.time()
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_idx += 1

        frame_start = time.time()
        rgba_np = engine.process_frame(frame_bgr, downsample_ratio=downsample_ratio, upscale_factor=upscale_factor)

        if target_size > 0:
            h_f, w_f, _ = rgba_np.shape
            if w_f > h_f:
                diff = w_f - h_f
                top_pad, bottom_pad = diff // 2, diff - diff // 2
                left_pad, right_pad = 0, 0
            else:
                diff = h_f - w_f
                left_pad, right_pad = diff // 2, diff - diff // 2
                top_pad, bottom_pad = 0, 0

            padded = cv2.copyMakeBorder(rgba_np, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
            rgba_np = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_AREA)

        if output_format == 'zip_png':
            pil_rgba = Image.fromarray(rgba_np)
            frame_name = f"frame_{frame_idx:05d}.png"
            pil_rgba.save(os.path.join(png_dir, frame_name), "PNG", optimize=True)

        elif output_format == 'gif':
            composite_bgr = engine.render_composite(rgba_np, frame_bgr, bg_type, rgb_color, bg_image_np, blur_radius)
            composite_rgb = cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(composite_rgb)
            if out_w > 640:
                new_h = int(out_h * (640.0 / out_w))
                pil_frame = pil_frame.resize((640, new_h), Image.Resampling.LANCZOS)
            frames_gif.append(pil_frame)

        else:
            effective_bg = bg_type if bg_type != 'transparent' else 'greenscreen'
            composite_bgr = engine.render_composite(rgba_np, frame_bgr, effective_bg, rgb_color, bg_image_np, blur_radius)
            if writer and writer.isOpened():
                writer.write(composite_bgr)

        elapsed = time.time() - start_time
        frame_elapsed = time.time() - frame_start
        current_fps = frame_idx / elapsed if elapsed > 0 else 0.0
        percent = int((frame_idx / total_frames) * 100) if total_frames > 0 else 0

        print(f"[{frame_idx}/{total_frames}] Frame {frame_idx-1} -> Processed ({frame_elapsed:.2f}s)")

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
