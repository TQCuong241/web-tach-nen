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
    Unified AI Video Matting Engine supporting --Max (IS-Net DIS) and --Min (RVM).
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


def create_sprite_sheet(frame_paths, target_size, output_path):
    """
    Tạo tệp Sprite Sheet PNG ma trận từ chuỗi N khung hình ảnh.
    """
    n = len(frame_paths)
    if n == 0 or Image is None:
        return None, 0, 0

    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    sheet_w = cols * target_size
    sheet_h = rows * target_size

    sprite_sheet = Image.new("RGBA", (sheet_w, sheet_h), (0, 0, 0, 0))
    for idx, fpath in enumerate(frame_paths):
        r = idx // cols
        c = idx % cols
        if os.path.exists(fpath):
            img = Image.open(fpath).convert("RGBA")
            if img.size != (target_size, target_size):
                img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
            sprite_sheet.paste(img, (c * target_size, r * target_size))
            img.close()

    sprite_sheet.save(output_path, "PNG", optimize=True)
    return output_path, cols, rows


def process_animation_sprites(
    video_path: str,
    output_dir: str,
    num_frames: int = 32,
    target_size: int = 256,
    upscale_factor: float = 2.0,
    engine_type: str = 'max',
    progress_callback = None
):
    """
    TOOL TẠO ANIMATION & SPRITE SHEET TỪ VIDEO (GAME ASSET STUDIO):
    1. Trích xuất đúng `num_frames` khung hình từ video.
    2. Tách nền từng frame bằng IS-Net DIS High-Precision AI (`--Max`).
    3. Căn khung vuông chuẩn `target_size x target_size` px.
    4. Ghép toàn bộ chuỗi frame thành tệp Sprite Sheet Grid PNG + Bộ ảnh Zip + Ảnh GIF đếm FPS.
    """
    if cv2 is None or np is None:
        raise RuntimeError("OpenCV is required for Animation Sprite Sheet processing.")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    print("=" * 80)
    print(f"[Animation Studio] Video: {base_name} | Total Frames: {frame_count}")
    print(f"[Animation Studio] Extracting {num_frames} frames @ {target_size}x{target_size}px (Upscale {upscale_factor}x)")
    print("=" * 80)

    if num_frames <= 1:
        indices = [0]
    else:
        indices = [int(i * (frame_count - 1) / (num_frames - 1)) for i in range(num_frames)]

    engine = VideoMattingEngine(model_name="isnet-general-use", engine_type=engine_type)
    engine.reset_rec_states()

    extracted_frame_paths = []
    log_messages = []
    frames_pil_gif = []

    start_time = time.time()
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_start_time = time.time()
        rgba_np = engine.process_frame(frame, upscale_factor=upscale_factor)

        # Căn vuông sprite chuẩn target_size x target_size
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
        cropped_resized = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_AREA)

        output_name = f"frame_{i+1:03d}.png"
        output_path = os.path.join(output_dir, output_name)

        out_pil = Image.fromarray(cropped_resized)
        out_pil.save(output_path, "PNG", optimize=True)

        frames_pil_gif.append(out_pil)
        extracted_frame_paths.append(output_path)

        elapsed = time.time() - frame_start_time
        log_line = f"[{i+1}/{num_frames}] Frame {idx} -> {output_name} ({elapsed:.2f}s)"
        print(log_line)
        log_messages.append(log_line)

        if progress_callback:
            progress_callback({
                "status": "processing",
                "current_frame": i + 1,
                "total_frames": num_frames,
                "percent": int(((i + 1) / num_frames) * 100),
                "log": log_line,
                "fps": round(1.0 / elapsed, 1) if elapsed > 0 else 0.0
            })

    cap.release()

    # 1. Tạo tệp Ma trận Sprite Sheet Grid PNG
    spritesheet_name = f"spritesheet_{base_name}_{num_frames}f_{target_size}px.png"
    spritesheet_path = os.path.join(output_dir, spritesheet_name)
    create_sprite_sheet(extracted_frame_paths, target_size, spritesheet_path)

    # 2. Đóng gói ZIP chuỗi frames PNG
    zip_filename = f"sprites_{base_name}_{num_frames}f.zip"
    zip_path = os.path.join(output_dir, zip_filename)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for fpath in extracted_frame_paths:
            zipf.write(fpath, arcname=os.path.basename(fpath))
        if os.path.exists(spritesheet_path):
            zipf.write(spritesheet_path, arcname=spritesheet_name)

    # 3. Tạo GIF xem trước Animation
    gif_name = f"anim_{base_name}.gif"
    gif_path = os.path.join(output_dir, gif_name)
    if frames_pil_gif:
        duration_ms = int(1000.0 / 12.0)
        frames_pil_gif[0].save(gif_path, save_all=True, append_images=frames_pil_gif[1:], optimize=True, duration=duration_ms, loop=0)

    print("\nDA HOAN THANH TAO SPRITE SHEET & ANIMATION!")

    return {
        "output_dir": output_dir,
        "extracted_files": [os.path.basename(p) for p in extracted_frame_paths],
        "spritesheet_url": f"/api/media-output/{spritesheet_name}",
        "spritesheet_name": spritesheet_name,
        "gif_url": f"/api/media-output/{gif_name}",
        "zip_file": zip_filename,
        "log_messages": log_messages,
        "num_frames": num_frames,
        "target_size": target_size
    }

process_video_task = process_animation_sprites
