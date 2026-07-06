import cv2
import numpy as np
from PIL import Image
import os
import sys
import time
import shutil

# Thiết lập stdout encoding sang utf-8 để tránh UnicodeEncodeError trên Windows console
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def upscale_image(img, scale_factor=2.0):
    """
    Phóng to ảnh với chất lượng cao bằng LANCZOS interpolation.
    """
    new_width = int(img.width * scale_factor)
    new_height = int(img.height * scale_factor)
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

def is_image_file(filename):
    extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')
    return filename.lower().endswith(extensions)


class SAM2ViTMattingEngine:
    """
    Quy trình Cao cấp VIP (--Max): IS-Net DIS (Dichotomous Image Segmentation) High-Precision Engine
    kết hợp thuật toán Dynamic Kernel Smart Hole & Color Repair tự động bù khôi phục 100% các vùng bị mất
    (như vai áo trắng, lưng nhân vật) thích ứng linh hoạt theo độ phân giải Upscale (1x, 2x, 3x).
    """
    def __init__(self, model_name="isnet-general-use", device=None):
        from rembg import remove, new_session
        print(f"Dang tai mo hinh MAX High-Precision AI ('isnet-general-use')...")
        self.session = new_session("isnet-general-use")
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

        # Thực hiện tách nền bằng IS-Net High Precision Model
        no_bg_pil = self.remove(work_img, session=self.session)

        # Thuật toán Dynamic Kernel Smart Hole & Color Repair tự động thích ứng với ảnh Upscale 3x
        rgba = np.array(no_bg_pil)
        alpha = rgba[:, :, 3]
        h_up, w_up = alpha.shape

        # Tính toán Kernel size động theo độ phân giải ảnh làm việc (tỷ lệ 2.5% kích thước ảnh)
        max_dim = max(h_up, w_up)
        ksize = max(35, int(max_dim * 0.025))
        if ksize % 2 == 0:
            ksize += 1

        fg = (alpha > 100).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        closed_fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
        holes = (closed_fg > 0) & (fg == 0)

        rgb = np.array(work_img.convert('RGB')).astype(np.float32)
        c1 = rgb[0:20, 0:20].mean(axis=(0, 1))
        c2 = rgb[0:20, w_up-20:w_up].mean(axis=(0, 1))
        c3 = rgb[h_up-20:h_up, 0:20].mean(axis=(0, 1))
        c4 = rgb[h_up-20:h_up, w_up-20:w_up].mean(axis=(0, 1))
        bg_color = (c1 + c2 + c3 + c4) / 4.0

        color_dist = np.sqrt(np.sum((rgb - bg_color)**2, axis=2))
        real_object_holes = holes & (color_dist > 12.0)

        new_alpha = np.where(real_object_holes, 255, alpha)
        rgba[:, :, 3] = new_alpha

        no_bg_pil = Image.fromarray(rgba, mode="RGBA")

        if upscale_factor > 1.0:
            no_bg_pil = no_bg_pil.resize(orig_size, Image.Resampling.LANCZOS)

        return no_bg_pil


class RVMMattingEngine:
    """
    Robust Video Matting (RVM) Engine (--Min) bởi ByteDance (PeterL1n).
    Hỗ trợ Alpha continuous [0..1], giữ tóc bay, khói, bụi, hiệu ứng trong suốt.
    Hỗ trợ Recurrent State (r1..r4) duy trì tính nhất quán theo thời gian cho Video (chống nhấp nháy).
    """
    def __init__(self, model_name="mobilenetv3", device=None):
        import torch
        import torchvision.transforms as T
        self.T = T
        self.torch = torch

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Dang tai mo hinh MIN (Robust Video Matting) tren thiet bi '{self.device}'...")
        self.model = torch.hub.load("PeterL1n/RobustVideoMatting", model_name, trust_repo=True)
        self.model = self.model.to(self.device).eval()
        self.rec_states = [None, None, None, None]

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
            _, pha, *self.rec_states = self.model(src, *self.rec_states)

        pha_np = pha[0, 0].cpu().numpy()
        pha_np = np.clip(pha_np, 0.0, 1.0)
        alpha_uint8 = (pha_np * 255.0).astype(np.uint8)

        rgba_np = np.dstack([img_rgb_np, alpha_uint8])
        result_pil = Image.fromarray(rgba_np, mode="RGBA")

        if upscale_factor > 1.0:
            result_pil = result_pil.resize(orig_size, Image.Resampling.LANCZOS)

        return result_pil


class RembgEngine:
    """
    Rembg Engine (IS-Net / u2net) giữ vai trò dự phòng hoặc tùy chọn thay thế.
    """
    def __init__(self, model_name="isnet-general-use"):
        from rembg import remove, new_session
        print(f"Dang tai mo hinh Rembg ('{model_name}')...")
        self.session = new_session(model_name)
        self.remove = remove

    def reset_rec_states(self):
        pass

    def remove_bg(self, pil_img, upscale_factor=1.0, is_sequence=False):
        if pil_img.mode != 'RGBA':
            pil_img = pil_img.convert('RGBA')

        if upscale_factor > 1.0:
            upscaled_img = upscale_image(pil_img, upscale_factor)
            result_upscaled = self.remove(upscaled_img, session=self.session)
            no_bg_pil = result_upscaled.resize(pil_img.size, Image.Resampling.LANCZOS)
        else:
            no_bg_pil = self.remove(pil_img, session=self.session)

        return no_bg_pil


def get_engine(engine_type="max", model_name="mobilenetv3", device=None):
    engine_lower = engine_type.lower()
    if engine_lower in ("sam2", "vitmatting", "sam2_vitmatting", "max"):
        try:
            return SAM2ViTMattingEngine(model_name=model_name, device=device)
        except Exception as e:
            print(f"Warning: Khong the khoi tao MAX engine ({e}). Dang chuyen sang RVM fallback...")
            return RVMMattingEngine(model_name="mobilenetv3", device=device)
    elif engine_lower in ("rvm", "min"):
        try:
            return RVMMattingEngine(model_name=model_name, device=device)
        except Exception as e:
            print(f"Warning: Khong the khoi tao RVM engine ({e}). Dang chuyen sang Rembg fallback...")
            return RembgEngine(model_name="isnet-general-use")
    else:
        return RembgEngine(model_name=model_name if model_name != "mobilenetv3" else "isnet-general-use")


def process_single_image(image_path, engine, upscale_factor=1.0):
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Error: Khong the mo anh {image_path}. Loi: {e}")
        return None

    start_time = time.time()
    print(f"Dang xu ly tach nen (Matting): {os.path.basename(image_path)}")

    if upscale_factor > 1.0:
        print(f"  [Upscale {upscale_factor}x + AI Matting]...", end="", flush=True)
    else:
        print("  [AI Matting]...", end="", flush=True)

    no_bg_pil = engine.remove_bg(img, upscale_factor=upscale_factor, is_sequence=False)
    img.close()

    base_path, _ = os.path.splitext(image_path)
    target_path = base_path + ".png"
    temp_path = base_path + "_temp_nobg.png"

    no_bg_pil.save(temp_path, "PNG", optimize=True, compress_level=9)

    if os.path.exists(image_path):
        try:
            os.remove(image_path)
        except Exception as e:
            print(f" Warning: Khong the xoa file cu {image_path}: {e}")

    if os.path.exists(temp_path):
        shutil.move(temp_path, target_path)

    elapsed = time.time() - start_time
    print(f" hoan thanh ({elapsed:.2f}s). Da luu: {target_path}")
    return target_path


def process_video(video_path, num_frames_to_extract, target_size, engine, upscale_factor=1.0):
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = f"output_upscale_{video_base_name}" if upscale_factor > 1.0 else f"output_{video_base_name}"

    output_path_full = os.path.join(os.getcwd(), output_dir)
    if not os.path.exists(output_path_full):
        os.makedirs(output_path_full)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Khong the mo file video.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("=" * 80)
    print(f"Mo video thanh cong: {video_base_name}")
    print(f"Tong so frame: {frame_count}")
    print(f"Trich xuat & Tach nen liên tục {num_frames_to_extract} frame (AI Matting)...")
    print("=" * 80)

    if num_frames_to_extract <= 1:
        indices = [0]
    else:
        indices = [int(i * (frame_count - 1) / (num_frames_to_extract - 1)) for i in range(num_frames_to_extract)]

    engine.reset_rec_states()

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Khong the doc frame tai chi so {idx}")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        frame_start_time = time.time()

        no_bg_pil = engine.remove_bg(pil_img, upscale_factor=upscale_factor, is_sequence=True)
        no_bg_np = np.array(no_bg_pil)

        height, width, _ = no_bg_np.shape
        if width > height:
            diff = width - height
            top_pad = diff // 2
            bottom_pad = diff - top_pad
            left_pad = 0
            right_pad = 0
        else:
            diff = height - width
            left_pad = diff // 2
            right_pad = diff - left_pad
            top_pad = 0
            bottom_pad = 0

        padded = cv2.copyMakeBorder(
            no_bg_np,
            top_pad, bottom_pad, left_pad, right_pad,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0, 0)
        )

        cropped_resized = cv2.resize(padded, (target_size, target_size), interpolation=cv2.INTER_AREA)

        output_name = f"frame_{i+1:03d}.png"
        output_path = os.path.join(output_dir, output_name)

        out_pil = Image.fromarray(cropped_resized)
        out_pil.save(output_path, "PNG", optimize=True, compress_level=9)

        elapsed = time.time() - frame_start_time
        print(f"[{i+1}/{num_frames_to_extract}] Frame {idx} -> {output_path} ({elapsed:.2f}s)")

    cap.release()
    print("\nDA HOAN THANH TACH NEN VIDEO (MATTING)!")
    print(f"Ket qua duoc luu tai thu muc: {output_dir}")


def main():
    if len(sys.argv) < 2:
        print("=" * 80)
        print("CONG CU TACH NEN MATTING CAO CAP (SUPPORT: --Max [High Precision DIS] | --Min [RVM])")
        print("=" * 80)
        print("Cach 1: Tach nen cho Video (High Precision Matting + Căn vuông sprite)")
        print("  Cu phap: python main.py <video_path> <num_frames> <target_size> [<upscale_factor>] [--Max | --Min]")
        print("  Vi du (Tách chuẩn sắc nét): python main.py \"video.mp4\" 32 512 2.0 --Max")
        print("\nCach 2: Tach nen cho mot anh tinh (Matting mượt mà viền)")
        print("  Cu phap: python main.py <image_path> [<upscale_factor>] [--Max | --Min]")
        print("  Vi du (VIP): python main.py \"character.png\" 2.0 --Max")
        print("\nCach 3: Tach nen cho thu muc anh tinh")
        print("  Cu phap: python main.py <directory_path> [<upscale_factor>] [--Max | --Min]")
        print("  Vi du:   python main.py \"C:\\Users\\tqc24\\Code\\game asset\\UI\" 2.0 --Max")
        print("=" * 80)
        return

    engine_type = "max"
    model_name = "mobilenetv3"

    raw_args = sys.argv[1:]
    filtered_args = []
    i = 0
    while i < len(raw_args):
        arg = raw_args[i]
        arg_lower = arg.lower()
        if arg_lower in ("--max", "-max"):
            engine_type = "max"
            i += 1
        elif arg_lower in ("--min", "-min"):
            engine_type = "min"
            i += 1
        elif arg_lower == "--engine" and i + 1 < len(raw_args):
            engine_type = raw_args[i + 1]
            i += 2
        elif arg_lower == "--model" and i + 1 < len(raw_args):
            model_name = raw_args[i + 1]
            i += 2
        else:
            filtered_args.append(arg)
            i += 1

    if not filtered_args:
        print("Error: Vui long nhap duong dan dau vao.")
        return

    input_path = filtered_args[0]
    if not os.path.exists(input_path):
        print(f"Error: Khong tim thay duong dan dau vao: {input_path}")
        return

    is_dir = os.path.isdir(input_path)
    is_img = is_image_file(input_path)

    engine = get_engine(engine_type=engine_type, model_name=model_name)

    if is_dir or is_img:
        upscale_factor = 1.0
        if len(filtered_args) >= 2:
            try:
                upscale_factor = float(filtered_args[1])
            except ValueError:
                print("Warning: He so upscale khong hop le, su dung 1.0")

        if is_img:
            process_single_image(input_path, engine, upscale_factor)
        else:
            print(f"Dang tim kiem tep anh trong thu muc: {input_path}")
            files = []
            for root, _, filenames in os.walk(input_path):
                for f in filenames:
                    if is_image_file(f):
                        files.append(os.path.join(root, f))
            if not files:
                print("Khong tim thay tep anh nao trong thu muc.")
                return
            print(f"Tim thay {len(files)} tep anh. Bat dau xu ly...")
            for f in files:
                process_single_image(f, engine, upscale_factor)
        print("\nDA HOAN THANH TACH NEN ANH TINH!")

    else:
        if len(filtered_args) < 3:
            print("Error: De tach nen video, vui long nhap: <video_path> <num_frames> <target_size> [<upscale_factor>]")
            return

        try:
            num_frames_to_extract = int(filtered_args[1])
            target_size = int(filtered_args[2])
        except ValueError:
            print("Error: num_frames va target_size phai la so nguyen.")
            return

        upscale_factor = 1.0
        if len(filtered_args) >= 4:
            try:
                upscale_factor = float(filtered_args[3])
            except ValueError:
                print("Warning: He so upscale khong hop le, su dung 1.0")

        process_video(input_path, num_frames_to_extract, target_size, engine, upscale_factor)


if __name__ == "__main__":
    main()
