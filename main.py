import os
import sys
import argparse

# Force stdout encoding to UTF-8 to prevent Windows cp1258/cp1252 charmap encoding errors
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def launch_web_server(host="0.0.0.0", port=8000):
    """Launch the CyberMatte AI FastAPI Web Studio."""
    import uvicorn
    print("=" * 80)
    print("  CYBERMATTE AI - STUDIO TACH NEN VIDEO CHUYEN NGHIEP VIP")
    print(f"  WEB URL: http://localhost:{port}")
    print("=" * 80)
    uvicorn.run("server:app", host=host, port=port, reload=False)

def main():
    parser = argparse.ArgumentParser(
        description="CyberMatte AI - Studio Tách Nền Video Chuyên Nghiệp"
    )
    parser.add_argument("--web", action="store_true", help="Chạy ứng dụng Web Dashboard Studio")
    parser.add_argument("--port", type=int, default=8000, help="Port cho Web Server (Default: 8000)")
    parser.add_argument("input_path", nargs="?", help="Đường dẫn file video hoặc ảnh")
    parser.add_argument("bg_type", nargs="?", default="greenscreen", help="Loại nền: greenscreen, bluescreen, transparent, color")
    parser.add_argument("bg_color", nargs="?", default="#00FF00", help="Mã màu HEX cho phông nền")

    args, unknown = parser.parse_known_args()

    if args.web or not args.input_path:
        launch_web_server(port=args.port)
        return

    if args.input_path and os.path.exists(args.input_path):
        from video_engine import process_video_task
        print(f"Bat dau tach nen file: {args.input_path}")
        output_dir = os.path.join(os.getcwd(), "outputs")
        
        def progress_cb(info):
            percent = info.get("percent", 0)
            cur = info.get("current_frame", 0)
            tot = info.get("total_frames", 0)
            fps = info.get("fps", 0)
            print(f"\r[Progress: {percent}%] Frame {cur}/{tot} ({fps} FPS)", end="", flush=True)

        res_path = process_video_task(
            video_path=args.input_path,
            output_dir=output_dir,
            bg_type=args.bg_type,
            bg_color=args.bg_color,
            progress_callback=progress_cb
        )
        print(f"\nDa hoan tat tach nen! File luu tai: {res_path}")

if __name__ == "__main__":
    main()
