import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

if __name__ == "__main__":
    import uvicorn
    print("=" * 80)
    print("  CYBERMATTE ANIMATION STUDIO - GAME ASSET & SPRITE SHEET GENERATOR")
    print("  WEB URL: http://localhost:8000")
    print("=" * 80)
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
