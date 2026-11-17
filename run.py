#!/usr/bin/env python3
"""Myeye - launcher. Usage: python run.py [--host H] [--port P]"""
import argparse
import os
import socket
import threading
import time
import webbrowser


def _free_port(host: str, port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return port
        except OSError:
            # the requested port is busy; let the OS pick a free one
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as alt:
                alt.bind((host, 0))
                return alt.getsockname()[1]


def _check_deps() -> None:
    required = {
        "cv2": "opencv-python", "numpy": "numpy", "fastapi": "fastapi",
        "uvicorn": "uvicorn", "multipart": "python-multipart",
    }
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except Exception:
            missing.append(package)
    if missing:
        print("\n  Missing dependencies: " + ", ".join(missing))
        print("  Install them with:  pip install -r requirements.txt\n")
        raise SystemExit(1)
    for module, package in (("ultralytics", "ultralytics"), ("lap", "lap"),
                            ("fast_plate_ocr", "fast-plate-ocr[onnx]")):
        try:
            __import__(module)
        except Exception:
            print(f"  Warning: {package} is not installed - that module will be unavailable.")


def main() -> None:
    _check_deps()
    ap = argparse.ArgumentParser(description="Myeye - All-in-One Video Analytics")
    ap.add_argument("--host", default=os.environ.get("MYEYE_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("MYEYE_PORT", "8000")))
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    import uvicorn
    from backend.app import app

    port = _free_port(args.host, args.port)
    url = f"http://{args.host}:{port}"
    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print("  ║   Myeye     ·  All-in-One Video Analytics        ║")
    print("  ╠══════════════════════════════════════════════════╣")
    print(f"  ║   Open the dashboard:  {url:<29s} ║")
    print("  ║   100% free · fully local · no cloud · unlimited ║")
    print("  ╚══════════════════════════════════════════════════╝")
    print()

    if not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(1.4),
                                         webbrowser.open(url)),
                         daemon=True).start()

    uvicorn.run(app, host=args.host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
