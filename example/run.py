"""
TRIM — One-click launcher
=========================
Usage:
    python run.py

What it does:
    1. Checks Python dependencies, installs missing ones automatically
    2. Starts the FastAPI backend (app/web_server.py)
    3. Waits until the server is ready
    4. Opens http://localhost:8000 in your default browser
    5. Keeps running until you press Ctrl+C

Enhanced with:
    - Better error handling and user feedback
    - Dependency installation progress indication
    - Graceful shutdown handling
    - Port conflict detection
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import time
import webbrowser
import threading
import os
import socket
from pathlib import Path

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
HOST        = "127.0.0.1"
PORT        = 8000
URL         = f"http://{HOST}:{PORT}"
SERVER_FILE = Path(__file__).parent / "app" / "web_server.py"
REQUIRED    = ["fastapi", "uvicorn", "sse_starlette"]   # importable names
INSTALL_MAP = {                                          # importable → pip name
    "sse_starlette": "sse-starlette",
}


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────
def print_banner() -> None:
    """Print startup banner."""
    print("=" * 60)
    print("  TRIM — Trajectory Refinement for Emission Modeling")
    print("  Web Interface Launcher")
    print("=" * 60)


def check_port_available(port: int) -> bool:
    """Check if a port is available."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
        sock.close()
        return True
    except OSError:
        return False


def find_available_port(start_port: int = 8000, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port."""
    for i in range(max_attempts):
        port = start_port + i
        if check_port_available(port):
            return port
    raise RuntimeError(f"No available ports found in range {start_port}-{start_port + max_attempts - 1}")


# ─────────────────────────────────────────────
# Step 1: Auto-install missing dependencies
# ─────────────────────────────────────────────
def ensure_dependencies() -> None:
    """Check for and install missing dependencies."""
    missing = []
    for pkg in REQUIRED:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)

    if not missing:
        print("[✓] All dependencies are installed")
        return

    pip_names = [INSTALL_MAP.get(p, p) for p in missing]
    print(f"\n[!] Installing missing packages: {', '.join(pip_names)}")
    print("    This may take a moment...")

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", *pip_names],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("[✓] Dependencies installed successfully\n")
    except subprocess.CalledProcessError as e:
        print(f"\n[✗] Failed to install dependencies: {e}")
        print("    Please install manually:")
        print(f"    pip install {' '.join(pip_names)}\n")
        sys.exit(1)


# ─────────────────────────────────────────────
# Step 2: Start backend server
# ─────────────────────────────────────────────
def start_server(port: int) -> subprocess.Popen:
    """Start the FastAPI backend server."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent)  # ensure project root in path

    # Pass port as environment variable
    env["TRIM_PORT"] = str(port)

    # Start server with output visible for debugging
    proc = subprocess.Popen(
        [sys.executable, str(SERVER_FILE)],
        env=env,
        cwd=str(Path(__file__).parent),
        # Let output show in terminal for debugging
    )
    return proc


# ─────────────────────────────────────────────
# Step 3: Wait until server responds
# ─────────────────────────────────────────────
def wait_for_server(url: str, timeout: int = 30) -> bool:
    """Wait for server to respond to health check."""
    import urllib.request
    import urllib.error

    deadline = time.time() + timeout
    attempts = 0

    print(f"[*] Waiting for server at {url}/api/health", end="", flush=True)

    while time.time() < deadline:
        attempts += 1
        try:
            urllib.request.urlopen(f"{url}/api/health", timeout=2)
            print()  # New line after dots
            return True
        except urllib.error.URLError as e:
            # Server not ready yet
            if attempts % 3 == 0:
                print(".", end="", flush=True)
            time.sleep(0.5)
        except Exception as e:
            if attempts % 3 == 0:
                print(".", end="", flush=True)
            time.sleep(0.5)

    print()  # New line after dots
    return False


# ─────────────────────────────────────────────
# Step 4: Open browser
# ─────────────────────────────────────────────
def open_browser(url: str) -> None:
    """Open browser to the application URL."""
    # Small extra delay so the page loads cleanly
    time.sleep(0.5)
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"\n[!] Could not auto-open browser: {e}")
        print(f"    Please open {url} manually in your browser")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main() -> None:
    """Main launcher function."""
    print_banner()

    # 0. Check Python version
    if sys.version_info < (3, 8):
        print("\n[✗] Python 3.8 or higher is required")
        print(f"    Current version: {sys.version}")
        sys.exit(1)

    # 1. Check server file exists
    if not SERVER_FILE.exists():
        print(f"\n[✗] Cannot find: {SERVER_FILE}")
        print("    Make sure you are running this script from the TRIM root directory.")
        sys.exit(1)

    # 2. Dependencies
    print("\n[1/4] Checking dependencies...")
    ensure_dependencies()

    # 3. Check port availability
    print("[2/4] Checking port availability...")
    global PORT, URL
    if not check_port_available(PORT):
        print(f"[!] Port {PORT} is already in use")
        try:
            PORT = find_available_port(PORT + 1)
            URL = f"http://{HOST}:{PORT}"
            print(f"[✓] Using alternative port: {PORT}")
        except RuntimeError as e:
            print(f"[✗] {e}")
            sys.exit(1)
    else:
        print(f"[✓] Port {PORT} is available")

    # 4. Start server
    print(f"\n[3/4] Starting backend server on port {PORT}...")
    proc = start_server(PORT)

    # Give server a moment to start
    time.sleep(2)

    # Check if process crashed immediately
    if proc.poll() is not None:
        print(f"\n[✗] Server process exited immediately with code {proc.returncode}")
        print("    Check the error messages above for details.")
        sys.exit(1)

    # 5. Wait for server to be ready
    print(f"[4/4] Waiting for server to respond (timeout: 30s)...")
    ready = wait_for_server(URL, timeout=30)

    if ready:
        print(f"\n{'='*60}")
        print(f"  ✓  Server is running at {URL}")
        print(f"{'='*60}")
        print("  Opening browser automatically...")
        print("  Press Ctrl+C to stop the server")
        print(f"{'='*60}\n")
        threading.Thread(target=open_browser, args=(URL,), daemon=True).start()
    else:
        print(f"\n[!] Server did not respond in time")
        print(f"    The server may still be starting up.")
        print(f"    Try opening {URL} manually in your browser.")
        print(f"    Or check for error messages above.\n")

        # Check if process is still running
        if proc.poll() is not None:
            print(f"[!] Server process has exited with code {proc.returncode}")
            print("    This indicates a startup error. Check messages above.\n")
            sys.exit(1)

    # Keep alive — forward Ctrl+C to subprocess
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n\n[!] Shutting down TRIM server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
            print("[✓] Server stopped gracefully")
        except subprocess.TimeoutExpired:
            print("[!] Force stopping server...")
            proc.kill()
            proc.wait()
            print("[✓] Server stopped")
        print("\nGoodbye! 👋\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[✗] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)