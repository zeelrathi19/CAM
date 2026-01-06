# run.py - Run both FastAPI and Streamlit together

import subprocess
import sys
import time
import signal
import os

processes = []

def cleanup(signum=None, frame=None):
    """Kill all child processes on exit"""
    print("\n🛑 Shutting down...")
    for p in processes:
        try:
            p.terminate()
            p.wait(timeout=5)
        except:
            p.kill()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("🚀 Starting L-Shape Detection App...")
    print("=" * 40)

    # Start FastAPI backend
    print("📡 Starting FastAPI backend on port 8000...")
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    processes.append(api_process)

    # Wait for API to start
    time.sleep(3)

    # Start Streamlit frontend
    print("🎨 Starting Streamlit frontend on port 8501...")
    streamlit_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "frontend.py", "--server.port", "8501"],
    )
    processes.append(streamlit_process)

    print("=" * 40)
    print("✅ Both services are running!")
    print("   📡 API:      http://localhost:8000")
    print("   🎨 Frontend: http://localhost:8501")
    print("=" * 40)
    print("Press Ctrl+C to stop both services")

    # Wait for processes
    try:
        while True:
            time.sleep(1)
            # Check if any process died
            if api_process.poll() is not None:
                print("❌ API process stopped unexpectedly")
                cleanup()
            if streamlit_process.poll() is not None:
                print("❌ Streamlit process stopped unexpectedly")
                cleanup()
    except KeyboardInterrupt:
        cleanup()
