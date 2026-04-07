import subprocess
import time
from pathlib import Path


def run() -> None:
    root = Path(__file__).resolve().parents[2]
    web_dir = root / "web"

    api_process = subprocess.Popen(
        ["uv", "run", "react-rag-api"],
        cwd=root,
    )
    web_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=web_dir,
    )

    try:
        while True:
            if api_process.poll() is not None:
                raise RuntimeError("API process exited unexpectedly")
            if web_process.poll() is not None:
                raise RuntimeError("Web process exited unexpectedly")
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if api_process.poll() is None:
            api_process.terminate()
        if web_process.poll() is None:
            web_process.terminate()


if __name__ == "__main__":
    run()
