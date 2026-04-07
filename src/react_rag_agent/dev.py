import subprocess
import time
from pathlib import Path
from shutil import which


def run() -> None:
    root = Path(__file__).resolve().parents[2]
    web_dir = root / "web"

    npm_path = which("npm")
    if npm_path is None:
        raise RuntimeError(
            "Could not find 'npm' in PATH. Install Node.js/npm, then re-run 'uv run react-rag-dev'. "
            "You can still run backend only with 'uv run react-rag-api'."
        )

    if not web_dir.exists():
        raise RuntimeError(f"Frontend directory not found: {web_dir}")

    package_json = web_dir / "package.json"
    if not package_json.exists():
        raise RuntimeError(f"Frontend package.json not found: {package_json}")

    vite_bin = web_dir / "node_modules" / ".bin" / "vite"
    vite_bin_cmd = web_dir / "node_modules" / ".bin" / "vite.cmd"
    if not vite_bin.exists() and not vite_bin_cmd.exists():
        raise RuntimeError(
            "Frontend dependencies are missing. Run 'cd web && npm install' first, "
            "then re-run 'uv run react-rag-dev'."
        )

    api_process = None
    web_process = None

    try:
        api_process = subprocess.Popen(
            ["uv", "run", "react-rag-api"],
            cwd=root,
        )
        web_process = subprocess.Popen(
            [npm_path, "run", "dev"],
            cwd=web_dir,
        )
    except Exception:
        if api_process is not None and api_process.poll() is None:
            api_process.terminate()
        raise

    try:
        while True:
            assert api_process is not None
            assert web_process is not None
            if api_process.poll() is not None:
                raise RuntimeError("API process exited unexpectedly")
            if web_process.poll() is not None:
                raise RuntimeError(
                    "Web process exited unexpectedly. Check frontend logs above "
                    "(common fix: run 'cd web && npm install')."
                )
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if api_process is not None and api_process.poll() is None:
            api_process.terminate()
        if web_process is not None and web_process.poll() is None:
            web_process.terminate()


if __name__ == "__main__":
    run()
