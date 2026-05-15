from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import platform
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def _bootstrap(extension_root: Path) -> None:
    vendor = extension_root / "_vendor"
    for path in (extension_root, vendor):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Beyond Pixal3D</title>
  <style>
    :root { color-scheme: dark; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #191b20; color: #f3f5f8; }
    main { padding: 22px; max-width: 820px; margin: 0 auto; }
    h1 { font-size: 22px; margin: 0 0 18px; font-weight: 650; }
    label { display: block; font-size: 12px; color: #aeb7c4; margin: 14px 0 6px; }
    input, select { box-sizing: border-box; width: 100%; border: 1px solid #3a414d; background: #101218; color: #f3f5f8; border-radius: 6px; padding: 10px; }
    .row { display: grid; grid-template-columns: 1fr 116px; gap: 12px; align-items: end; }
    .split { display: grid; grid-template-columns: 1fr 110px 130px; gap: 12px; align-items: end; }
    .split.options { grid-template-columns: 1fr 1fr 1fr; }
    .actions { margin-top:18px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
    .toggle { display:flex; align-items:center; gap:8px; color:#c7d0dd; font-size:13px; }
    .toggle input { width:auto; }
    button { border: 0; background: #e8edf7; color: #101218; border-radius: 6px; padding: 10px 13px; font-weight: 650; cursor: pointer; }
    button.secondary { background: #2b313b; color: #f3f5f8; border: 1px solid #4a5260; }
    button:disabled { opacity: .5; cursor: default; }
    .stage { margin-top: 18px; }
    .preview { min-height: 230px; border: 1px solid #303743; border-radius: 6px; background: #101218; overflow: hidden; position: relative; }
    .preview img { width: 100%; height: 100%; object-fit: contain; display: block; opacity: .98; }
    body.is-running .preview img { opacity: .72; filter: saturate(.9) contrast(.96); }
    .preview-empty { position:absolute; inset:0; display:grid; place-items:center; color:#6f7a89; font-size:12px; }
    .processing-overlay { position:absolute; inset:0; display:none; place-items:center; perspective: 650px; background: rgba(8, 12, 18, .28); }
    body.is-running .processing-overlay { display:grid; }
    .processing-overlay::before { content:""; position:absolute; inset:0; background: radial-gradient(circle at 50% 44%, rgba(132, 191, 255, .18), transparent 52%); opacity:.9; }
    .cube { width: 118px; height: 118px; transform-style: preserve-3d; animation: spin 7s linear infinite; opacity: .9; }
    .cube span { position:absolute; inset:0; border:1px solid rgba(166,210,255,.58); background-image: linear-gradient(rgba(166,210,255,.18) 1px, transparent 1px), linear-gradient(90deg, rgba(166,210,255,.18) 1px, transparent 1px); background-size: 19px 19px; box-shadow: inset 0 0 32px rgba(86, 161, 255, .1); }
    .cube span:nth-child(1) { transform: translateZ(59px); }
    .cube span:nth-child(2) { transform: rotateY(180deg) translateZ(59px); }
    .cube span:nth-child(3) { transform: rotateY(90deg) translateZ(59px); }
    .cube span:nth-child(4) { transform: rotateY(-90deg) translateZ(59px); }
    .cube span:nth-child(5) { transform: rotateX(90deg) translateZ(59px); }
    .cube span:nth-child(6) { transform: rotateX(-90deg) translateZ(59px); }
    .pulse { position:absolute; width: 8px; height: 8px; border-radius: 50%; background:#dcecff; box-shadow: 0 0 18px #8cc7ff; animation: scan 2.4s ease-in-out infinite; }
    .processing-label { position:absolute; left:12px; right:12px; bottom:10px; color:#d9e8f8; font-size:12px; text-align:center; text-shadow: 0 1px 8px #05070a; }
    @keyframes spin { from { transform: rotateX(-18deg) rotateY(0deg); } to { transform: rotateX(-18deg) rotateY(360deg); } }
    @keyframes scan { 0%,100% { transform: translate(-72px, 52px); opacity:.25; } 50% { transform: translate(72px, -52px); opacity:1; } }
    .console-bar { margin-top: 18px; display:flex; align-items:center; justify-content:space-between; gap:12px; }
    .console-title { color: #aeb7c4; font-size: 12px; }
    .copy-button { padding: 7px 10px; font-size: 12px; }
    #log { white-space: pre-wrap; height: 320px; margin-top: 6px; padding: 14px; background: #101218; border: 1px solid #303743; border-radius: 6px; color: #d9e1ed; overflow: auto; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; line-height: 1.45; }
    .status { margin-top: 12px; color: #aeb7c4; font-size: 12px; min-height: 18px; }
  </style>
</head>
<body>
  <main>
    <h1>Beyond Pixal3D</h1>
    <label>Input image</label>
    <div class="row"><input id="image" placeholder="/path/to/image.png"><button class="secondary" onclick="chooseImage()">Choose</button></div>
    <label>Output GLB</label>
    <div class="row"><input id="output" value="~/Pixal3D_Outputs/pixal3d_asset.glb"><button class="secondary" onclick="chooseOutput()">Save As</button></div>
    <div class="split">
      <div>
        <label>Model</label>
        <input id="model" value="TencentARC/Pixal3D">
      </div>
      <div>
        <label>Seed</label>
        <input id="seed" type="number" value="42">
      </div>
      <div>
        <label>Device</label>
        <select id="device" onchange="syncPrepareButton()">
          <option value="auto">Auto</option>
          <option value="cuda">CUDA</option>
          <option value="mps">Metal</option>
          <option value="cpu">CPU</option>
        </select>
      </div>
    </div>
    <div class="split options">
      <div>
        <label>Low-poly face target</label>
        <input id="decimationTarget" type="number" value="1000000" min="0" step="1000">
      </div>
      <div>
        <label>Target resolution</label>
        <select id="targetResolution">
          <option value="1024">1024</option>
          <option value="1536" selected>1536</option>
        </select>
      </div>
      <div>
        <label>PBR texture size</label>
        <select id="textureSize">
          <option value="1024">1024</option>
          <option value="2048">2048</option>
          <option value="4096" selected>4096</option>
        </select>
      </div>
    </div>
    <div class="actions">
      <button id="generateButton" onclick="startGenerate()">Generate GLB</button>
      <button class="secondary" onclick="status()">Check Runtime</button>
      <button id="prepareButton" class="secondary" onclick="prepareModels()">Prepare Models</button>
      <label class="toggle"><input id="importAfter" type="checkbox" checked> Import into Blender</label>
      <label class="toggle"><input id="mpsFallback" type="checkbox" checked> Allow unsupported-op CPU fallback</label>
    </div>
    <div id="state" class="status"></div>
    <div class="stage">
      <div class="preview">
        <div id="previewEmpty" class="preview-empty">No image selected</div>
        <img id="preview" alt="">
        <div class="processing-overlay" aria-hidden="true">
          <div class="cube">
            <span></span><span></span><span></span><span></span><span></span><span></span>
          </div>
          <div class="pulse"></div>
          <div class="processing-label">Generating GLB</div>
        </div>
      </div>
    </div>
    <div class="console-bar">
      <div class="console-title">Run console</div>
      <button class="secondary copy-button" onclick="copyLog()">Copy Log</button>
    </div>
    <div id="log">Ready.</div>
  </main>
  <script>
    let pollTimer = null;

    function log(message) {
      const node = document.getElementById('log');
      node.textContent = message || '';
      node.scrollTop = node.scrollHeight;
    }

    function setRunning(running) {
      document.getElementById('generateButton').disabled = running;
      document.getElementById('state').textContent = running ? 'Generation running...' : '';
      document.body.classList.toggle('is-running', running);
    }

    async function chooseImage() {
      const path = await pywebview.api.choose_image();
      if (path) {
        document.getElementById('image').value = path;
        await loadPreview(path);
      }
    }

    async function chooseOutput() {
      const current = document.getElementById('output').value;
      const path = await pywebview.api.choose_output(current);
      if (path) document.getElementById('output').value = path;
    }

    async function status() {
      log(await pywebview.api.status(document.getElementById('device').value || 'auto'));
    }

    async function prepareModels() {
      setRunning(true);
      log('Preparing open Pixal3D model assets...');
      const message = await pywebview.api.prepare_model_assets(document.getElementById('device').value || 'auto');
      log(message);
      setRunning(false);
    }

    async function syncPrepareButton() {
      const button = document.getElementById('prepareButton');
      const visible = await pywebview.api.prepare_model_assets_available(document.getElementById('device').value || 'auto');
      button.style.display = visible ? '' : 'none';
    }

    async function loadPreview(path) {
      const preview = document.getElementById('preview');
      const empty = document.getElementById('previewEmpty');
      if (!path) {
        preview.removeAttribute('src');
        empty.style.display = 'grid';
        return;
      }
      const result = await pywebview.api.image_data_url(path);
      if (result.ok) {
        preview.src = result.data_url;
        empty.style.display = 'none';
      } else {
        preview.removeAttribute('src');
        empty.style.display = 'grid';
      }
    }

    async function copyLog() {
      const message = await pywebview.api.copy_log(document.getElementById('log').textContent || '');
      document.getElementById('state').textContent = message;
    }

    function payload() {
      return {
        image: document.getElementById('image').value,
        output: document.getElementById('output').value,
        model: document.getElementById('model').value,
        seed: parseInt(document.getElementById('seed').value || '42', 10),
        device: document.getElementById('device').value || 'auto',
        decimation_target: parseInt(document.getElementById('decimationTarget').value || '0', 10),
        target_resolution: parseInt(document.getElementById('targetResolution').value || '1536', 10),
        texture_size: parseInt(document.getElementById('textureSize').value || '4096', 10),
        import_after_generate: document.getElementById('importAfter').checked,
        enable_mps_fallback: document.getElementById('mpsFallback').checked
      };
    }

    async function startGenerate() {
      if (pollTimer) clearInterval(pollTimer);
      setRunning(true);
      log('Starting Pixal3D...');
      await loadPreview(document.getElementById('image').value);
      const response = await pywebview.api.start_generation(payload());
      if (!response.ok) {
        setRunning(false);
        log(response.message || 'Could not start Pixal3D.');
        return;
      }
      log(response.message || 'Pixal3D is running.');
      pollTimer = setInterval(pollStatus, 1000);
      await pollStatus();
    }

    async function pollStatus() {
      const response = await pywebview.api.generation_status();
      if (response.log) log(response.log);
      if (response.running) {
        const stage = response.stage || 'Generation running';
        const elapsed = response.elapsed || '';
        document.getElementById('state').textContent = elapsed ? `${stage} (${elapsed})` : stage;
      }
      if (!response.running) {
        if (pollTimer) clearInterval(pollTimer);
        pollTimer = null;
        setRunning(false);
        if (response.message) document.getElementById('state').textContent = response.message;
      }
    }

    async function generate() {
      const payload = {
        image: document.getElementById('image').value,
        output: document.getElementById('output').value,
        model: document.getElementById('model').value,
        seed: parseInt(document.getElementById('seed').value || '42', 10),
        device: document.getElementById('device').value || 'auto',
        decimation_target: parseInt(document.getElementById('decimationTarget').value || '0', 10),
        target_resolution: parseInt(document.getElementById('targetResolution').value || '1536', 10),
        texture_size: parseInt(document.getElementById('textureSize').value || '4096', 10)
      };
      log('Running Pixal3D...');
      log(await pywebview.api.generate(payload));
    }

    window.addEventListener('pywebviewready', syncPrepareButton);
  </script>
</body>
</html>
"""


class Pixal3DApi:
    def __init__(self, extension_root: Path, session_id: str):
        self.extension_root = extension_root
        self.session_id = session_id
        self._job_lock = threading.Lock()
        self._job: dict = {
            "running": False,
            "returncode": None,
            "log": "Ready.",
            "message": "",
            "output": "",
            "stage": "",
            "started_at": None,
            "last_output_at": None,
        }

    def _state_path(self) -> Path:
        return self.extension_root / "wheels" / "webview_state.json"

    def _write_last_output(self, output_path: str, import_requested: bool) -> None:
        state = {
            "last_output_path": output_path,
            "import_requested": import_requested,
            "session_id": self.session_id,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        path = self._state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def _first_dialog_path(result) -> str:
        if not result:
            return ""
        if isinstance(result, (list, tuple)):
            return str(result[0]) if result else ""
        return str(result)

    def choose_image(self):
        import webview

        file_types = ("Images (*.png;*.jpg;*.jpeg;*.webp;*.bmp)", "All files (*.*)")
        result = webview.windows[0].create_file_dialog(webview.OPEN_DIALOG, allow_multiple=False, file_types=file_types)
        return self._first_dialog_path(result)

    def choose_output(self, current_path: str = ""):
        import webview

        expanded = Path(os.path.expanduser(str(current_path or "~/Pixal3D_Outputs/pixal3d_asset.glb")))
        directory = str(expanded.parent if expanded.parent else Path.home())
        filename = expanded.name or "pixal3d_asset.glb"
        file_types = ("glTF Binary (*.glb)", "All files (*.*)")
        result = webview.windows[0].create_file_dialog(
            webview.SAVE_DIALOG,
            directory=directory,
            allow_multiple=False,
            save_filename=filename,
            file_types=file_types,
        )
        return self._first_dialog_path(result)

    def status(self, device: str = "auto") -> str:
        from dependency_manager import get_runtime_status

        status = get_runtime_status(device)
        lines = [
            f"Pywebview ready: {status.webview_ready}",
            f"Generation ready: {status.generation_ready}",
            f"Platform: {status.platform_key}",
        ]
        if status.missing_generation_modules:
            lines.append("Missing generation modules: " + ", ".join(status.missing_generation_modules))
        lines.extend(status.unsupported_notes)
        return "\n".join(lines)

    def copy_log(self, text: str = "") -> str:
        value = str(text or "")
        if not value:
            return "No log to copy."
        try:
            if sys.platform == "darwin":
                subprocess.run(["pbcopy"], input=value, text=True, check=True)
            elif sys.platform == "win32":
                subprocess.run(["clip"], input=value, text=True, check=True)
            else:
                return "Log copying is not available on this platform."
        except Exception as error:
            return f"Could not copy log: {error}"
        return "Log copied."

    def image_data_url(self, path: str = "") -> dict:
        expanded = Path(os.path.expanduser(str(path or "").strip()))
        if not expanded.is_file():
            return {"ok": False, "message": "Image file does not exist."}
        mime = mimetypes.guess_type(str(expanded))[0] or "image/png"
        try:
            data = base64.b64encode(expanded.read_bytes()).decode("ascii")
        except Exception as error:
            return {"ok": False, "message": f"Could not read image: {error}"}
        return {"ok": True, "data_url": f"data:{mime};base64,{data}"}

    def prepare_model_assets_available(self, device: str = "auto") -> bool:
        from dependency_manager import open_model_asset_prep_available

        return open_model_asset_prep_available(device)

    def prepare_model_assets(self, device: str = "auto") -> str:
        from dependency_manager import prepare_open_model_assets

        ok, message = prepare_open_model_assets(device)
        prefix = "Model assets ready." if ok else "Model asset preparation failed."
        return f"{prefix}\n{message}"

    def _worker_command(self, payload: dict) -> tuple[str | None, str | None, list[str] | None, dict]:
        image = os.path.expanduser(str(payload.get("image", "")).strip())
        output = os.path.expanduser(str(payload.get("output", "")).strip())
        model = str(payload.get("model", "")).strip() or "TencentARC/Pixal3D"
        seed = int(payload.get("seed", 42) or 42)
        device = str(payload.get("device", "auto") or "auto")
        decimation_target = max(0, int(payload.get("decimation_target", 1000000) or 1000000))
        target_resolution = int(payload.get("target_resolution", 1536) or 1536)
        if target_resolution not in {1024, 1536}:
            target_resolution = 1536
        max_num_tokens = int(payload.get("max_num_tokens", 49152) or 49152)
        texture_size = max(256, int(payload.get("texture_size", 4096) or 4096))
        if not image or not Path(image).is_file():
            return "Choose an existing input image.", None, None, {}
        if not output:
            return "Choose an output GLB path.", None, None, {}
        Path(output).parent.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(self.extension_root / "_vendor"), str(self.extension_root), env.get("PYTHONPATH", "")]
        )
        env.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        env.setdefault("PYTHONUTF8", "1")
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("PYTHONIOENCODING", "utf-8")
        try:
            from dependency_manager import configure_windows_triton_environment

            configure_windows_triton_environment(env)
        except Exception:
            pass
        try:
            import certifi
            env.setdefault("SSL_CERT_FILE", certifi.where())
            env.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
        except Exception:
            pass
        command = [
            sys.executable,
            "-u",
            str(self.extension_root / "worker" / "pixal3d_worker.py"),
            "--image",
            image,
            "--output",
            output,
            "--seed",
            str(seed),
            "--model_path",
            model,
            "--device",
            device,
            "--decimation_target",
            str(decimation_target),
            "--target_resolution",
            str(target_resolution),
            "--max_num_tokens",
            str(max_num_tokens),
            "--texture_size",
            str(texture_size),
        ]
        if not bool(payload.get("enable_mps_fallback", True)):
            command.append("--disable_mps_fallback")
        return None, output, command, env

    def _run_header(self, payload: dict, command: list[str], output: str) -> str:
        from dependency_manager import get_runtime_status, resolved_generation_backend

        device = str(payload.get("device", "auto") or "auto")
        status = get_runtime_status(device)
        backend = resolved_generation_backend(device)
        lines = [
            f"[{self._timestamp()}] Pixal3D generation requested",
            f"  Image: {os.path.expanduser(str(payload.get('image', '')).strip())}",
            f"  Output: {output}",
            f"  Model: {str(payload.get('model', '')).strip() or 'TencentARC/Pixal3D'}",
            f"  Seed: {int(payload.get('seed', 42) or 42)}",
            f"  Low-poly face target: {int(payload.get('decimation_target', 1000000) or 1000000)}",
            f"  Target resolution: {int(payload.get('target_resolution', 1536) or 1536)}",
            f"  Max sparse tokens: {int(payload.get('max_num_tokens', 49152) or 49152)}",
            f"  PBR texture size: {int(payload.get('texture_size', 4096) or 4096)}",
            f"  Device request: {device}",
            f"  Resolved backend: {backend.upper()}",
            f"  Import into Blender: {bool(payload.get('import_after_generate', True))}",
            f"  Runtime platform: {status.platform_key}",
            f"  Pywebview ready: {status.webview_ready}",
            f"  Generation ready: {status.generation_ready}",
        ]
        if backend == "metal":
            lines.append("  Metal GPU backend: PyTorch MPS")
            lines.append(f"  Unsupported-op CPU fallback: {bool(payload.get('enable_mps_fallback', True))}")
        elif backend == "cuda":
            lines.append("  CUDA runtime profile: Torch 2.7 / CUDA 12.8 with native Pixal3D wheels")
        if status.missing_generation_modules:
            lines.append("  Missing generation modules: " + ", ".join(status.missing_generation_modules))
        for note in status.unsupported_notes:
            lines.append("  Note: " + note)
        lines.extend(
            [
                "",
                "Command:",
                "  " + shlex.join(command),
                "",
                "Worker output:",
            ]
        )
        return "\n".join(lines)

    def start_generation(self, payload: dict) -> dict:
        validation_error, output, command, env = self._worker_command(payload)
        if validation_error:
            return {"ok": False, "message": validation_error}
        from dependency_manager import get_runtime_status

        device = str(payload.get("device", "auto") or "auto")
        status = get_runtime_status(device)
        if not status.generation_ready:
            lines = [
                "Pixal3D generation runtime is incomplete.",
                "",
                self.status(device),
            ]
            return {"ok": False, "message": "\n".join(lines)}

        with self._job_lock:
            if self._job.get("running"):
                return {"ok": False, "message": "A Pixal3D generation is already running."}
            self._job = {
                "running": True,
                "returncode": None,
                "log": self._run_header(payload, command, output),
                "message": "",
                "output": output,
                "stage": "Starting Pixal3D worker",
                "started_at": time.time(),
                "last_output_at": time.time(),
            }

        import_requested = bool(payload.get("import_after_generate", True))
        thread = threading.Thread(
            target=self._run_generation_job,
            args=(command, env, output, import_requested),
            daemon=True,
        )
        thread.start()
        return {"ok": True, "message": f"Pixal3D generation started.\nOutput: {output}"}

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        seconds = max(0, int(seconds))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes}m {seconds}s"
        if minutes:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    @staticmethod
    def _stage_from_text(text: str) -> str:
        for line in reversed(str(text or "").splitlines()):
            clean = line.strip()
            if clean:
                return clean[:180]
        return ""

    def _append_log(self, text: str, *, worker_output: bool = True, update_stage: bool = True) -> None:
        if not text:
            return
        with self._job_lock:
            current = str(self._job.get("log") or "")
            combined = (current + "\n" + text).strip()
            self._job["log"] = combined
            if worker_output:
                self._job["last_output_at"] = time.time()
            if update_stage:
                stage = self._stage_from_text(text)
                if stage:
                    self._job["stage"] = stage

    def _heartbeat(self, stop_event: threading.Event) -> None:
        while not stop_event.wait(25):
            with self._job_lock:
                if not self._job.get("running"):
                    return
                started_at = self._job.get("started_at") or time.time()
                last_output_at = self._job.get("last_output_at") or started_at
                stage = str(self._job.get("stage") or "working")
            elapsed = self._format_elapsed(time.time() - float(started_at))
            quiet = self._format_elapsed(time.time() - float(last_output_at))
            self._append_log(
                f"[{self._timestamp()}] Still working: elapsed {elapsed}, no new worker output for {quiet}. Current stage: {stage}",
                worker_output=False,
                update_stage=False,
            )

    def _run_generation_job(self, command: list[str], env: dict, output: str, import_requested: bool) -> None:
        stop_event = threading.Event()
        try:
            process = subprocess.Popen(
                command,
                cwd=str(self.extension_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            with self._job_lock:
                self._job["pid"] = process.pid
            self._append_log(f"[{self._timestamp()}] Worker process started: pid={process.pid}")
            heartbeat_thread = threading.Thread(target=self._heartbeat, args=(stop_event,), daemon=True)
            heartbeat_thread.start()
            if process.stdout is not None:
                for line in process.stdout:
                    self._append_log(line.rstrip())
            returncode = process.wait()
            stop_event.set()
            self._append_log(f"[{self._timestamp()}] Worker process exited: code={returncode}")
            if returncode == 0 and Path(output).is_file():
                self._write_last_output(output, import_requested)
                try:
                    size_mb = Path(output).stat().st_size / (1024 * 1024)
                    self._append_log(f"[{self._timestamp()}] Output GLB size: {size_mb:.2f} MB")
                except Exception:
                    pass
                message = f"Finished: {output}"
            elif returncode == 0:
                message = "Pixal3D finished, but the output GLB was not found."
            else:
                message = f"Pixal3D failed with exit code {returncode}."
            with self._job_lock:
                self._job.update({"running": False, "returncode": returncode, "message": message})
        except Exception as error:
            stop_event.set()
            with self._job_lock:
                self._job.update(
                    {
                        "running": False,
                        "returncode": -1,
                        "message": f"Pixal3D worker could not start: {error}",
                        "log": (str(self._job.get("log") or "") + f"\n[{self._timestamp()}] {error}").strip(),
                    }
                )

    def generation_status(self) -> dict:
        with self._job_lock:
            status = dict(self._job)
        if status.get("running") and status.get("started_at"):
            status["elapsed"] = self._format_elapsed(time.time() - float(status["started_at"]))
        return status

    def generate(self, payload: dict) -> str:
        validation_error, output, command, env = self._worker_command(payload)
        if validation_error:
            return validation_error
        header = self._run_header(payload, command, output or "")
        result = subprocess.run(
            command,
            cwd=str(self.extension_root),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode == 0 and output and Path(output).is_file():
            self._write_last_output(output, bool(payload.get("import_after_generate", True)))
        return (header + "\n" + result.stdout + "\n" + result.stderr).strip()


def _raise_window_on_launch(window) -> None:
    if sys.platform not in {"darwin", "win32"}:
        return
    try:
        if sys.platform == "darwin":
            try:
                import AppKit
                import Foundation
                from PyObjCTools import AppHelper
                from webview.platforms import cocoa as cocoa_platform

                def _focus_native_window():
                    try:
                        app = AppKit.NSApplication.sharedApplication()
                        native = cocoa_platform.BrowserView.instances.get(window.uid)
                        if hasattr(app, "setActivationPolicy_"):
                            app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyRegular)
                        if native is None:
                            return
                        try:
                            native.window.setLevel_(AppKit.NSStatusWindowLevel)
                        except Exception:
                            pass
                        try:
                            native.window.orderFrontRegardless()
                        except Exception:
                            pass
                        try:
                            native.window.makeKeyAndOrderFront_(native.window)
                        except Exception:
                            pass
                        try:
                            app.activateIgnoringOtherApps_(Foundation.YES)
                        except Exception:
                            pass
                    except Exception:
                        pass

                def _normalize_window_level():
                    try:
                        native = cocoa_platform.BrowserView.instances.get(window.uid)
                        if native is not None:
                            native.window.setLevel_(AppKit.NSNormalWindowLevel)
                    except Exception:
                        pass

                def _focus_burst():
                    for _index in range(8):
                        try:
                            AppHelper.callAfter(_focus_native_window)
                        except Exception:
                            break
                        time.sleep(0.18)
                    try:
                        AppHelper.callAfter(_normalize_window_level)
                    except Exception:
                        pass

                threading.Thread(target=_focus_burst, daemon=True).start()
                return
            except Exception:
                pass
        window.on_top = True
        time.sleep(0.5)
        window.on_top = False
    except Exception:
        pass


def _bind_shown(window) -> None:
    if platform.system().lower() != "darwin":
        return
    try:
        window.shown += lambda: _raise_window_on_launch(window)
        return
    except Exception:
        pass
    try:
        events = getattr(window, "events", None)
        shown = getattr(events, "shown", None)
        if shown is not None:
            shown += lambda: _raise_window_on_launch(window)
    except Exception:
        pass


def _start_webview(window, storage_path: str) -> None:
    import webview

    def on_loaded(_window=None):
        if platform.system().lower() == "darwin":
            _raise_window_on_launch(window)

    try:
        webview.start(on_loaded, (window,), debug=False, storage_path=storage_path)
    except TypeError:
        try:
            webview.start(on_loaded, (window,), debug=False)
        except TypeError:
            webview.start(debug=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension-root", required=True)
    parser.add_argument("--session-id", default="")
    args = parser.parse_args()
    extension_root = Path(args.extension_root).resolve()
    _bootstrap(extension_root)

    import webview

    profile_dir = extension_root / "wheels" / "webview_profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    session_id = str(args.session_id or "standalone").strip()
    window = webview.create_window(
        "Beyond Pixal3D",
        html=HTML,
        js_api=Pixal3DApi(extension_root, session_id),
        width=860,
        height=700,
    )
    _bind_shown(window)
    _start_webview(window, str(profile_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
