"""Holy Score bridge plugin for WanGP.

Runs inside the normal WanGP process and gives the Holy Score studio a small,
CORS-enabled REST API for YuE2 on 127.0.0.1:7867, plus a cloudflared quick
tunnel so the hosted studio can reach this PC.

* Renders go through WanGP's own web queue (the same path WanGP's in-UI MCP
  server uses), so they show up in the WanGP queue and never fight another
  generation for the GPU.
* Score files (.abc / .mid) are saved next to every song.
* Song <-> ABC conversion uses WanGP's own SheetSage2 serializer (score.py).
* /music/llm forwards chat requests to any OpenAI-compatible endpoint, so API
  keys stay on this PC.

Every request needs the pairing key shown in the Holy Score tab
(header ``X-Holy-Music-Key`` or ``?key=`` for audio elements).
"""

import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

import gradio as gr

from shared.utils.plugins import WAN2GPPlugin

PlugIn_Name = "Holy Score"
PlugIn_Id = "HolyMusicBridge"
VERSION = "0.8.3"

API_HOST = "127.0.0.1"
API_PORT = 7867

_here = os.path.dirname(os.path.abspath(__file__))
_app_dir = os.path.dirname(os.path.dirname(_here))
_outputs = os.path.join(_app_dir, "outputs")
_work = os.path.join(_outputs, "holy-music")
_key_file = os.path.join(_here, "holy-music-key.txt")
_config_file = os.path.join(_here, "holy-music-config.json")
_tunnel_log = os.path.join(_here, "holy-music-tunnel.log")
_TUNNEL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
_CLOUDFLARED = "cloudflared.exe" if os.name == "nt" else "cloudflared"

_state = {"api": "starting", "tunnel": "starting", "url": ""}
_started = False
_session = None
_session_lock = threading.Lock()
_jobs = {}
_jobs_lock = threading.Lock()
# Foley runs PrismAudio directly (outside WanGP's queue), so it and the YuE2 renders take turns on the GPU.
_gpu_gate = threading.Lock()
_MAX_JOBS = 50

DEFAULT_CONFIG = {
    "llm": {
        "backend": "opencode",   # "opencode" (your opencode login) or "openai" (any OpenAI-compatible API)
        "opencode_model": "",    # empty -> your opencode default model; e.g. "nvidia-nim/kimi-k2.6"
        "opencode_agent": "plan",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": "",           # empty -> NVIDIA_API_KEY / OPENAI_API_KEY environment variable
        "model": "deepseek-ai/deepseek-v4-flash",
    }
}


def _log(msg):
    print(f"[holy-music] {msg}", flush=True)


def _pairing_key():
    try:
        with open(_key_file, encoding="ascii") as f:
            key = f.read().strip()
            if key:
                return key
    except OSError:
        pass
    key = secrets.token_urlsafe(18)
    with open(_key_file, "w", encoding="ascii") as f:
        f.write(key)
    return key


def _load_config():
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    try:
        with open(_config_file, encoding="utf-8") as f:
            saved = json.load(f)
        for section, values in saved.items():
            if isinstance(values, dict):
                config.setdefault(section, {}).update(values)
    except (OSError, ValueError):
        pass
    return config


def _save_config(config):
    with open(_config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


# ------------------------------------------------------------- WanGP side ---

def _get_session():
    """A WanGP API session that feeds the live web-UI queue when available."""
    global _session
    with _session_lock:
        if _session is not None:
            return _session
        wgp = sys.modules.get("wgp")
        hybrid = getattr(wgp, "_deepy_hybrid", None)
        if hybrid is not None:
            from shared.api import WanGPSession

            session = WanGPSession(webui_state=hybrid._state, console_output=False, console_isatty=False, webui_append=True)
            session._gradio_webui_context = {"load_queue": hybrid.load_inline_queue}
            _log("renders join the WanGP web queue")
        else:
            from shared.api import init

            session = init(root=_app_dir, console_output=False, console_isatty=False)
            _log("no live web queue found - using a headless session")
        _session = session
        return session


def _inside(base, path):
    base, path = os.path.realpath(base), os.path.realpath(path)
    return path == base or path.startswith(base + os.sep)


def _upload_path(upload_id):
    path = os.path.join(_work, "uploads", *str(upload_id).replace("\\", "/").split("/"))
    if not upload_id or not _inside(os.path.join(_work, "uploads"), path) or not os.path.isfile(path):
        raise ValueError(f"Unknown upload {upload_id!r}")
    return path


def _save_score(abc):
    os.makedirs(os.path.join(_work, "scores"), exist_ok=True)
    path = os.path.join(_work, "scores", f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}.abc")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(abc.strip() + "\n")
    return path


def _render_settings(body):
    """Studio render request -> WanGP YuE2 settings."""
    from holy_music_score import compile_song

    hum = bool(body.get("hum"))
    planning = int(body.get("planning", 0))
    settings = {
        "model_type": "yue2_hum" if hum else "yue2",
        "prompt": str(body.get("lyrics") or "").strip() or "[Instrumental]",
        "alt_prompt": str(body.get("style") or "").strip(),
        "model_mode": planning if not hum else int(body.get("humMode", 0)),
        "num_inference_steps": int(body.get("steps", 32)),
        "guidance_scale": float(body.get("cfg", 1.0)),
        "temperature": float(body.get("temperature", 1.0)),
        "top_k": int(body.get("top_k", 100)),
        "top_p": float(body.get("top_p", 0.95)),
        "duration_seconds": int(body.get("maxSeconds", 180)),
        "seed": int(body.get("seed", -1)),
        "custom_settings": {"save_score": 1},
        "multi_prompts_gen_type": "FG",
        "repeat_generation": 1,
        "prompt_enhancer": "",
        "_api": {"return_side_files": False},  # write .abc/.mid next to the song
    }
    if settings["seed"] < 0:
        settings["seed"] = secrets.randbelow(2**31 - 1)
    abc = str(body.get("abc") or "").strip()
    score_song = None
    if abc:
        from holy_music_score import parse_abc

        score_song = parse_abc(abc)
    elif isinstance(body.get("song"), dict) and body.get("useScore", True):
        score_song = body["song"]
        abc, _ = compile_song(score_song, melody_only=planning == 1)
    source = body.get("sourceAudio")
    # YuE2 (especially the instrumental LoRA) often keeps playing after the written score ends.
    # Playing an exact score: cap the length at the score plus a short tail, then fade that tail.
    fade = None
    explicit_end = body.get("endSeconds")  # the studio's full timeline length (also covers "free" sections YuE2 continues into)
    if (explicit_end or (score_song and not body.get("extend"))) and not source and not hum and body.get("fitToScore", True):
        from holy_music_score import score_seconds

        length = float(explicit_end) if explicit_end else score_seconds(score_song)
        tail = float(body.get("tailSeconds", 4))
        settings["duration_seconds"] = max(5, min(600, int(length + tail + 0.999)))
        fade = {"at": length, "tail": tail}
    if hum:
        settings["audio_prompt_type"] = "A"
        settings["audio_guide"] = _upload_path(source)
    elif source:
        settings["audio_prompt_type"] = "A2" if body.get("extend") else "A"
        settings["audio_guide"] = _upload_path(source)
    elif abc:
        settings["audio_prompt_type"] = "12" if body.get("extend") else "1"
        settings["custom_guide"] = _save_score(abc)
    else:
        settings["audio_prompt_type"] = "0"
    loras = [lora for lora in body.get("loras") or [] if lora.get("name")]
    if loras:
        available = {os.path.basename(str(path)): str(path) for path in _get_session().list_loras("yue2").get("loras", [])}
        missing = [lora["name"] for lora in loras if lora["name"] not in available]
        if missing:
            raise ValueError(f"LoRA not found in loras/yue2: {', '.join(missing)}")
        settings["activated_loras"] = [available[lora["name"]] for lora in loras]
        settings["loras_multipliers"] = " ".join(str(float(lora.get("strength", 1.0))) for lora in loras)
    return settings, fade


def _fade_tail(path, fade):
    """Fade out whatever YuE2 played after the score ended (only if it ran past it)."""
    import numpy as np
    import soundfile as sf

    audio, rate = sf.read(path, always_2d=True)
    start = int(fade["at"] * rate)
    if len(audio) <= start + rate // 2:
        return False  # the song ended on its own
    end = min(len(audio), start + int(fade["tail"] * rate))
    ramp = np.cos(np.linspace(0, np.pi / 2, end - start)) ** 2
    audio = audio[:end]
    audio[start:end] *= ramp[:, None]
    info = sf.info(path)
    sf.write(path, audio, rate, subtype=info.subtype)
    return True


def _watch(record):
    job = record["job"]
    while True:
        event = job.events.get(timeout=0.5)
        if event is None:
            if job.done or job.events.closed:
                break
            continue
        data = event.data
        if event.kind == "progress" and data is not None:
            record["progress"] = {
                "phase": getattr(data, "phase", ""),
                "status": getattr(data, "status", ""),
                "percent": getattr(data, "progress", 0),
                "step": getattr(data, "current_step", None),
                "total": getattr(data, "total_steps", None),
            }
        elif event.kind == "status" and data:
            record["statusText"] = str(data)
        elif event.kind == "stream" and data is not None:
            text = str(getattr(data, "text", "")).strip()
            if text:
                record["log"] = (record["log"] + [text])[-30:]
    result = job.result()
    files = []
    for path in result.generated_files:
        if not path:
            continue
        stem, _ = os.path.splitext(path)
        if record.get("fade"):
            try:
                record["fadedTail"] = _fade_tail(path, record["fade"])
            except Exception as exc:
                _log(f"fade-out skipped: {exc}")
        entry = {"audio": _rel(path)}
        for ext in (".abc", ".mid"):
            if os.path.isfile(stem + ext):
                entry[ext[1:]] = _rel(stem + ext)
        files.append(entry)
    record["files"] = files
    record["errors"] = [error.message for error in result.errors]
    record["status"] = "cancelled" if result.cancelled else "done" if result.success and files else "error"
    record["finishedAt"] = time.time()
    record.pop("job", None)


def _rel(path):
    try:
        return os.path.relpath(os.path.realpath(path), os.path.realpath(_outputs)).replace("\\", "/")
    except ValueError:
        return os.path.basename(path)


def _register(record):
    with _jobs_lock:
        _jobs[record["id"]] = record
        while len(_jobs) > _MAX_JOBS:
            oldest = next(iter(_jobs))
            if _jobs[oldest].get("status") == "running":
                break
            _jobs.pop(oldest)


def _submit(body):
    settings, fade = _render_settings(body)
    session = _get_session()
    job_id = uuid.uuid4().hex[:12]
    record = {"id": job_id, "status": "running", "createdAt": time.time(), "settings": _public_settings(settings),
              "fade": fade, "progress": None, "statusText": "Queued in WanGP...", "log": [], "files": [], "errors": [], "job": None}
    _register(record)

    def start():
        with _gpu_gate:  # wait for a foley run to finish with the GPU
            pass
        if record.get("cancel"):
            record.update(status="cancelled", finishedAt=time.time())
            return
        try:
            record["job"] = session.submit_task(settings)
        except Exception as exc:
            record.update(status="error", errors=[str(exc)], finishedAt=time.time())
            return
        _watch(record)

    threading.Thread(target=start, name=f"holy-music-{job_id}", daemon=True).start()
    return record


def _wangp_busy():
    """True while WanGP's own queue is generating (or has work waiting)."""
    wgp = sys.modules.get("wgp")
    hybrid = getattr(wgp, "_deepy_hybrid", None)
    state = getattr(hybrid, "_state", None) if hybrid is not None else None
    gen = state.get("gen", {}) if isinstance(state, dict) else {}
    return bool(gen.get("in_progress")) or bool(gen.get("queue"))


def _listen_submit(body):
    """What a recording actually plays, written back as a score: YuE2's own transcriber (SheetSage2 + MERT2) listens to it."""
    audio = _output_path(body.get("audio", ""))
    job_id = uuid.uuid4().hex[:12]
    record = {"id": job_id, "kind": "listen", "status": "running", "createdAt": time.time(),
              "settings": {"model": "SheetSage2", "audio": body.get("audio", "")},
              "progress": None, "statusText": "Starting...", "log": [], "files": [], "errors": [], "song": None}
    _register(record)
    threading.Thread(target=_listen_run, args=(record, audio), name=f"holy-music-listen-{job_id}", daemon=True).start()
    return record


def _listen_run(record, audio):
    try:
        while _wangp_busy():
            if record.get("cancel"):
                record.update(status="cancelled", finishedAt=time.time())
                return
            record["statusText"] = "Waiting for WanGP to finish what it's doing..."
            time.sleep(1)
        with _gpu_gate:
            import importlib

            from shared.utils import offload_registry
            from shared.utils import files_locator as fl

            offload_registry.unload_vram()  # whatever WanGP left in VRAM goes back to RAM first
            scoring = importlib.import_module("models.TTS.yue2.sheetsage2.scoring")
            checkpoint = fl.locate_file(os.path.join("sheetsage2", "SheetSage2_MERT2_bf16.safetensors"))

            def progress(step_idx=0, override_num_inference_steps=None, progress_title="Listening", **_):
                total = override_num_inference_steps or 0
                record["progress"] = {"phase": progress_title, "status": progress_title, "percent": int(100 * (step_idx + 1) / total) if total else 0, "step": step_idx + 1, "total": total}
                record["statusText"] = progress_title

            abc, _midi = scoring.score_audio(audio, checkpoint, False, progress, lambda: bool(record.get("cancel")))
        from holy_music_score import parse_abc

        record["song"] = parse_abc(abc)
        record["abc"] = abc
        record.update(status="done", statusText="Done", finishedAt=time.time())
    except Exception as exc:  # report, never crash WanGP
        _log(f"listen failed: {exc}")
        record.update(status="error", errors=[str(exc) or exc.__class__.__name__], finishedAt=time.time())


def _foley_submit(body):
    """Diegetic sound for a stretch of the video, made by PrismAudio from the picture and a description."""
    video = _output_path(body.get("video", ""))
    start = max(0.0, float(body.get("start", 0)))
    end = float(body.get("end", 0))
    if end - start < 1:
        raise ValueError("PrismAudio needs at least one second of video")
    prompt = str(body.get("prompt") or "").strip()
    seed = int(body.get("seed", -1))
    if seed < 0:
        seed = secrets.randbelow(2**31 - 1)
    job_id = uuid.uuid4().hex[:12]
    record = {"id": job_id, "kind": "foley", "status": "running", "createdAt": time.time(),
              "settings": {"model": "PrismAudio", "prompt": prompt, "start": start, "end": end, "seed": seed},
              "progress": None, "statusText": "Starting...", "log": [], "files": [], "errors": []}
    _register(record)
    threading.Thread(target=_foley_run, args=(record, video, start, end, prompt, seed), name=f"holy-music-foley-{job_id}", daemon=True).start()
    return record


def _foley_run(record, video, start, end, prompt, seed):
    import subprocess
    from functools import partial

    import imageio_ffmpeg

    try:
        while _wangp_busy():
            if record.get("cancel"):
                record.update(status="cancelled", finishedAt=time.time())
                return
            record["statusText"] = "Waiting for WanGP to finish what it's doing..."
            time.sleep(1)
        with _gpu_gate:
            folder = os.path.join(_work, "foley")
            os.makedirs(folder, exist_ok=True)
            clip = os.path.join(folder, f"{record['id']}_picture.mp4")
            out = os.path.join(folder, f"{record['id']}.wav")
            record["statusText"] = "Cutting the shot..."
            # an exact cut of the stretch (re-encoded so it starts on the right frame)
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-i", video,
                            "-t", f"{end - start:.3f}", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", clip],
                           check=True, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            wgp = sys.modules.get("wgp")
            from postprocessing import audio_processors as audio_api
            from shared.utils import offload_registry

            offload_registry.unload_vram()  # whatever WanGP left in VRAM goes back to RAM first
            _fix_t5gemma()

            def progress(phase, current=None, total=None):
                percent = int(100 * current / total) if current is not None and total else (record.get("progress") or {}).get("percent", 0)
                record["progress"] = {"phase": phase, "status": phase, "percent": percent, "step": current, "total": total}
                record["statusText"] = phase

            path = audio_api.generate_soundtrack(
                "prismaudio", video_path=clip, prompt=prompt, seed=seed, duration=end - start, output_path=out, verbose_level=0,
                init_pipe=partial(wgp.init_pipe, output_type="audio") if wgp is not None else None,
                profile=(wgp.server_config.get("audio_profile", 4) if wgp is not None else 4),
                abort_callback=lambda: bool(record.get("cancel")), progress_callback=progress)
            try:
                os.remove(clip)
            except OSError:
                pass
        if path is None or not os.path.isfile(path):
            record.update(status="cancelled" if record.get("cancel") else "error", errors=[] if record.get("cancel") else ["PrismAudio made no sound"])
        else:
            record.update(status="done", files=[{"audio": _rel(path)}])
    except Exception as exc:
        _log(f"foley failed: {exc}")
        record.update(status="error", errors=[str(exc)])
    finally:
        record["finishedAt"] = time.time()


def _shots(name):
    """Where the picture cuts, in seconds: peaks of ffmpeg's scene-change score, judged against the video's own motion."""
    import re as _re
    import statistics
    import subprocess

    import imageio_ffmpeg

    video = _output_path(name)
    run = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", video, "-an", "-vf", "scale=320:-2,select='gte(scene,0)',metadata=print",
                          "-f", "null", "-"], capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    times, scores = [], []
    for line in run.stderr.splitlines():
        m = _re.search(r"pts_time:([0-9.]+)", line)
        if m:
            times.append(float(m[1]))
        m = _re.search(r"lavfi\.scene_score=([0-9.]+)", line)
        if m:
            scores.append(float(m[1]))
    n = min(len(times), len(scores))
    if not n:
        return {"cuts": [], "duration": 0}
    times, scores = times[:n], scores[:n]
    floor = max(0.05, statistics.median(scores) * 8)
    cuts = []
    for i in range(1, n - 1):
        if scores[i] >= floor and scores[i] >= scores[i - 1] and scores[i] >= scores[i + 1]:
            if not cuts or times[i] - cuts[-1] >= 1.0:
                cuts.append(round(times[i], 2))
    return {"cuts": cuts, "duration": round(times[-1], 2)}


def _fix_t5gemma():
    """WanGP's PrismAudio reaches the text encoder's layers as t5.encoder.block (the T5 layout), but with the
    transformers version WanGP pins (4.54) the T5Gemma encoder keeps them at t5.layers, so every run fails.
    Give the encoder that path (read-only, nothing else changes)."""
    import types

    try:
        from transformers.models.t5gemma.modeling_t5gemma import T5GemmaEncoder
    except Exception:
        return
    if not hasattr(T5GemmaEncoder, "encoder"):
        T5GemmaEncoder.encoder = property(lambda self: types.SimpleNamespace(block=self.layers))


def _output_path(name):
    path = os.path.join(_outputs, *str(name).replace("\\", "/").split("/"))
    if not name or not _inside(_outputs, path) or not os.path.isfile(path):
        raise ValueError(f"file not found: {name}")
    return path


def _public_settings(settings):
    public = {k: v for k, v in settings.items() if k not in ("_api",)}
    for key in ("custom_guide", "audio_guide"):
        if key in public:
            public[key] = _rel(public[key])
    return public


def _job_view(record):
    return {k: v for k, v in record.items() if k != "job"}


def _find_opencode():
    found = shutil.which("opencode")
    if found:
        return found
    appdata = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Roaming")
    for name in ("opencode.cmd", "opencode.exe", "opencode"):
        candidate = os.path.join(appdata, "npm", name)
        if os.path.isfile(candidate):
            return candidate
    return None


def _llm_opencode(body, config):
    """Ask the user's own opencode (its login, its default model) - no API key needed here.

    Runs the read-only "plan" agent in an empty scratch folder; the whole request goes in an
    attached file so long songs don't hit command-line limits.
    """
    exe = _find_opencode()
    if not exe:
        raise RuntimeError("opencode was not found. Install it (npm i -g opencode-ai) or switch the co-writer to an OpenAI-compatible API in WanGP's Holy Score tab.")
    import tempfile

    lines = []
    for message in body.get("messages") or []:
        role = str(message.get("role", "user")).upper()
        lines.append(f"### {role}\n{message.get('content', '')}\n")
    with tempfile.TemporaryDirectory(prefix="holy-music-llm-") as folder:
        prompt = os.path.join(folder, "request.md")
        with open(prompt, "w", encoding="utf-8") as f:
            f.write("You are answering one request from the Holy Score app. The SYSTEM section defines your role and the exact answer format. "
                    "Reply to the last USER message. Do not use tools, do not create files: answer directly with the requested JSON only.\n\n" + "\n".join(lines))
        # The message goes first: -f takes every following positional as another file.
        command = [exe, "run", "Answer the request in the attached file. Output only the JSON object it asks for.",
                   "--agent", str(config.get("opencode_agent") or "plan")]
        if config.get("opencode_model"):
            command += ["-m", str(config["opencode_model"])]
        command += ["-f", prompt]
        result = subprocess.run(command, cwd=folder, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=600, stdin=subprocess.DEVNULL)
    text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", result.stdout or "").strip()
    if result.returncode != 0 or not text:
        detail = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", (result.stderr or "")).strip()[-600:]
        raise RuntimeError(f"opencode failed ({result.returncode}): {detail or 'no output'}")
    return {"choices": [{"message": {"role": "assistant", "content": text}}], "backend": "opencode"}


def _llm(body):
    config = _load_config()["llm"]
    if config.get("backend", "opencode") == "opencode":
        return _llm_opencode(body, config)
    base = str(body.get("base_url") or config["base_url"]).rstrip("/")
    key = config.get("api_key") or os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    payload = {
        "model": body.get("model") or config["model"],
        "messages": body.get("messages") or [],
        "temperature": float(body.get("temperature", 0.6)),
        "max_tokens": int(body.get("max_tokens", 4096)),
    }
    if body.get("response_format"):
        payload["response_format"] = body["response_format"]
    request = urllib.request.Request(base + "/chat/completions", data=json.dumps(payload).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


# ------------------------------------------------------------- HTTP side ---

def _build_app():
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.cors import CORSMiddleware
    from starlette.responses import FileResponse, JSONResponse
    from starlette.routing import Route

    key = _pairing_key()

    def authorized(request):
        given = request.headers.get("x-holy-music-key") or request.query_params.get("key") or ""
        return secrets.compare_digest(given, key)

    def guard(handler):
        async def wrapped(request):
            if not authorized(request):
                return JSONResponse({"error": "pairing key missing or wrong - copy it from WanGP's Holy Score tab"}, status_code=401)
            try:
                return await handler(request)
            except Exception as exc:  # report, never crash the server
                _log(f"{request.url.path} failed: {exc}")
                return JSONResponse({"error": str(exc) or exc.__class__.__name__}, status_code=400)
        return wrapped

    async def health(request):
        info = {"ok": True, "plugin": VERSION, "wangp": getattr(sys.modules.get("wgp"), "WanGP_version", None)}
        try:
            info["yue2"] = _get_session().get_model_availability("yue2")
        except Exception as exc:
            info["yue2"] = {"error": str(exc)}
        return JSONResponse(info)

    async def loras(request):
        result = _get_session().list_loras("yue2")
        return JSONResponse({"loras": [os.path.basename(str(path)) for path in result.get("loras", [])]})

    async def compile_route(request):
        from holy_music_score import compile_song

        body = await request.json()
        abc, warnings = compile_song(body.get("song") or {}, melody_only=bool(body.get("melodyOnly")))
        return JSONResponse({"abc": abc, "warnings": warnings})

    async def parse_route(request):
        from holy_music_score import parse_abc

        body = await request.json()
        return JSONResponse({"song": parse_abc(str(body.get("abc") or ""))})

    async def render(request):
        body = await request.json()
        record = await _run_sync(_submit, body)
        return JSONResponse(_job_view(record))

    async def job(request):
        record = _jobs.get(request.query_params.get("id", ""))
        if record is None:
            return JSONResponse({"error": "unknown job"}, status_code=404)
        return JSONResponse(_job_view(record))

    async def jobs(request):
        with _jobs_lock:
            return JSONResponse({"jobs": [_job_view(record) for record in reversed(list(_jobs.values()))]})

    async def cancel(request):
        record = _jobs.get(request.query_params.get("id", ""))
        if record is None or record.get("status") != "running":
            return JSONResponse({"error": "unknown or finished job"}, status_code=404)
        record["cancel"] = True
        if record.get("job") is not None:
            record["job"].cancel()
        return JSONResponse({"ok": True})

    async def shots(request):
        return JSONResponse(await _run_sync(_shots, request.query_params.get("video", "")))

    async def listen(request):
        body = await request.json()
        return JSONResponse(_job_view(await _run_sync(_listen_submit, body)))

    async def foley(request):
        body = await request.json()
        return JSONResponse(_job_view(await _run_sync(_foley_submit, body)))

    async def file(request):
        name = request.query_params.get("name", "")
        path = os.path.join(_outputs, *name.replace("\\", "/").split("/"))
        if not name or not _inside(_outputs, path) or not os.path.isfile(path):
            return JSONResponse({"error": "not found"}, status_code=404)
        import mimetypes

        kind = {".abc": "text/vnd.abc; charset=utf-8", ".mid": "audio/midi"}.get(os.path.splitext(path)[1].lower())
        return FileResponse(path, media_type=kind or mimetypes.guess_type(path)[0] or "application/octet-stream")

    async def upload(request):
        name = os.path.basename(request.query_params.get("name", "") or "upload.wav")
        if not re.fullmatch(r"[\w .()-]+\.(wav|mp3|flac|ogg|m4a|aac|webm|mid|midi|abc|mp4|mov|m4v|mkv)", name, re.I):
            return JSONResponse({"error": "unsupported file type"}, status_code=400)
        folder = uuid.uuid4().hex[:12]
        target_dir = os.path.join(_work, "uploads", folder)
        os.makedirs(target_dir, exist_ok=True)
        target = os.path.join(target_dir, name)
        size = 0
        with open(target, "xb") as writer:
            async for chunk in request.stream():
                size += len(chunk)
                if size > 1024 * 1024 * 1024:
                    raise ValueError("upload too large (1 GB max)")
                writer.write(chunk)
        return JSONResponse({"id": f"{folder}/{name}", "size": size})

    async def projects(request):
        folder = os.path.join(_work, "projects")
        os.makedirs(folder, exist_ok=True)
        project_id = request.query_params.get("id", "")
        if project_id and not re.fullmatch(r"[\w-]{1,80}", project_id):
            return JSONResponse({"error": "bad project id"}, status_code=400)
        path = os.path.join(folder, project_id + ".json")
        if request.method == "PUT":
            body = await request.body()
            json.loads(body)  # must be valid JSON
            with open(path + ".tmp", "wb") as f:
                f.write(body)
            os.replace(path + ".tmp", path)
            return JSONResponse({"ok": True, "updatedAt": os.path.getmtime(path)})
        if project_id:
            if not os.path.isfile(path):
                return JSONResponse({"error": "not found"}, status_code=404)
            with open(path, encoding="utf-8") as f:
                return JSONResponse({"project": json.load(f), "updatedAt": os.path.getmtime(path)})
        items = []
        for name in sorted(os.listdir(folder)):
            if name.endswith(".json"):
                full = os.path.join(folder, name)
                items.append({"id": name[:-5], "updatedAt": os.path.getmtime(full)})
        return JSONResponse({"projects": items})

    async def transcribe_route(request):
        """PUT raw audio (WAV) -> notes in song time. Query: bpm, offset, latency, key, snap, tight, grid, transpose."""
        import tempfile

        from holy_music_hum import transcribe

        q = request.query_params
        body = await request.body()
        if len(body) > 50 * 1024 * 1024:
            raise ValueError("recording too long")
        # every hum is kept (outputs/holy-music/hums) so a bad transcription can be looked at and redone
        hums = os.path.join(_work, "hums")
        os.makedirs(hums, exist_ok=True)
        temp = os.path.join(hums, time.strftime("%Y-%m-%d %Hh%Mm%Ss") + ".wav")
        with open(temp, "wb") as handle:
            handle.write(body)
        try:
            result = await _run_sync(lambda: transcribe(
                temp, bpm=float(q.get("bpm", 90)), offset=float(q.get("offset", 0)), latency=float(q.get("latency", 0)),
                key=q.get("key", "C"), snap=q.get("snap", "1") == "1", tight=q.get("tight", "1") == "1",
                grid=float(q.get("grid", 0.25)), transpose=int(float(q.get("transpose", 0))), keyfit=q.get("keyfit", "1") == "1"))
        except Exception:
            raise
        result["file"] = _rel(temp)
        return JSONResponse(result)

    async def stretch(request):
        q = request.query_params
        rate = min(2.0, max(0.5, float(q.get("rate", 1))))
        return JSONResponse({"name": await _run_sync(_stretch, q.get("name", ""), rate)})

    async def concat(request):
        body = await request.json()
        return JSONResponse(await _run_sync(_concat, list(body.get("files") or [])))

    async def compose(request):
        body = await request.json()
        return JSONResponse(await _run_sync(_compose, list(body.get("clips") or []), body.get("duration", 0)))

    async def probe(request):
        return JSONResponse(await _run_sync(lambda: _probe(_output_path(request.query_params.get("name", "")))))

    async def mux(request):
        body = await request.json()
        return JSONResponse({"name": await _run_sync(_mux, body.get("video", ""), body.get("audio", ""), body.get("title", ""), body.get("seconds", 0))})

    async def llm(request):
        body = await request.json()
        return JSONResponse(await _run_sync(_llm, body))

    routes = [
        Route("/music/health", guard(health)),
        Route("/music/loras", guard(loras)),
        Route("/music/compile", guard(compile_route), methods=["POST"]),
        Route("/music/parse", guard(parse_route), methods=["POST"]),
        Route("/music/render", guard(render), methods=["POST"]),
        Route("/music/job", guard(job)),
        Route("/music/jobs", guard(jobs)),
        Route("/music/cancel", guard(cancel), methods=["POST"]),
        Route("/music/file", guard(file)),
        Route("/music/upload", guard(upload), methods=["PUT"]),
        Route("/music/projects", guard(projects), methods=["GET", "PUT"]),
        Route("/music/llm", guard(llm), methods=["POST"]),
        Route("/music/stretch", guard(stretch)),
        Route("/music/mux", guard(mux), methods=["POST"]),
        Route("/music/concat", guard(concat), methods=["POST"]),
        Route("/music/probe", guard(probe)),
        Route("/music/compose", guard(compose), methods=["POST"]),
        Route("/music/foley", guard(foley), methods=["POST"]),
        Route("/music/listen", guard(listen), methods=["POST"]),
        Route("/music/shots", guard(shots)),
        Route("/music/transcribe", guard(transcribe_route), methods=["PUT"]),
    ]
    # Access is gated by the pairing key, so any origin may call (Vercel previews, localhost, ...).
    middleware = [Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "PUT", "OPTIONS"],
                             allow_headers=["Content-Type", "X-Holy-Music-Key"], max_age=600)]
    return Starlette(routes=routes, middleware=middleware)


def _stretch(name, rate):
    """Change a song's speed without changing its pitch (Rubber Band, finer engine). Cached per speed."""
    import subprocess

    import imageio_ffmpeg

    source = os.path.join(_outputs, *name.replace("\\", "/").split("/"))
    if not name or not _inside(_outputs, source) or not os.path.isfile(source):
        raise ValueError("song file not found")
    stem = os.path.splitext(os.path.basename(source))[0]
    out_name = f"holy-music/stretched/{stem}@{rate:.2f}.wav"
    target = os.path.join(_outputs, *out_name.split("/"))
    if not os.path.isfile(target) or os.path.getmtime(target) < os.path.getmtime(source):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        # transients=536870912 is Rubber Band's "finer" (R3) engine: the cleanest it has for orchestral music
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", source,
                        "-af", f"rubberband=tempo={rate:.4f}:transients=536870912:channels=together", target],
                       check=True, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return out_name


def _probe(path):
    """Duration (s), size and whether it has sound, from ffmpeg's own report."""
    import subprocess

    import imageio_ffmpeg

    err = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", path], capture_output=True, text=True,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    duration = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else 0.0
    size = re.search(r"Video:.*?, (\d{2,5})x(\d{2,5})", err)
    return {"duration": duration, "width": int(size[1]) if size else 1920, "height": int(size[2]) if size else 1080,
            "audio": bool(re.search(r"Stream #\S+.*Audio:", err))}


def _concat(files):
    """Several video clips joined into one film, in order: every clip fitted to the first one's frame (letterboxed if needed),
    30 fps, its own sound kept (silence where a clip has none)."""
    import hashlib
    import subprocess

    import imageio_ffmpeg

    paths = [_output_path(f) for f in files]
    if not paths:
        raise ValueError("no clips")
    key = hashlib.sha1("|".join(f"{p}:{os.path.getmtime(p)}" for p in paths).encode()).hexdigest()[:12]
    out_name = f"holy-music/films/film-{key}.mp4"
    target = os.path.join(_outputs, *out_name.split("/"))
    if os.path.isfile(target):
        return {"name": out_name, "duration": _probe(target)["duration"]}
    os.makedirs(os.path.dirname(target), exist_ok=True)
    infos = [_probe(p) for p in paths]
    w, h = infos[0]["width"] // 2 * 2, infos[0]["height"] // 2 * 2
    args = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y"]
    for p in paths:
        args += ["-i", p]
    chains, pairs = [], []
    extra = len(paths)
    for i, info in enumerate(infos):
        chains.append(f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,format=yuv420p[v{i}]")
        if info["audio"]:
            chains.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo[a{i}]")
        else:
            args += ["-f", "lavfi", "-t", f"{max(0.05, info['duration']):.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
            chains.append(f"[{extra}:a]anull[a{i}]")
            extra += 1
        pairs.append(f"[v{i}][a{i}]")
    graph = ";".join(chains) + ";" + "".join(pairs) + f"concat=n={len(paths)}:v=1:a=1[v][a]"
    args += ["-filter_complex", graph, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "17",
             "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", target]
    subprocess.run(args, check=True, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return {"name": out_name, "duration": _probe(target)["duration"]}


def _compose(clips, total):
    """The film from clips placed on a timeline: each clip (its trimmed stretch) where it sits, black and silence in
    between and after, every clip fitted to the first one's frame, 30 fps."""
    import hashlib
    import subprocess

    import imageio_ffmpeg

    items = sorted(({"path": _output_path(c["file"]), "at": max(0.0, float(c.get("at", 0))), "from": max(0.0, float(c.get("from", 0))),
                     "to": float(c.get("to", 0))} for c in clips), key=lambda c: c["at"])
    if not items:
        raise ValueError("no clips")
    total = max(float(total or 0), max(c["at"] + c["to"] - c["from"] for c in items))
    key = hashlib.sha1(("|".join(f'{c["path"]}:{os.path.getmtime(c["path"])}:{c["at"]:.3f}:{c["from"]:.3f}:{c["to"]:.3f}' for c in items) + f"|{total:.3f}").encode()).hexdigest()[:12]
    out_name = f"holy-music/films/film-{key}.mp4"
    target = os.path.join(_outputs, *out_name.split("/"))
    if os.path.isfile(target):
        return {"name": out_name, "duration": _probe(target)["duration"]}
    os.makedirs(os.path.dirname(target), exist_ok=True)
    first = _probe(items[0]["path"])
    w, h = first["width"] // 2 * 2, first["height"] // 2 * 2
    args = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y"]
    chains, pairs, n = [], [], 0

    def gap(seconds):
        nonlocal n
        args.extend(["-f", "lavfi", "-t", f"{seconds:.3f}", "-i", f"color=c=black:s={w}x{h}:r=30",
                     "-f", "lavfi", "-t", f"{seconds:.3f}", "-i", "anullsrc=r=48000:cl=stereo"])
        chains.append(f"[{n}:v]setsar=1,format=yuv420p[v{len(pairs)}]")
        chains.append(f"[{n + 1}:a]anull[a{len(pairs)}]")
        pairs.append(f"[v{len(pairs)}][a{len(pairs)}]")
        n += 2

    t = 0.0
    for c in items:
        length = max(0.05, c["to"] - c["from"])
        start = max(c["at"], t)  # clips never overlap: a later one starts after the one before it
        if start - t > 0.02:
            gap(start - t)
        info = _probe(c["path"])
        args.extend(["-ss", f'{c["from"]:.3f}', "-t", f"{length:.3f}", "-i", c["path"]])
        i = len(pairs)
        chains.append(f"[{n}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,format=yuv420p[v{i}]")
        if info["audio"]:
            chains.append(f"[{n}:a]aresample=48000,aformat=channel_layouts=stereo,apad,atrim=0:{length:.3f}[a{i}]")
            n += 1
        else:
            n += 1
            args.extend(["-f", "lavfi", "-t", f"{length:.3f}", "-i", "anullsrc=r=48000:cl=stereo"])
            chains.append(f"[{n}:a]anull[a{i}]")
            n += 1
        pairs.append(f"[v{i}][a{i}]")
        t = start + length
    if total - t > 0.02:
        gap(total - t)
    graph = ";".join(chains) + ";" + "".join(pairs) + f"concat=n={len(pairs)}:v=1:a=1[v][a]"
    args += ["-filter_complex", graph, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
             "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", target]
    subprocess.run(args, check=True, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return {"name": out_name, "duration": _probe(target)["duration"]}


def _mux(video, audio, title, seconds):
    """The video with the studio's mix as its sound (picture copied untouched, length = the video's)."""
    import subprocess
    import time

    import imageio_ffmpeg

    paths = []
    for name in (video, audio):
        path = os.path.join(_outputs, *str(name).replace("\\", "/").split("/"))
        if not name or not _inside(_outputs, path) or not os.path.isfile(path):
            raise ValueError(f"file not found: {name}")
        paths.append(path)
    safe = re.sub(r"[^\w .()-]+", "_", str(title or "song")).strip() or "song"
    out_name = f"holy-music/exports/{safe} {time.strftime('%Y-%m-%d %Hh%Mm%Ss')}.mp4"
    target = os.path.join(_outputs, *out_name.split("/"))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", paths[0], "-i", paths[1],
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-af", "apad", "-c:a", "aac", "-b:a", "320k", "-t", f"{float(seconds):.3f}",
                    "-movflags", "+faststart", target],
                   check=True, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return out_name


async def _run_sync(fn, *args):
    from starlette.concurrency import run_in_threadpool

    return await run_in_threadpool(fn, *args)


def _start():
    global _started
    if _started:
        return
    _started = True
    if _here not in sys.path:
        sys.path.insert(0, _here)
    os.makedirs(_work, exist_ok=True)

    def serve():
        try:
            import uvicorn

            _state["api"] = "ok"
            _log(f"API listening on {API_HOST} port {API_PORT}")  # no "http://" here: Pinokio grabs the first http URL in the log as the WebUI address
            uvicorn.run(_build_app(), host=API_HOST, port=API_PORT, log_level="warning")
        except Exception as exc:
            _state["api"] = f"error: {exc}"
            _log(f"API failed: {exc}")

    threading.Thread(target=serve, name="holy-music-api", daemon=True).start()
    threading.Thread(target=_tunnel, name="holy-music-tunnel", daemon=True).start()


def _find_cloudflared():
    folder = os.path.abspath(_app_dir)
    candidates = [os.path.join(_app_dir, _CLOUDFLARED)]
    while True:
        candidates.append(os.path.join(folder, "bin", _CLOUDFLARED))
        parent = os.path.dirname(folder)
        if parent == folder:
            break
        folder = parent
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return shutil.which("cloudflared")


def _tunnel():
    exe = _find_cloudflared()
    if not exe:
        _state["tunnel"] = "cloudflared not found (enable the Holy Script plugin once, or install cloudflared) - local only"
        return
    try:
        os.remove(_tunnel_log)
    except OSError:
        pass
    try:
        subprocess.Popen([exe, "tunnel", "--url", f"http://{API_HOST}:{API_PORT}", "--logfile", _tunnel_log],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    except Exception as exc:
        _state["tunnel"] = f"error: {exc}"
        return
    for _ in range(90):
        try:
            with open(_tunnel_log, encoding="utf-8", errors="replace") as f:
                match = _TUNNEL_RE.search(f.read())
            if match:
                _state["tunnel"], _state["url"] = "ok", match.group(0)
                _log(f"tunnel up: {match.group(0)}")
                return
        except OSError:
            pass
        time.sleep(2)
    _state["tunnel"] = "error: no tunnel URL after 3 minutes (see holy-music-tunnel.log)"


def _port_open():
    try:
        socket.create_connection((API_HOST, API_PORT), timeout=1).close()
        return True
    except OSError:
        return False


def _status():
    api = f"OK - http://{API_HOST}:{API_PORT}" if _port_open() else _state["api"]
    tunnel = "RUNNING" if _state["tunnel"] == "ok" else _state["tunnel"]
    return f"API: {api}\nTunnel: {tunnel}", _state["url"] or f"http://{API_HOST}:{API_PORT}", _pairing_key()


class HolyMusicPlugin(WAN2GPPlugin):
    def setup_ui(self):
        _start()
        self.add_tab(tab_id=PlugIn_Id, label=PlugIn_Name, component_constructor=self.create_tab)

    def on_tab_select(self, state):
        return _status()

    def on_tab_deselect(self, state):
        pass

    def on_model_change(self, state, model_type):
        pass

    def create_tab(self, api_session):
        status_text, url, key = _status()
        config = _load_config()["llm"]
        with gr.Column():
            gr.Markdown("### Holy Score bridge\nPaste the address and pairing key into the Holy Score studio. "
                        "Renders appear in WanGP's queue like any other job.")
            status = gr.Text(label="Status", value=status_text, lines=2, interactive=False)
            url_box = gr.Text(label="Address", value=url, interactive=False, show_copy_button=True)
            key_box = gr.Text(label="Pairing key", value=key, interactive=False, show_copy_button=True)
            refresh = gr.Button("Refresh status")
            refresh.click(_status, outputs=[status, url_box, key_box])
            with gr.Accordion("Co-writer LLM", open=False):
                backend = gr.Radio(label="Backend", choices=[("opencode (your opencode login, no key)", "opencode"), ("OpenAI-compatible API", "openai")],
                                   value=config.get("backend", "opencode"))
                oc_model = gr.Text(label="opencode model (blank = your opencode default), e.g. nvidia-nim/kimi-k2.6", value=config.get("opencode_model", ""))
                base = gr.Text(label="API base URL", value=config["base_url"])
                model = gr.Text(label="API model", value=config["model"])
                api_key = gr.Text(label="API key (blank = NVIDIA_API_KEY / OPENAI_API_KEY env var)", value="", type="password",
                                  placeholder="saved" if config.get("api_key") else "")
                saved = gr.Markdown("")

                def save(backend_value, oc_model_value, base_url, model_name, key_value):
                    current = _load_config()
                    current["llm"].update({"backend": backend_value, "opencode_model": oc_model_value.strip(),
                                           "base_url": base_url.strip(), "model": model_name.strip()})
                    if key_value.strip():
                        current["llm"]["api_key"] = key_value.strip()
                    _save_config(current)
                    return "Saved.", ""

                gr.Button("Save LLM settings").click(save, inputs=[backend, oc_model, base, model, api_key], outputs=[saved, api_key])
        self.on_tab_outputs = [status, url_box, key_box]
