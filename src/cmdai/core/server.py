"""Background Ollama & OpenAI-compatible HTTP Server for CMDAI."""

from __future__ import annotations

import datetime
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any, Dict, List, Optional, Tuple

from .engine import SimpleGGUFLoader
from .hf_hub import download_gguf_model, parse_quant_tag, resolve_pull_target
from .providers import stream_chat_completion
from .settings import get_settings


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _utc_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class BackgroundHTTPServer:
    """Thread-safe local HTTP server exposing Ollama and OpenAI endpoints."""

    def __init__(self, port: int = 11434, host: str = "127.0.0.1", app_ref: Any = None):
        self.host = host or "127.0.0.1"
        self.port = 11434 if port is None else int(port)
        self.app_ref = app_ref
        self.server: Optional[_ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.started_at: float = 0.0
        self.requests_served: int = 0
        self.last_request_at: float = 0.0
        self.last_error: str = ""
        self._infer_lock = threading.RLock()
        self._standalone_loader: Optional[SimpleGGUFLoader] = None

    @property
    def running(self) -> bool:
        return bool(self.server is not None and self.thread is not None and self.thread.is_alive())

    def get_loader(self) -> SimpleGGUFLoader:
        if self.app_ref is not None and getattr(self.app_ref, "gguf_loader", None) is not None:
            return self.app_ref.gguf_loader
        if self._standalone_loader is None:
            self._standalone_loader = SimpleGGUFLoader()
        return self._standalone_loader

    def get_status(self) -> Dict[str, Any]:
        loader = self.get_loader()
        loaded_path = getattr(loader, "current_model_path", None)
        loaded_name = os.path.basename(loaded_path) if loaded_path else ""
        available = loader.list_models()
        return {
            "running": self.running,
            "host": self.host,
            "port": self.port,
            "base_url": f"http://{self.host}:{self.port}",
            "openai_url": f"http://{self.host}:{self.port}/v1",
            "loaded_model": loaded_name if getattr(loader, "llm", None) else "",
            "n_ctx": getattr(loader, "current_n_ctx", 0) or 0,
            "loader": getattr(loader, "current_loader", "") or "",
            "available_models": len(available),
            "requests_served": self.requests_served,
            "started_at": self.started_at,
            "last_error": self.last_error,
        }

    def start(self) -> bool:
        if self.running:
            return True

        server_self = self

        class CMDAIHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send_cors_headers(self) -> None:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

            def _send_json(self, payload: Any, status: int = 200) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_text(self, text: str, status: int = 200) -> None:
                body = text.encode("utf-8")
                self.send_response(status)
                self._send_cors_headers()
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _start_stream_headers(self, content_type: str) -> None:
                self.send_response(200)
                self._send_cors_headers()
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True

            def _read_json_body(self) -> Dict[str, Any]:
                try:
                    length = int(self.headers.get("Content-Length", 0) or 0)
                except ValueError:
                    length = 0
                if length <= 0:
                    return {}
                raw = self.rfile.read(length)
                if not raw:
                    return {}
                data = json.loads(raw.decode("utf-8"))
                return data if isinstance(data, dict) else {}

            def _touch(self) -> None:
                server_self.requests_served += 1
                server_self.last_request_at = time.time()

            def do_OPTIONS(self) -> None:
                self.send_response(204)
                self._send_cors_headers()
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self) -> None:
                self._touch()
                path = self.path.split("?")[0].rstrip("/") or "/"

                if path == "/":
                    self._send_text("CMDAI server is running")
                    return

                if path in ("/health", "/api/version"):
                    self._send_json(
                        {
                            "status": "ok",
                            "app": "CMDAI",
                            "version": "1.0.0",
                        }
                    )
                    return

                if path == "/api/tags":
                    loader = server_self.get_loader()
                    models_out = []
                    for m in loader.list_models():
                        fpath = m["path"]
                        try:
                            stat = os.stat(fpath)
                            mtime = datetime.datetime.fromtimestamp(
                                stat.st_mtime, datetime.timezone.utc
                            ).strftime("%Y-%m-%dT%H:%M:%SZ")
                            size_bytes = stat.st_size
                        except OSError:
                            mtime = _utc_iso()
                            size_bytes = int(m.get("size_mb", 0) * 1024 * 1024)
                        quant = parse_quant_tag(m["id"])
                        models_out.append(
                            {
                                "name": m["id"],
                                "model": m["id"],
                                "modified_at": mtime,
                                "size": size_bytes,
                                "digest": m["id"],
                                "details": {
                                    "format": "gguf",
                                    "family": "gguf",
                                    "quantization_level": quant,
                                },
                            }
                        )
                    self._send_json({"models": models_out})
                    return

                if path == "/api/ps":
                    loader = server_self.get_loader()
                    running_models = []
                    if getattr(loader, "llm", None) and getattr(loader, "current_model_path", None):
                        fpath = loader.current_model_path
                        fname = os.path.basename(fpath)
                        try:
                            size_b = os.path.getsize(fpath)
                        except OSError:
                            size_b = 0
                        running_models.append(
                            {
                                "name": fname,
                                "model": fname,
                                "size": size_b,
                                "digest": fname,
                                "details": {
                                    "format": "gguf",
                                    "quantization_level": parse_quant_tag(fname),
                                },
                                "context_length": getattr(loader, "current_n_ctx", 0) or 0,
                                "loader": getattr(loader, "current_loader", "") or "",
                            }
                        )
                    self._send_json({"models": running_models})
                    return

                if path == "/v1/models":
                    loader = server_self.get_loader()
                    data = []
                    now_ts = int(time.time())
                    for m in loader.list_models():
                        data.append(
                            {
                                "id": m["id"],
                                "object": "model",
                                "created": now_ts,
                                "owned_by": "cmdai-local",
                            }
                        )
                    self._send_json({"object": "list", "data": data})
                    return

                self._send_json({"error": f"Unknown endpoint: {path}"}, status=404)

            def do_DELETE(self) -> None:
                self._touch()
                path = self.path.split("?")[0].rstrip("/")
                if path == "/api/delete":
                    self._handle_delete()
                    return
                self._send_json({"error": f"Unknown DELETE endpoint: {path}"}, status=404)

            def do_POST(self) -> None:
                self._touch()
                path = self.path.split("?")[0].rstrip("/")

                if path == "/api/show":
                    self._handle_show()
                elif path == "/api/load":
                    self._handle_load()
                elif path == "/api/unload":
                    loader = server_self.get_loader()
                    with server_self._infer_lock:
                        loader.unload()
                    self._send_json({"ok": True})
                elif path == "/api/pull":
                    self._handle_pull()
                elif path == "/api/delete":
                    self._handle_delete()
                elif path == "/api/chat":
                    self._handle_ollama_chat()
                elif path == "/api/generate":
                    self._handle_ollama_generate()
                elif path == "/v1/chat/completions":
                    self._handle_openai_chat()
                else:
                    self._send_json({"error": f"Unknown endpoint: {path}"}, status=404)

            def _resolve_local_model_name(self, requested: str) -> str:
                loader = server_self.get_loader()
                models = loader.list_models()
                req = (requested or "").strip()
                if req:
                    for m in models:
                        if m["id"] == req or m["name"] == req:
                            return m["id"]
                    low = req.lower()
                    for m in models:
                        if low in m["id"].lower():
                            return m["id"]
                if getattr(loader, "current_model_path", None):
                    return os.path.basename(loader.current_model_path)
                if server_self.app_ref is not None and getattr(server_self.app_ref, "current_model_id", None):
                    cid = str(server_self.app_ref.current_model_id)
                    for m in models:
                        if m["id"] == cid:
                            return m["id"]
                return models[0]["id"] if models else req

            def _run_completion(
                self,
                requested_model: str,
                messages: List[Dict[str, Any]],
                temperature: float = 0.6,
                max_tokens: int = 2048,
                on_token: Any = None,
                on_thinking: Any = None,
            ) -> Tuple[str, str, str, Dict[str, Any]]:
                usage: Dict[str, Any] = {}

                def _on_usage(u: Dict[str, Any]) -> None:
                    if isinstance(u, dict):
                        usage.update(u)

                # Explicit cloud syntax `<provider>:<model>` or fallback when no local GGUF matches.
                if ":" in (requested_model or "") and not requested_model.lower().endswith(".gguf"):
                    prov_id, _, cloud_model = requested_model.partition(":")
                    if prov_id and cloud_model:
                        settings = get_settings()
                        api_key = settings.get_api_key(prov_id)
                        text, think = stream_chat_completion(
                            provider_id=prov_id,
                            model_id=cloud_model,
                            messages=messages,
                            api_key=api_key,
                            generation_params={"temperature": temperature, "max_tokens": max_tokens},
                            on_token=on_token,
                            on_thinking=on_thinking,
                            on_usage=_on_usage,
                        )
                        return requested_model, text, think, usage

                loader = server_self.get_loader()
                resolved = self._resolve_local_model_name(requested_model)
                if not resolved:
                    raise RuntimeError("No local GGUF model found in CMDAI models/. Use `cmdai pull` or `/installmodels`.")

                with server_self._infer_lock:
                    text, think = loader.stream_chat(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        on_token=on_token,
                        on_thinking=on_thinking,
                        model_filename=resolved,
                        on_usage=_on_usage,
                    )
                return resolved, text, think, usage

            def _handle_show(self) -> None:
                try:
                    body = self._read_json_body()
                    name = str(body.get("name") or body.get("model") or "").strip()
                    loader = server_self.get_loader()
                    resolved = self._resolve_local_model_name(name)
                    match = next((m for m in loader.list_models() if m["id"] == resolved), None)
                    if not match:
                        self._send_json({"error": f"Model not found: {name}"}, status=404)
                        return
                    self._send_json(
                        {
                            "modelfile": f"FROM {match['path']}",
                            "parameters": "",
                            "template": "",
                            "details": {
                                "format": "gguf",
                                "quantization_level": parse_quant_tag(match["id"]),
                            },
                            "size_mb": match.get("size_mb", 0),
                        }
                    )
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=400)

            def _handle_load(self) -> None:
                try:
                    body = self._read_json_body()
                    model_req = str(body.get("model") or body.get("name") or "").strip()
                    n_ctx = int(body.get("n_ctx") or 0)
                    n_gpu = body.get("n_gpu_layers")
                    loader_type = body.get("loader")
                    loader = server_self.get_loader()
                    resolved = self._resolve_local_model_name(model_req)
                    if not resolved:
                        self._send_json({"error": "No GGUF model available to load."}, status=404)
                        return
                    with server_self._infer_lock:
                        ok = loader.load_model(
                            resolved,
                            n_ctx=n_ctx,
                            n_gpu_layers=n_gpu,
                            loader_type=loader_type,
                        )
                    if not ok:
                        self._send_json({"error": f"Failed to load model {resolved}"}, status=500)
                        return
                    self._send_json(
                        {
                            "ok": True,
                            "model": resolved,
                            "n_ctx": loader.current_n_ctx,
                            "loader": loader.current_loader,
                        }
                    )
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=400)

            def _handle_delete(self) -> None:
                try:
                    body = self._read_json_body()
                    name = str(body.get("name") or body.get("model") or "").strip()
                    loader = server_self.get_loader()
                    resolved = self._resolve_local_model_name(name)
                    match = next((m for m in loader.list_models() if m["id"] == resolved), None)
                    if not match:
                        self._send_json({"error": f"Model not found: {name}"}, status=404)
                        return
                    if getattr(loader, "current_model_path", None) and os.path.abspath(loader.current_model_path) == os.path.abspath(match["path"]):
                        with server_self._infer_lock:
                            loader.unload()
                    os.remove(match["path"])
                    self._send_json({"ok": True, "deleted": match["id"]})
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=400)

            def _handle_pull(self) -> None:
                stream_started = False
                try:
                    body = self._read_json_body()
                    target = str(body.get("name") or body.get("model") or "").strip()
                    quant = str(body.get("quant") or "").strip()
                    stream = body.get("stream", True)
                    repo_id, file_entry = resolve_pull_target(target, quant_hint=quant)

                    if not stream:
                        dest_path = download_gguf_model(repo_id, file_entry["rfilename"])
                        self._send_json(
                            {
                                "status": "success",
                                "repo_id": repo_id,
                                "file": os.path.basename(dest_path),
                                "path": dest_path,
                            }
                        )
                        return

                    self._start_stream_headers("application/x-ndjson; charset=utf-8")
                    stream_started = True

                    def _emit_ndjson(obj: Dict[str, Any]) -> None:
                        line = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
                        self.wfile.write(line)
                        self.wfile.flush()

                    _emit_ndjson({"status": f"pulling {repo_id}/{file_entry['filename']}"})

                    def _on_prog(info: Dict[str, Any]) -> None:
                        _emit_ndjson(
                            {
                                "status": "downloading",
                                "digest": info.get("filename", ""),
                                "total": int(info.get("total_bytes") or 0),
                                "completed": int(info.get("downloaded_bytes") or 0),
                            }
                        )

                    download_gguf_model(repo_id, file_entry["rfilename"], on_progress=_on_prog)
                    _emit_ndjson({"status": "success"})
                except Exception as exc:
                    if stream_started:
                        try:
                            self.wfile.write((json.dumps({"error": str(exc)}) + "\n").encode("utf-8"))
                            self.wfile.flush()
                        except Exception:
                            pass
                    else:
                        self._send_json({"error": str(exc)}, status=500)

            def _handle_ollama_chat(self) -> None:
                stream_started = False
                try:
                    body = self._read_json_body()
                    model_req = str(body.get("model") or "").strip()
                    messages = body.get("messages") or []
                    stream = bool(body.get("stream", True))
                    opts = body.get("options") if isinstance(body.get("options"), dict) else {}
                    temp = float(opts.get("temperature", body.get("temperature", 0.6)))
                    max_tok = int(opts.get("num_predict", body.get("max_tokens", 2048)) or 2048)
                    keep_alive = body.get("keep_alive")

                    started_ns = time.monotonic_ns()
                    if not stream:
                        used_model, text, think, usage = self._run_completion(
                            model_req, messages, temperature=temp, max_tokens=max_tok
                        )
                        if keep_alive == 0:
                            server_self.get_loader().unload()
                        msg_obj: Dict[str, Any] = {"role": "assistant", "content": text}
                        if think:
                            msg_obj["thinking"] = think
                        self._send_json(
                            {
                                "model": used_model,
                                "created_at": _utc_iso(),
                                "message": msg_obj,
                                "done": True,
                                "done_reason": "stop",
                                "total_duration": max(1, time.monotonic_ns() - started_ns),
                                "prompt_eval_count": int(usage.get("prompt_tokens") or 0),
                                "eval_count": int(usage.get("completion_tokens") or 0),
                            }
                        )
                        return

                    out_model = self._resolve_local_model_name(model_req)
                    if not out_model and ":" not in (model_req or ""):
                        raise RuntimeError("No local GGUF model found in CMDAI models/.")
                    out_model = out_model or model_req or "cmdai-local"

                    self._start_stream_headers("application/x-ndjson; charset=utf-8")
                    stream_started = True

                    def _emit(obj: Dict[str, Any]) -> None:
                        self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
                        self.wfile.flush()

                    def _on_tok(tok: str) -> None:
                        _emit(
                            {
                                "model": out_model,
                                "created_at": _utc_iso(),
                                "message": {"role": "assistant", "content": tok},
                                "done": False,
                            }
                        )

                    used_model, _, _, usage = self._run_completion(
                        model_req,
                        messages,
                        temperature=temp,
                        max_tokens=max_tok,
                        on_token=_on_tok,
                    )
                    if keep_alive == 0:
                        server_self.get_loader().unload()
                    _emit(
                        {
                            "model": used_model,
                            "created_at": _utc_iso(),
                            "message": {"role": "assistant", "content": ""},
                            "done": True,
                            "done_reason": "stop",
                            "total_duration": max(1, time.monotonic_ns() - started_ns),
                            "prompt_eval_count": int(usage.get("prompt_tokens") or 0),
                            "eval_count": int(usage.get("completion_tokens") or 0),
                        }
                    )
                except Exception as exc:
                    if stream_started:
                        try:
                            self.wfile.write((json.dumps({"error": str(exc), "done": True}) + "\n").encode("utf-8"))
                            self.wfile.flush()
                        except Exception:
                            pass
                    else:
                        self._send_json({"error": str(exc)}, status=500)

            def _handle_ollama_generate(self) -> None:
                stream_started = False
                try:
                    body = self._read_json_body()
                    model_req = str(body.get("model") or "").strip()
                    prompt = str(body.get("prompt") or "")
                    system_prompt = str(body.get("system") or "").strip()
                    stream = bool(body.get("stream", True))
                    opts = body.get("options") if isinstance(body.get("options"), dict) else {}
                    temp = float(opts.get("temperature", body.get("temperature", 0.6)))
                    max_tok = int(opts.get("num_predict", body.get("max_tokens", 2048)) or 2048)

                    messages: List[Dict[str, str]] = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    messages.append({"role": "user", "content": prompt})

                    started_ns = time.monotonic_ns()
                    if not stream:
                        used_model, text, _, usage = self._run_completion(
                            model_req, messages, temperature=temp, max_tokens=max_tok
                        )
                        self._send_json(
                            {
                                "model": used_model,
                                "created_at": _utc_iso(),
                                "response": text,
                                "done": True,
                                "done_reason": "stop",
                                "total_duration": max(1, time.monotonic_ns() - started_ns),
                                "prompt_eval_count": int(usage.get("prompt_tokens") or 0),
                                "eval_count": int(usage.get("completion_tokens") or 0),
                            }
                        )
                        return

                    out_model = self._resolve_local_model_name(model_req)
                    if not out_model and ":" not in (model_req or ""):
                        raise RuntimeError("No local GGUF model found in CMDAI models/.")
                    out_model = out_model or model_req or "cmdai-local"

                    self._start_stream_headers("application/x-ndjson; charset=utf-8")
                    stream_started = True

                    def _emit(obj: Dict[str, Any]) -> None:
                        self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
                        self.wfile.flush()

                    def _on_tok(tok: str) -> None:
                        _emit(
                            {
                                "model": out_model,
                                "created_at": _utc_iso(),
                                "response": tok,
                                "done": False,
                            }
                        )

                    used_model, _, _, usage = self._run_completion(
                        model_req, messages, temperature=temp, max_tokens=max_tok, on_token=_on_tok
                    )
                    _emit(
                        {
                            "model": used_model,
                            "created_at": _utc_iso(),
                            "response": "",
                            "done": True,
                            "done_reason": "stop",
                            "total_duration": max(1, time.monotonic_ns() - started_ns),
                            "prompt_eval_count": int(usage.get("prompt_tokens") or 0),
                            "eval_count": int(usage.get("completion_tokens") or 0),
                        }
                    )
                except Exception as exc:
                    if stream_started:
                        try:
                            self.wfile.write((json.dumps({"error": str(exc), "done": True}) + "\n").encode("utf-8"))
                            self.wfile.flush()
                        except Exception:
                            pass
                    else:
                        self._send_json({"error": str(exc)}, status=500)

            def _handle_openai_chat(self) -> None:
                stream_started = False
                try:
                    body = self._read_json_body()
                    model_req = str(body.get("model") or "").strip()
                    messages = body.get("messages") or []
                    stream = bool(body.get("stream", False))
                    temp = float(body.get("temperature", 0.6))
                    max_tok = int(body.get("max_tokens", 2048) or 2048)
                    req_id = f"chatcmpl-{uuid.uuid4().hex[:16]}"
                    created = int(time.time())

                    if not stream:
                        used_model, text, _, usage = self._run_completion(
                            model_req, messages, temperature=temp, max_tokens=max_tok
                        )
                        prompt_t = int(usage.get("prompt_tokens") or 0)
                        comp_t = int(usage.get("completion_tokens") or 0)
                        self._send_json(
                            {
                                "id": req_id,
                                "object": "chat.completion",
                                "created": created,
                                "model": used_model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "message": {"role": "assistant", "content": text},
                                        "finish_reason": "stop",
                                    }
                                ],
                                "usage": {
                                    "prompt_tokens": prompt_t,
                                    "completion_tokens": comp_t,
                                    "total_tokens": prompt_t + comp_t,
                                },
                            }
                        )
                        return

                    out_model = self._resolve_local_model_name(model_req)
                    if not out_model and ":" not in (model_req or ""):
                        raise RuntimeError("No local GGUF model found in CMDAI models/.")
                    out_model = out_model or model_req or "cmdai-local"

                    self._start_stream_headers("text/event-stream; charset=utf-8")
                    stream_started = True

                    def _emit_sse(payload: Dict[str, Any]) -> None:
                        chunk = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
                        self.wfile.write(chunk)
                        self.wfile.flush()

                    _emit_sse(
                        {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": out_model,
                            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                        }
                    )

                    def _on_tok(tok: str) -> None:
                        _emit_sse(
                            {
                                "id": req_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": out_model,
                                "choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}],
                            }
                        )

                    used_model, _, _, _ = self._run_completion(
                        model_req, messages, temperature=temp, max_tokens=max_tok, on_token=_on_tok
                    )
                    _emit_sse(
                        {
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": used_model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        }
                    )
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except Exception as exc:
                    if stream_started:
                        try:
                            self.wfile.write(f"data: {json.dumps({'error': {'message': str(exc)}})}\n\n".encode("utf-8"))
                            self.wfile.write(b"data: [DONE]\n\n")
                            self.wfile.flush()
                        except Exception:
                            pass
                    else:
                        self._send_json({"error": {"message": str(exc)}}, status=500)

        try:
            self.server = _ThreadingHTTPServer((self.host, self.port), CMDAIHandler)
            # Capture OS-assigned port when port=0 is passed in tests.
            self.port = int(self.server.server_address[1])
            self.started_at = time.time()
            self.last_error = ""
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
            return True
        except Exception as exc:
            self.last_error = str(exc)
            self.server = None
            self.thread = None
            return False

    def stop(self) -> None:
        if self.server is not None:
            try:
                self.server.shutdown()
            except Exception:
                pass
            try:
                self.server.server_close()
            except Exception:
                pass
            self.server = None
        if self.thread is not None:
            try:
                self.thread.join(timeout=2.0)
            except Exception:
                pass
            self.thread = None

    def restart(self, port: Optional[int] = None, host: Optional[str] = None) -> bool:
        self.stop()
        if port is not None:
            self.port = int(port)
        if host is not None:
            self.host = host
        return self.start()
