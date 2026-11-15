"""Controller bridge between CMDAI and standalone CMDAI-CODE (e:\\CMDAI CODE)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

from .model_paths import app_root
from .settings import get_settings


def resolve_cmdai_code_script() -> Optional[str]:
    """Locate the external CMDAI-CODE `cmdai.py` entrypoint without self-matching CMDAI."""
    self_script = os.path.normcase(os.path.abspath(os.path.join(app_root(), "cmdai.py")))
    settings = get_settings()

    candidates: List[str] = []
    env_path = os.environ.get("CMDAI_CODE_PATH", "").strip()
    if env_path:
        candidates.append(env_path)

    cfg_path = str(settings.config.get("cmdai_code_path") or "").strip()
    if cfg_path:
        candidates.append(cfg_path)

    env_root = os.environ.get("CMDAI_CODE_ROOT", "").strip()
    if env_root:
        candidates.append(os.path.join(env_root, "cmdai.py"))

    candidates.extend(
        [
            r"E:\CMDAI CODE\cmdai.py",
            r"D:\CMDAI CODE\cmdai.py",
            r"E:\CMDAI-CODE\cmdai.py",
            os.path.join(os.path.expanduser("~"), "CMDAI-CODE", "cmdai.py"),
            os.path.join(os.path.expanduser("~"), "CMDAI CODE", "cmdai.py"),
        ]
    )

    for cand in candidates:
        if not cand:
            continue
        expanded = os.path.abspath(os.path.expanduser(cand))
        if os.path.isdir(expanded):
            expanded = os.path.join(expanded, "cmdai.py")
        if os.path.normcase(expanded) == self_script:
            continue
        if os.path.isfile(expanded):
            return expanded
    return None


def _resolve_python_for_root(code_root: str) -> str:
    venv_py = os.path.join(code_root, ".venv", "Scripts", "python.exe")
    if os.path.isfile(venv_py):
        return venv_py
    venv_unix = os.path.join(code_root, ".venv", "bin", "python")
    if os.path.isfile(venv_unix):
        return venv_unix
    return sys.executable or "python"


def _build_isolated_env(code_root: str) -> Dict[str, str]:
    """Prepare a clean child environment so CMDAI-CODE loads its own src/cmdai & config.json."""
    env = os.environ.copy()
    env.pop("CMDAI_ROOT", None)
    env["CMDAI_CODE_ROOT"] = code_root
    env["PYTHONIOENCODING"] = "utf-8"

    self_root = os.path.normcase(os.path.abspath(app_root()))
    py_path = env.get("PYTHONPATH", "")
    if py_path:
        kept = []
        for entry in py_path.split(os.pathsep):
            if not entry:
                continue
            norm_e = os.path.normcase(os.path.abspath(entry))
            if norm_e == self_root or norm_e.startswith(self_root + os.sep):
                continue
            kept.append(entry)
        if kept:
            env["PYTHONPATH"] = os.pathsep.join(kept)
        else:
            env.pop("PYTHONPATH", None)
    return env


def _load_recent_code_sessions(code_root: str, limit: int = 6) -> List[Dict[str, Any]]:
    sess_dir = os.path.join(code_root, "app", "sessions")
    if not os.path.isdir(sess_dir):
        return []
    entries: List[Tuple[float, Dict[str, Any]]] = []
    try:
        for fname in os.listdir(sess_dir):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(sess_dir, fname)
            try:
                mtime = os.path.getmtime(fpath)
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entries.append(
                    (
                        mtime,
                        {
                            "id": data.get("id") or fname[:-5],
                            "title": data.get("title") or "Untitled",
                            "workdir": data.get("workdir") or "",
                            "model": data.get("model") or "",
                            "updated_at": data.get("updated_at") or "",
                        },
                    )
                )
            except Exception:
                continue
    except Exception:
        return []
    entries.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in entries[:limit]]


def get_cmdai_code_status() -> Dict[str, Any]:
    """Inspect whether CMDAI-CODE is available and collect recent session metadata."""
    script_path = resolve_cmdai_code_script()
    if not script_path:
        return {
            "installed": False,
            "script_path": "",
            "root_dir": "",
            "python_bin": "",
            "recent_sessions": [],
        }
    code_root = os.path.dirname(script_path)
    return {
        "installed": True,
        "script_path": script_path,
        "root_dir": code_root,
        "python_bin": _resolve_python_for_root(code_root),
        "recent_sessions": _load_recent_code_sessions(code_root),
    }


def build_headless_command(
    task: str,
    workdir: str = "",
    continue_session: bool = False,
    provider: str = "",
    model: str = "",
) -> Tuple[List[str], Dict[str, str], str]:
    """Construct `(argv, env, effective_workdir)` for CMDAI-CODE headless invocation."""
    script_path = resolve_cmdai_code_script()
    if not script_path:
        raise FileNotFoundError(
            "CMDAI-CODE installation not found. Set `cmdai_code_path` in config.json or CMDAI_CODE_PATH."
        )
    clean_task = (task or "").strip()
    if not clean_task:
        raise ValueError("Task description cannot be empty.")

    code_root = os.path.dirname(script_path)
    eff_workdir = os.path.abspath(workdir or os.getcwd())
    if not os.path.isdir(eff_workdir):
        eff_workdir = code_root

    python_bin = _resolve_python_for_root(code_root)
    cmd = [
        python_bin,
        script_path,
        "-p",
        clean_task,
        "--workdir",
        eff_workdir,
        "--output-format",
        "json",
    ]
    if continue_session:
        cmd.append("--continue")
    if provider.strip():
        cmd.extend(["--provider", provider.strip()])
    if model.strip():
        cmd.extend(["--model", model.strip()])

    return cmd, _build_isolated_env(code_root), eff_workdir


def parse_headless_json_output(raw_stdout: str) -> Optional[Dict[str, Any]]:
    """Extract the headless JSON payload emitted by CMDAI-CODE."""
    txt = (raw_stdout or "").strip()
    if not txt:
        return None
    try:
        parsed = json.loads(txt)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    for line in reversed(txt.splitlines()):
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
    return None


def run_cmdai_code_headless(
    task: str,
    workdir: str = "",
    continue_session: bool = False,
    provider: str = "",
    model: str = "",
    timeout: int = 300,
) -> Dict[str, Any]:
    """Execute a coding task via `CMDAI-CODE -p ... --output-format json`."""
    try:
        cmd, env, eff_workdir = build_headless_command(
            task=task,
            workdir=workdir,
            continue_session=continue_session,
            provider=provider,
            model=model,
        )
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "response": "",
            "tool_calls": [],
            "session_id": "",
        }

    try:
        proc = subprocess.run(
            cmd,
            cwd=eff_workdir,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(5, int(timeout or 300)),
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": f"CMDAI-CODE timed out after {timeout}s.",
            "response": "",
            "tool_calls": [],
            "session_id": "",
            "workdir": eff_workdir,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Failed to run CMDAI-CODE: {exc}",
            "response": "",
            "tool_calls": [],
            "session_id": "",
            "workdir": eff_workdir,
        }

    payload = parse_headless_json_output(proc.stdout)
    if payload is not None:
        return {
            "ok": bool(payload.get("ok", proc.returncode == 0)),
            "session_id": payload.get("session_id", ""),
            "provider": payload.get("provider", ""),
            "model": payload.get("model", ""),
            "response": payload.get("response", ""),
            "tool_calls": payload.get("tool_calls", []),
            "error": payload.get("error") or (proc.stderr.strip() if proc.returncode != 0 else ""),
            "returncode": proc.returncode,
            "workdir": eff_workdir,
        }

    return {
        "ok": proc.returncode == 0,
        "session_id": "",
        "provider": provider,
        "model": model,
        "response": proc.stdout.strip(),
        "tool_calls": [],
        "error": proc.stderr.strip() if proc.returncode != 0 else "",
        "returncode": proc.returncode,
        "workdir": eff_workdir,
    }


def launch_cmdai_code_window(
    workdir: str = "",
    initial_task: str = "",
    wait: bool = False,
    timeout: int = 600,
) -> Dict[str, Any]:
    """Open CMDAI-CODE visibly in a new terminal window, optionally watching until completion."""
    import tempfile
    import time
    import uuid

    script_path = resolve_cmdai_code_script()
    if not script_path:
        return {"ok": False, "error": "CMDAI-CODE script not found."}

    code_root = os.path.dirname(script_path)
    if workdir and workdir.strip():
        eff_workdir = os.path.abspath(os.path.expanduser(workdir.strip()))
        try:
            os.makedirs(eff_workdir, exist_ok=True)
        except Exception:
            eff_workdir = code_root
    else:
        eff_workdir = os.path.abspath(os.getcwd())
        if not os.path.isdir(eff_workdir):
            eff_workdir = code_root

    python_bin = _resolve_python_for_root(code_root)
    env = _build_isolated_env(code_root)
    creation_flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) if os.name == "nt" else 0

    task_clean = (initial_task or "").strip()
    if not task_clean:
        cmd = [python_bin, script_path, "--workdir", eff_workdir]
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=eff_workdir,
                env=env,
                creationflags=creation_flags,
            )
            return {
                "ok": True,
                "pid": proc.pid,
                "workdir": eff_workdir,
                "script_path": script_path,
                "mode": "terminal_window",
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # When a task is given, run via a dedicated wrapper script in a new console window
    # so Windows path quoting never breaks and CMDAI can watch the live log + exit status.
    run_id = uuid.uuid4().hex[:10]
    bridge_dir = os.path.join(tempfile.gettempdir(), "cmdai_code_bridge")
    os.makedirs(bridge_dir, exist_ok=True)
    log_path = os.path.join(bridge_dir, f"run_{run_id}.log")
    done_path = os.path.join(bridge_dir, f"run_{run_id}.done")
    wrapper_py = os.path.join(bridge_dir, f"wrap_{run_id}.py")
    launcher_cmd = os.path.join(bridge_dir, f"launch_{run_id}.cmd")

    wrapper_code = (
        "import json, os, subprocess, sys\n"
        f"cmd = {json.dumps([python_bin, script_path, '--workdir', eff_workdir, '-p', task_clean])}\n"
        f"cwd = {json.dumps(eff_workdir)}\n"
        f"log_path = {json.dumps(log_path)}\n"
        f"done_path = {json.dumps(done_path)}\n"
        "rc = 1\n"
        "with open(log_path, 'w', encoding='utf-8', errors='replace') as lf:\n"
        "    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')\n"
        "    for line in proc.stdout:\n"
        "        sys.stdout.write(line)\n"
        "        sys.stdout.flush()\n"
        "        lf.write(line)\n"
        "        lf.flush()\n"
        "    rc = proc.wait()\n"
        "with open(done_path, 'w', encoding='utf-8') as df:\n"
        "    json.dump({'returncode': rc}, df)\n"
        "print(f'\\n[CMDAI-CODE finished with exit code {rc}]')\n"
    )
    with open(wrapper_py, "w", encoding="utf-8") as f:
        f.write(wrapper_code)

    if os.name == "nt":
        bat_body = (
            "@echo off\r\n"
            "chcp 65001 >nul\r\n"
            f"title CMDAI-CODE - {eff_workdir}\r\n"
            f'"{python_bin}" -u "{wrapper_py}"\r\n'
        )
        with open(launcher_cmd, "w", encoding="utf-8") as f:
            f.write(bat_body)
        pop_cmd = ["cmd.exe", "/k", launcher_cmd]
    else:
        pop_cmd = [python_bin, "-u", wrapper_py]

    try:
        proc = subprocess.Popen(
            pop_cmd,
            cwd=eff_workdir,
            env=env,
            creationflags=creation_flags,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    if not wait:
        return {
            "ok": True,
            "pid": proc.pid,
            "workdir": eff_workdir,
            "script_path": script_path,
            "log_path": log_path,
            "mode": "terminal_window",
        }

    deadline = time.time() + max(10, int(timeout or 600))
    rc: Optional[int] = None
    while time.time() < deadline:
        if os.path.isfile(done_path):
            try:
                with open(done_path, "r", encoding="utf-8") as df:
                    rc = int(json.load(df).get("returncode", 0))
                break
            except Exception:
                pass
        if proc.poll() is not None and not os.path.isfile(done_path):
            rc = proc.returncode
            break
        time.sleep(0.5)

    out_text = ""
    if os.path.isfile(log_path):
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as lf:
                out_text = lf.read().strip()
        except Exception:
            pass
    if len(out_text) > 6000:
        out_text = "... [truncated]\n" + out_text[-6000:]

    if rc is None:
        return {
            "ok": True,
            "pid": proc.pid,
            "workdir": eff_workdir,
            "script_path": script_path,
            "mode": "terminal_window",
            "response": f"CMDAI-CODE is still running in terminal window (PID {proc.pid}).\n{out_text}".strip(),
        }

    return {
        "ok": rc == 0,
        "pid": proc.pid,
        "returncode": rc,
        "workdir": eff_workdir,
        "script_path": script_path,
        "mode": "terminal_window",
        "response": out_text or f"CMDAI-CODE finished in {eff_workdir} with exit code {rc}.",
        "error": "" if rc == 0 else f"CMDAI-CODE exited with code {rc}.",
    }


CMDAI_CODE_REPO_URL = "https://github.com/DevP10TR/CMDAI-CODE.git"
CMDAI_CODE_ZIP_URL = "https://github.com/DevP10TR/CMDAI-CODE/archive/refs/heads/main.zip"


def install_cmdai_code(target_dir: str = "") -> Dict[str, Any]:
    """Install the latest CMDAI-CODE from GitHub into target_dir (default ~/CMDAI-CODE)."""
    import shutil
    import tempfile
    import zipfile

    dest_dir = os.path.abspath(
        os.path.expanduser(target_dir.strip() or os.path.join(os.path.expanduser("~"), "CMDAI-CODE"))
    )
    if os.path.isfile(os.path.join(dest_dir, "cmdai.py")):
        script_path = os.path.join(dest_dir, "cmdai.py")
        settings = get_settings()
        settings.config["cmdai_code_path"] = script_path
        settings.save_config()
        return {"ok": True, "script_path": script_path, "root": dest_dir, "method": "existing"}

    parent = os.path.dirname(dest_dir)
    os.makedirs(parent, exist_ok=True)
    method = ""

    git_bin = shutil.which("git")
    if git_bin:
        try:
            proc = subprocess.run(
                [git_bin, "clone", "--depth", "1", CMDAI_CODE_REPO_URL, dest_dir],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
            if proc.returncode == 0 and os.path.isfile(os.path.join(dest_dir, "cmdai.py")):
                method = "git"
        except Exception:
            pass

    if not method:
        try:
            import requests

            resp = requests.get(CMDAI_CODE_ZIP_URL, timeout=60)
            resp.raise_for_status()
            with tempfile.TemporaryDirectory() as tmpdir:
                zpath = os.path.join(tmpdir, "cmdai_code.zip")
                with open(zpath, "wb") as f:
                    f.write(resp.content)
                with zipfile.ZipFile(zpath, "r") as zf:
                    zf.extractall(tmpdir)
                extracted = os.path.join(tmpdir, "CMDAI-CODE-main")
                if not os.path.isdir(extracted):
                    subs = [
                        os.path.join(tmpdir, d)
                        for d in os.listdir(tmpdir)
                        if os.path.isdir(os.path.join(tmpdir, d))
                    ]
                    extracted = subs[0] if subs else ""
                if not extracted or not os.path.isfile(os.path.join(extracted, "cmdai.py")):
                    return {"ok": False, "error": "Downloaded archive did not contain cmdai.py."}
                if os.path.exists(dest_dir):
                    shutil.rmtree(dest_dir, ignore_errors=True)
                shutil.move(extracted, dest_dir)
                method = "zip"
        except Exception as exc:
            return {"ok": False, "error": f"Failed to install CMDAI-CODE: {exc}"}

    script_path = os.path.join(dest_dir, "cmdai.py")
    if not os.path.isfile(script_path):
        return {"ok": False, "error": "cmdai.py not found after installation."}

    settings = get_settings()
    settings.config["cmdai_code_path"] = script_path
    settings.save_config()
    return {"ok": True, "script_path": script_path, "root": dest_dir, "method": method}


_ACTIVE_CODE_SESSIONS: Dict[str, Dict[str, Any]] = {}
_BRIDGE_VERSION = 4


def _ensure_monitor_window(eff_workdir: str, log_path: str) -> Optional[int]:
    """Open one persistent CMDAI-CODE TUI window bridged via IPC for eff_workdir."""
    norm_wd = os.path.normcase(os.path.abspath(eff_workdir))
    sess = _ACTIVE_CODE_SESSIONS.get(norm_wd) or {}
    existing_proc = sess.get("_monitor_proc")
    if existing_proc is not None and existing_proc.poll() is None:
        if sess.get("_bridge_version") == _BRIDGE_VERSION:
            return existing_proc.pid
        try:
            existing_proc.terminate()
        except Exception:
            pass

    script_path = resolve_cmdai_code_script()
    if not script_path:
        return None
    code_root = os.path.dirname(script_path)
    python_bin = _resolve_python_for_root(code_root)
    env = _build_isolated_env(code_root)

    bridge_dir = os.path.dirname(log_path)
    os.makedirs(bridge_dir, exist_ok=True)
    wd_hash = f"{abs(hash(norm_wd)) & 0xFFFFFFFF:08x}"
    ipc_dir = os.path.join(bridge_dir, f"ipc_{wd_hash}")
    os.makedirs(ipc_dir, exist_ok=True)
    for stale in ("ready.json", "req.json", "ask_req.json"):
        sp = os.path.join(ipc_dir, stale)
        if os.path.isfile(sp):
            try:
                os.remove(sp)
            except Exception:
                pass

    bridge_py = os.path.join(bridge_dir, f"tui_bridge_{wd_hash}.py")
    bat_path = os.path.join(bridge_dir, f"mon_{wd_hash}.cmd")

    bridge_code = (
        "import json, os, sys, threading, time\n"
        f"CODE_ROOT = {json.dumps(code_root)}\n"
        f"WORKDIR = {json.dumps(eff_workdir)}\n"
        f"IPC_DIR = {json.dumps(ipc_dir)}\n"
        "src_dir = os.path.join(CODE_ROOT, 'src')\n"
        "if src_dir not in sys.path:\n"
        "    sys.path.insert(0, src_dir)\n"
        "if CODE_ROOT not in sys.path:\n"
        "    sys.path.append(CODE_ROOT)\n"
        "try:\n"
        "    os.chdir(WORKDIR)\n"
        "except Exception:\n"
        "    pass\n"
        "try:\n"
        "    if os.name == 'nt':\n"
        "        import ctypes\n"
        "        ctypes.windll.kernel32.SetConsoleOutputCP(65001)\n"
        "        ctypes.windll.kernel32.SetConsoleCP(65001)\n"
        "    if hasattr(sys.stdout, 'reconfigure'):\n"
        "        sys.stdout.reconfigure(encoding='utf-8', errors='replace')\n"
        "    if hasattr(sys.stderr, 'reconfigure'):\n"
        "        sys.stderr.reconfigure(encoding='utf-8', errors='replace')\n"
        "    sys.stdout.write('\\x1b[2J\\x1b[3J\\x1b[H\\x1b]11;#000000\\x07')\n"
        "    sys.stdout.flush()\n"
        "except Exception:\n"
        "    pass\n"
        "from cmdai.tui.app import CMDAICodeTUI\n"
        "\n"
        "def _atomic_write(path, data):\n"
        "    tmp = path + '.tmp'\n"
        "    with open(tmp, 'w', encoding='utf-8') as f:\n"
        "        json.dump(data, f, ensure_ascii=False)\n"
        "    os.replace(tmp, path)\n"
        "\n"
        "class BridgedCMDAICodeTUI(CMDAICodeTUI):\n"
        "    def _maybe_auto_name_session(self):\n"
        "        # Prevent background session-naming LLM calls from colliding on single-slot local servers.\n"
        "        pass\n"
        "\n"
        "    def on_mount(self):\n"
        "        super().on_mount()\n"
        "        try:\n"
        "            self.agent_mode = 'auto'\n"
        "            if hasattr(self, 'agent_ctx') and self.agent_ctx is not None:\n"
        "                self.agent_ctx.mode = 'auto'\n"
        "            if hasattr(self, '_cmd_mode'):\n"
        "                self._cmd_mode('auto')\n"
        "            self._update_meta_bars()\n"
        "        except Exception:\n"
        "            pass\n"
        "        _atomic_write(os.path.join(IPC_DIR, 'ready.json'), {\n"
        "            'pid': os.getpid(),\n"
        "            'bridge_version': 4,\n"
        "            'session_id': getattr(self, 'current_session_id', ''),\n"
        "            'workdir': WORKDIR,\n"
        "        })\n"
        "        t = threading.Thread(target=self._bridge_loop, daemon=True)\n"
        "        t.start()\n"
        "\n"
        "    def _handle_agent_ask(self, question, options=None, *args, **kwargs):\n"
        "        opts = [str(o) for o in (options or ['Yes, proceed', 'No, skip'])]\n"
        "        details = args[0] if (args and isinstance(args[0], dict)) else kwargs.get('details', {})\n"
        "        ask_id = f'{int(time.time() * 1000)}_{os.getpid()}'\n"
        "        ask_file = os.path.join(IPC_DIR, 'ask_req.json')\n"
        "        resp_p = os.path.join(IPC_DIR, f'ask_resp_{ask_id}.json')\n"
        "        hb_file = os.path.join(IPC_DIR, 'heartbeat.json')\n"
        "        try:\n"
        "            self.call_from_thread(self.notify, f'CMDAI deciding: {str(question)[:60]}', title='CMDAI Controller', timeout=2.5)\n"
        "        except Exception:\n"
        "            pass\n"
        "        _atomic_write(ask_file, {\n"
        "            'id': ask_id,\n"
        "            'question': str(question or ''),\n"
        "            'options': opts,\n"
        "            'details': details if isinstance(details, dict) else {},\n"
        "            'ts': time.time(),\n"
        "        })\n"
        "        deadline = time.time() + 120.0\n"
        "        last_hb = 0.0\n"
        "        while time.time() < deadline:\n"
        "            now_t = time.time()\n"
        "            if now_t - last_hb >= 1.0:\n"
        "                last_hb = now_t\n"
        "                try:\n"
        "                    _cur_todos = list(getattr(getattr(self, 'agent_ctx', None), 'todo_list', []) or [])\n"
        "                    _atomic_write(hb_file, {'ts': now_t, 'is_generating': True, 'todo_list': _cur_todos})\n"
        "                except Exception:\n"
        "                    pass\n"
        "            if os.path.isfile(resp_p):\n"
        "                try:\n"
        "                    with open(resp_p, 'r', encoding='utf-8') as rf:\n"
        "                        data = json.load(rf)\n"
        "                    try:\n"
        "                        os.remove(resp_p)\n"
        "                    except Exception:\n"
        "                        pass\n"
        "                    ans = str(data.get('answer') or '').strip()\n"
        "                    if ans:\n"
        "                        try:\n"
        "                            self.call_from_thread(self.notify, f'CMDAI decision: {ans}', title='CMDAI Controller', timeout=2.5)\n"
        "                        except Exception:\n"
        "                            pass\n"
        "                        return ans\n"
        "                except Exception:\n"
        "                    pass\n"
        "            time.sleep(0.15)\n"
        "        return opts[0] if opts else 'Yes, proceed'\n"
        "\n"
        "    def _tui_state_snapshot(self):\n"
        "        ctx = getattr(self, 'agent_ctx', None)\n"
        "        todo_list = list(getattr(ctx, 'todo_list', []) or []) if ctx is not None else []\n"
        "        mod_files = list(getattr(self, '_last_modified_files', []) or [])\n"
        "        diff_stats = dict(getattr(self, '_turn_diff_stats', {}) or {})\n"
        "        pend = bool(getattr(self, '_post_turn_approval_active', False) or getattr(self, '_pending_approval_evt', None))\n"
        "        return {\n"
        "            'session_id': getattr(self, 'current_session_id', ''),\n"
        "            'workdir': getattr(self, 'workdir', WORKDIR),\n"
        "            'agent_mode': getattr(self, 'agent_mode', 'auto'),\n"
        "            'model': getattr(self, 'current_model_id', ''),\n"
        "            'is_generating': bool(getattr(self, 'is_generating', False)),\n"
        "            'todo_list': todo_list,\n"
        "            'modified_files': mod_files,\n"
        "            'diff_stats': diff_stats,\n"
        "            'pending_approval': pend,\n"
        "        }\n"
        "\n"
        "    def _bridge_loop(self):\n"
        "        req_file = os.path.join(IPC_DIR, 'req.json')\n"
        "        while True:\n"
        "            req_id = ''\n"
        "            try:\n"
        "                if not os.path.isfile(req_file):\n"
        "                    time.sleep(0.15)\n"
        "                    continue\n"
        "                try:\n"
        "                    with open(req_file, 'r', encoding='utf-8') as f:\n"
        "                        req = json.load(f)\n"
        "                    os.remove(req_file)\n"
        "                except Exception:\n"
        "                    time.sleep(0.1)\n"
        "                    continue\n"
        "                req_id = str(req.get('id') or int(time.time() * 1000))\n"
        "                act = str(req.get('action') or 'run').strip().lower()\n"
        "                task = str(req.get('task') or '').strip()\n"
        "                req_mode = str(req.get('mode') or '').strip().lower()\n"
        "                new_sess = bool(req.get('new_session', False))\n"
        "                timeout_s = max(10, int(req.get('timeout', 300) or 300))\n"
        "                resp_file = os.path.join(IPC_DIR, f'resp_{req_id}.json')\n"
        "                if act == 'reset' or task == '__RESET__':\n"
        "                    self.call_from_thread(self.handle_submit, '/new')\n"
        "                    time.sleep(0.35)\n"
        "                    snap = self._tui_state_snapshot()\n"
        "                    snap.update({'ok': True, 'response': 'Session reset.', 'tool_calls': []})\n"
        "                    _atomic_write(resp_file, snap)\n"
        "                    continue\n"
        "                if act == 'status':\n"
        "                    snap = self._tui_state_snapshot()\n"
        "                    snap['ok'] = True\n"
        "                    _atomic_write(resp_file, snap)\n"
        "                    continue\n"
        "                if act == 'set_mode':\n"
        "                    target_mode = req_mode or task.lower()\n"
        "                    if target_mode not in ('auto', 'plan', 'code'):\n"
        "                        _atomic_write(resp_file, {'ok': False, 'error': f'Invalid mode: {target_mode}', 'response': ''})\n"
        "                        continue\n"
        "                    self.call_from_thread(self._cmd_mode, target_mode)\n"
        "                    time.sleep(0.15)\n"
        "                    snap = self._tui_state_snapshot()\n"
        "                    snap.update({'ok': True, 'action': 'set_mode', 'agent_mode': target_mode, 'response': f'CMDAI-CODE mode set to {target_mode}.', 'tool_calls': []})\n"
        "                    _atomic_write(resp_file, snap)\n"
        "                    continue\n"
        "                if act in ('approve', 'reject'):\n"
        "                    def _do_approval(a=act):\n"
        "                        from cmdai.tui.widgets import ApprovalBar\n"
        "                        act_name = 'allow' if a == 'approve' else 'reject'\n"
        "                        files = list(getattr(self, '_last_modified_files', []) or [])\n"
        "                        self.on_approval_action_selected(ApprovalBar.ActionSelected(act_name, {'files': files}))\n"
        "                    self.call_from_thread(_do_approval)\n"
        "                    time.sleep(0.25)\n"
        "                    snap = self._tui_state_snapshot()\n"
        "                    msg = 'Changes approved.' if act == 'approve' else 'Changes rejected and reverted.'\n"
        "                    snap.update({'ok': True, 'action': act, 'response': f'{msg} in CMDAI-CODE.', 'tool_calls': []})\n"
        "                    _atomic_write(resp_file, snap)\n"
        "                    continue\n"
        "                if act == 'interrupt':\n"
        "                    self.call_from_thread(self.action_handle_interrupt)\n"
        "                    time.sleep(0.2)\n"
        "                    snap = self._tui_state_snapshot()\n"
        "                    snap.update({'ok': True, 'action': 'interrupt', 'response': 'Interrupted active generation in CMDAI-CODE.', 'tool_calls': []})\n"
        "                    _atomic_write(resp_file, snap)\n"
        "                    continue\n"
        "                if act == 'command':\n"
        "                    cmd_str = task if task.startswith(('/', '!', '$ ')) else ('!' + task)\n"
        "                    prev_msgs = len(getattr(self, 'messages', []))\n"
        "                    self.call_from_thread(self.handle_submit, cmd_str)\n"
        "                    time.sleep(0.5)\n"
        "                    cmd_deadline = time.time() + min(30, timeout_s)\n"
        "                    while getattr(self, 'is_generating', False) and time.time() < cmd_deadline:\n"
        "                        time.sleep(0.2)\n"
        "                    msgs = list(getattr(self, 'messages', []))[prev_msgs:]\n"
        "                    parts = [str(m.get('content', '')).strip() for m in msgs if m.get('role') == 'assistant' and str(m.get('content', '')).strip()]\n"
        "                    snap = self._tui_state_snapshot()\n"
        "                    snap.update({'ok': True, 'action': 'command', 'response': '\\n\\n'.join(parts).strip() or f'Executed {cmd_str} in CMDAI-CODE TUI.', 'tool_calls': []})\n"
        "                    _atomic_write(resp_file, snap)\n"
        "                    continue\n"
        "                if req_mode in ('auto', 'plan', 'code') and req_mode != getattr(self, 'agent_mode', ''):\n"
        "                    self.call_from_thread(self._cmd_mode, req_mode)\n"
        "                    time.sleep(0.15)\n"
        "                elif not req_mode and getattr(self, 'agent_mode', '') not in ('auto', 'plan'):\n"
        "                    self.call_from_thread(self._cmd_mode, 'auto')\n"
        "                    time.sleep(0.15)\n"
        "                if new_sess and len(getattr(self, 'messages', [])) > 0:\n"
        "                    self.call_from_thread(self.handle_submit, '/new')\n"
        "                    time.sleep(0.35)\n"
        "                prev_msgs = len(getattr(self, 'messages', []))\n"
        "                prev_tr = len(getattr(self, '_transcript', []))\n"
        "                if getattr(self, 'agent_runner', None) is not None:\n"
        "                    self.agent_runner.ask_handler = self._handle_agent_ask\n"
        "                self.call_from_thread(self.handle_submit, task)\n"
        "                start_deadline = time.time() + 5.0\n"
        "                while time.time() < start_deadline:\n"
        "                    if getattr(self, 'is_generating', False):\n"
        "                        break\n"
        "                    if len(getattr(self, 'messages', [])) > prev_msgs + 1:\n"
        "                        break\n"
        "                    time.sleep(0.1)\n"
        "                hb_file = os.path.join(IPC_DIR, 'heartbeat.json')\n"
        "                last_hb = 0.0\n"
        "                end_deadline = time.time() + timeout_s\n"
        "                approval_since = 0.0\n"
        "                while getattr(self, 'is_generating', False) and time.time() < end_deadline:\n"
        "                    now_t = time.time()\n"
        "                    if now_t - last_hb >= 1.0:\n"
        "                        last_hb = now_t\n"
        "                        try:\n"
        "                            _cur_todos = list(getattr(getattr(self, 'agent_ctx', None), 'todo_list', []) or [])\n"
        "                            _atomic_write(hb_file, {'ts': now_t, 'is_generating': True, 'todo_list': _cur_todos})\n"
        "                        except Exception:\n"
        "                            pass\n"
        "                    if getattr(self, '_pending_approval_evt', None) is not None and not getattr(self, '_pending_approval_evt').is_set():\n"
        "                        if not approval_since:\n"
        "                            approval_since = now_t\n"
        "                        elif now_t - approval_since > 1.0:\n"
        "                            break\n"
        "                    else:\n"
        "                        approval_since = 0.0\n"
        "                    time.sleep(0.25)\n"
        "                timed_out_gen = bool(getattr(self, 'is_generating', False)) and not bool(getattr(self, '_pending_approval_evt', None))\n"
        "                if timed_out_gen:\n"
        "                    try:\n"
        "                        self.call_from_thread(self.action_handle_interrupt)\n"
        "                    except Exception:\n"
        "                        pass\n"
        "                    rel_deadline = time.time() + 10.0\n"
        "                    while getattr(self, 'is_generating', False) and time.time() < rel_deadline:\n"
        "                        time.sleep(0.2)\n"
        "                    time.sleep(0.6)\n"
        "                else:\n"
        "                    time.sleep(0.2)\n"
        "                msgs = list(getattr(self, 'messages', []))[prev_msgs:]\n"
        "                parts = [\n"
        "                    str(m.get('content', '')).strip()\n"
        "                    for m in msgs\n"
        "                    if m.get('role') == 'assistant' and str(m.get('content', '')).strip()\n"
        "                ]\n"
        "                reply_txt = '\\n\\n'.join(parts).strip()\n"
        "                tr_items = list(getattr(self, '_transcript', []))[prev_tr:]\n"
        "                tcalls = []\n"
        "                for it in tr_items:\n"
        "                    if isinstance(it, dict) and it.get('type') == 'tool':\n"
        "                        tn = it.get('name') or it.get('tool_name') or ''\n"
        "                        if tn:\n"
        "                            tcalls.append(str(tn))\n"
        "                snap = self._tui_state_snapshot()\n"
        "                still_gen = bool(getattr(self, 'is_generating', False)) and not snap['pending_approval']\n"
        "                snap.update({\n"
        "                    'ok': not still_gen,\n"
        "                    'response': reply_txt,\n"
        "                    'tool_calls': tcalls,\n"
        "                    'error': 'CMDAI-CODE timed out.' if still_gen else '',\n"
        "                })\n"
        "                _atomic_write(resp_file, snap)\n"
        "            except Exception as exc:\n"
        "                try:\n"
        "                    if req_id:\n"
        "                        _atomic_write(os.path.join(IPC_DIR, f'resp_{req_id}.json'), {\n"
        "                            'ok': False,\n"
        "                            'error': str(exc),\n"
        "                            'response': '',\n"
        "                            'tool_calls': [],\n"
        "                        })\n"
        "                except Exception:\n"
        "                    pass\n"
        "                time.sleep(0.25)\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    try:\n"
        "        BridgedCMDAICodeTUI(workdir=WORKDIR).run()\n"
        "    finally:\n"
        "        try:\n"
        "            sys.stdout.write('\\x1b]111\\x07')\n"
        "            sys.stdout.flush()\n"
        "        except Exception:\n"
        "            pass\n"
    )
    with open(bridge_py, "w", encoding="utf-8") as f:
        f.write(bridge_code)

    creation_flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) if os.name == "nt" else 0
    if os.name == "nt":
        with open(bat_path, "w", encoding="utf-8") as f:
            f.write(
                "@echo off\r\n"
                "chcp 65001 >nul\r\n"
                f"title CMDAI-CODE TUI - {eff_workdir}\r\n"
                f'"{python_bin}" -u "{bridge_py}"\r\n'
            )
        pop_cmd = ["cmd.exe", "/k", bat_path]
    else:
        pop_cmd = [python_bin, "-u", bridge_py]

    try:
        proc = subprocess.Popen(pop_cmd, cwd=eff_workdir, env=env, creationflags=creation_flags)
        entry = _ACTIVE_CODE_SESSIONS.setdefault(norm_wd, {})
        entry["_monitor_proc"] = proc
        entry["_ipc_dir"] = ipc_dir
        entry["_bridge_version"] = _BRIDGE_VERSION
        return proc.pid
    except Exception:
        return None


_DANGEROUS_CMD_MARKERS = (
    "format ",
    "format\t",
    "shutdown",
    "rm -rf /",
    "rm -fr /",
    "del /s /q c:\\",
    "rd /s /q c:\\",
    "rmdir /s /q c:\\",
    "reg delete",
    "diskpart",
    "bcdedit",
)


def _match_option(candidate: str, options: List[str]) -> str:
    cand = (candidate or "").strip()
    if not cand:
        return ""
    if not options:
        return cand
    cand_low = cand.lower()
    for opt in options:
        if opt.lower() == cand_low:
            return opt
    for opt in options:
        if opt.lower() in cand_low or cand_low in opt.lower():
            return opt
    return cand


def decide_for_cmdai_code(
    question: str,
    options: List[str],
    task: str = "",
    workdir: str = "",
    details: Optional[Dict[str, Any]] = None,
    llm_decide: Optional[Callable[[str, List[str]], str]] = None,
) -> str:
    """Decide how CMDAI answers an ask/permission prompt coming from CMDAI-CODE."""
    opts = [str(o) for o in (options or []) if str(o).strip()]
    det = details if isinstance(details, dict) else {}
    haystack = " ".join(
        [
            str(question or ""),
            str(det.get("cmd") or ""),
            str(det.get("command") or ""),
            str(det.get("target") or ""),
            str(det.get("path") or ""),
        ]
    ).lower()

    if any(marker in haystack for marker in _DANGEROUS_CMD_MARKERS):
        for reject_opt in opts:
            low = reject_opt.lower()
            if "no" in low or "skip" in low or "reject" in low or "cancel" in low or "deny" in low:
                return reject_opt
        return "No, skip"

    if llm_decide is not None:
        try:
            ans = str(llm_decide(question, opts) or "").strip()
            if ans:
                return _match_option(ans, opts)
        except Exception:
            pass

    for opt in opts:
        if opt.lower() in ("yes, proceed", "allow", "yes"):
            return opt
    if opts:
        return opts[0]
    return "Yes, proceed"


def _send_ipc_turn(
    ipc_dir: str,
    proc: Any,
    task: str = "",
    new_session: bool = False,
    timeout: int = 1800,
    action: str = "run",
    mode: str = "",
    decision_handler: Optional[Callable[[str, List[str], Dict[str, Any]], str]] = None,
    todo_handler: Optional[Callable[[List[Any]], None]] = None,
) -> Optional[Dict[str, Any]]:
    """Send a command or turn to an already-launched BridgedCMDAICodeTUI window via IPC."""
    import time
    import uuid

    ready_file = os.path.join(ipc_dir, "ready.json")
    ready_deadline = time.time() + 15.0
    while not os.path.isfile(ready_file) and time.time() < ready_deadline:
        if proc is not None and proc.poll() is not None:
            return None
        time.sleep(0.15)
    if not os.path.isfile(ready_file):
        return None

    req_id = uuid.uuid4().hex[:10]
    req_file = os.path.join(ipc_dir, "req.json")
    req_tmp = req_file + ".tmp"
    resp_file = os.path.join(ipc_dir, f"resp_{req_id}.json")
    hb_file = os.path.join(ipc_dir, "heartbeat.json")
    ask_req_file = os.path.join(ipc_dir, "ask_req.json")
    decisions: List[Dict[str, Any]] = []
    last_todo_sig = ""
    with open(req_tmp, "w", encoding="utf-8") as f:
        json.dump(
            {
                "id": req_id,
                "action": action,
                "task": task,
                "mode": mode,
                "new_session": new_session,
                "timeout": timeout,
            },
            f,
            ensure_ascii=False,
        )
    os.replace(req_tmp, req_file)

    deadline = time.time() + max(10, int(timeout or 1800)) + 10
    while time.time() < deadline:
        if os.path.isfile(ask_req_file):
            try:
                with open(ask_req_file, "r", encoding="utf-8") as af:
                    ask_req = json.load(af)
                try:
                    os.remove(ask_req_file)
                except Exception:
                    pass
                ask_id = str(ask_req.get("id") or "")
                q = str(ask_req.get("question") or "")
                opts = [str(o) for o in (ask_req.get("options") or [])]
                det = ask_req.get("details") if isinstance(ask_req.get("details"), dict) else {}
                if decision_handler is not None:
                    chosen = str(decision_handler(q, opts, det) or "").strip() or (opts[0] if opts else "Yes, proceed")
                else:
                    chosen = decide_for_cmdai_code(q, opts, task=task, workdir=ipc_dir, details=det)
                decisions.append({"question": q, "options": opts, "decision": chosen})
                if ask_id:
                    ask_resp_file = os.path.join(ipc_dir, f"ask_resp_{ask_id}.json")
                    ask_resp_tmp = ask_resp_file + ".tmp"
                    with open(ask_resp_tmp, "w", encoding="utf-8") as rf:
                        json.dump({"answer": chosen}, rf, ensure_ascii=False)
                    os.replace(ask_resp_tmp, ask_resp_file)
                deadline = max(deadline, time.time() + 60.0)
            except Exception:
                pass
        if os.path.isfile(resp_file):
            try:
                with open(resp_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                try:
                    os.remove(resp_file)
                except Exception:
                    pass
                if isinstance(data, dict) and decisions:
                    data["decisions"] = decisions
                if todo_handler is not None and isinstance(data, dict) and isinstance(data.get("todo_list"), list) and data["todo_list"]:
                    try:
                        sig = json.dumps(data["todo_list"], sort_keys=True, ensure_ascii=False)
                        if sig != last_todo_sig:
                            last_todo_sig = sig
                            todo_handler(list(data["todo_list"]))
                    except Exception:
                        pass
                return data
            except Exception:
                pass
        if os.path.isfile(hb_file):
            try:
                with open(hb_file, "r", encoding="utf-8") as hf:
                    hb = json.load(hf)
                if hb.get("is_generating") and (time.time() - float(hb.get("ts", 0))) < 30.0:
                    deadline = max(deadline, time.time() + 60.0)
                hb_todos = hb.get("todo_list")
                if todo_handler is not None and isinstance(hb_todos, list) and hb_todos:
                    sig = json.dumps(hb_todos, sort_keys=True, ensure_ascii=False)
                    if sig != last_todo_sig:
                        last_todo_sig = sig
                        todo_handler(list(hb_todos))
            except Exception:
                pass
        if proc is not None and proc.poll() is not None:
            return {"ok": False, "error": "CMDAI-CODE TUI window was closed.", "response": "", "tool_calls": [], "decisions": decisions}
        time.sleep(0.15)
    return {"ok": False, "error": f"CMDAI-CODE TUI timed out after {timeout}s.", "response": "", "tool_calls": [], "decisions": decisions}


_LIST_CMD_RE = re.compile(
    r"^(?:dir(?:\s+/[a-z0-9]+)*|ls(?:\s+-[a-z0-9]+)*|tree(?:\s+/[a-z0-9]+)*|pwd|cd|get-childitem|gci)(?:\s+[\.\"\'\\/\w\-\s:]+)?$",
    re.IGNORECASE,
)
_FILE_CMD_RE = re.compile(r"^(?:type|cat|more|get-content|gc)\s+(.+)$", re.IGNORECASE)


def _enrich_cmdai_code_prompt(task: str, action: str = "run", workdir: str = "") -> Tuple[str, str]:
    """Ensure CMDAI-CODE receives rich natural-language prompts instead of bare shell commands."""
    act = (action or "run").strip().lower()
    t = (task or "").strip()
    if act not in ("run", "command") or not t:
        return act, t

    if act == "command" and t.startswith("/"):
        return "command", t

    if _LIST_CMD_RE.match(t) and len(t) < 120:
        t = (
            f"Przeanalizuj dokładnie zawartość katalogu roboczego projektu ({workdir or '.'}): "
            "wylistuj wszystkie pliki i podkatalogi, odczytaj kluczowe pliki źródłowe, szablony oraz pliki konfiguracyjne, "
            "zbadaj architekturę i logikę działania aplikacji, sprawdź czy w kodzie występują błędy lub brakujące elementy, "
            "a następnie przygotuj wyczerpujący raport techniczny podsumowujący strukturę i działanie projektu."
        )
        return "run", t

    m_file = _FILE_CMD_RE.match(t)
    if m_file:
        raw_arg = m_file.group(1).strip()
        is_quoted = (raw_arg.startswith('"') and raw_arg.endswith('"')) or (raw_arg.startswith("'") and raw_arg.endswith("'"))
        if is_quoted or len(raw_arg.split()) == 1:
            target_file = raw_arg.strip('"\'')
            t = (
                f"Odczytaj i szczegółowo przeanalizuj plik `{target_file}` w katalogu `{workdir or '.'}`: "
                "sprawdź jego strukturę, logikę, zależności oraz ewentualne błędy, i przedstaw pełne podsumowanie techniczne."
            )
            return "run", t

    if act == "command" and not t.startswith("/"):
        first_tok = t.split()[0].lower().lstrip("!")
        shell_bins = ("!", "pytest", "python", "npm", "pip", "git", "cargo", "go", "ruff", "flake8", "mypy")
        if len(t.split()) > 4 and not t.startswith("!") and first_tok not in shell_bins:
            act = "run"

    if act == "run" and not t.startswith("/") and len(t) < 80 and "\n" not in t:
        t = (
            f"{t}\n\n"
            "Wytyczne wykonawcze:\n"
            "1. Dokładnie zbadaj strukturę plików i istniejący kod źródłowy w katalogu roboczym projektu.\n"
            "2. Wykonaj kompletną analizę lub implementację zgodnie z powyższym celem, dbając o wszystkie szczegóły techniczne.\n"
            "3. Zweryfikuj poprawność zmian oraz spójność kodu i przedstaw szczegółowe podsumowanie."
        )

    return act, t


def send_to_cmdai_code(
    task: str = "",
    workdir: str = "",
    action: str = "run",
    mode: str = "",
    new_session: bool = False,
    timeout: int = 1800,
    open_window: bool = True,
    decision_handler: Optional[Callable[[str, List[str], Dict[str, Any]], str]] = None,
    todo_handler: Optional[Callable[[List[Any]], None]] = None,
) -> Dict[str, Any]:
    """Persistent MCP/subagent-style bridge to CMDAI-CODE.

    Maintains one ongoing CMDAI-CODE session per workdir inside a live CMDAI-CODE TUI
    window (bridged via IPC), falling back to headless `--continue` if no TUI is active.
    """
    import tempfile

    act = (action or "run").strip().lower()
    mode_clean = (mode or "").strip().lower()
    eff_workdir = os.path.abspath(os.path.expanduser(workdir.strip())) if workdir and workdir.strip() else os.path.abspath(os.getcwd())
    os.makedirs(eff_workdir, exist_ok=True)
    norm_wd = os.path.normcase(eff_workdir)

    def _live_ipc() -> Optional[Tuple[str, Any]]:
        s = _ACTIVE_CODE_SESSIONS.get(norm_wd) or {}
        ipd = s.get("_ipc_dir")
        mp = s.get("_monitor_proc")
        if ipd and mp is not None and mp.poll() is None:
            return ipd, mp
        return None

    if act == "close":
        old = _ACTIVE_CODE_SESSIONS.pop(norm_wd, {}) or {}
        mp = old.get("_monitor_proc")
        if mp is not None and mp.poll() is None:
            try:
                mp.terminate()
            except Exception:
                pass
        return {
            "ok": True,
            "action": "close",
            "workdir": eff_workdir,
            "response": f"Closed CMDAI-CODE window for {eff_workdir}.",
        }

    if act == "reset":
        live = _live_ipc()
        if live:
            _send_ipc_turn(live[0], live[1], task="__RESET__", action="reset", timeout=15)
        old = _ACTIVE_CODE_SESSIONS.pop(norm_wd, {}) or {}
        if live:
            _ACTIVE_CODE_SESSIONS[norm_wd] = {
                "_monitor_proc": old["_monitor_proc"],
                "_ipc_dir": old["_ipc_dir"],
                "turns": 0,
                "agent_mode": old.get("agent_mode", "auto"),
                "workdir": eff_workdir,
            }
        return {"ok": True, "action": "reset", "workdir": eff_workdir, "response": f"Reset CMDAI-CODE session for {eff_workdir}."}

    if act in ("set_mode", "approve", "reject", "interrupt"):
        live = _live_ipc()
        sess = _ACTIVE_CODE_SESSIONS.setdefault(norm_wd, {"workdir": eff_workdir})
        if live:
            r = _send_ipc_turn(live[0], live[1], task=task, action=act, mode=mode_clean, timeout=30)
            if r is not None:
                if r.get("agent_mode"):
                    sess["agent_mode"] = r["agent_mode"]
                r.setdefault("action", act)
                r.setdefault("workdir", eff_workdir)
                return r
        if act == "set_mode":
            target_mode = mode_clean or (task or "").strip().lower() or "auto"
            if target_mode not in ("auto", "plan", "code"):
                return {"ok": False, "action": act, "workdir": eff_workdir, "error": f"Invalid mode: {target_mode}"}
            sess["agent_mode"] = target_mode
            return {"ok": True, "action": "set_mode", "agent_mode": target_mode, "workdir": eff_workdir, "response": f"CMDAI-CODE mode set to {target_mode}."}
        msg_map = {
            "approve": "Changes approved in CMDAI-CODE.",
            "reject": "Changes rejected in CMDAI-CODE.",
            "interrupt": "Interrupted active generation in CMDAI-CODE.",
        }
        if act in ("approve", "reject"):
            sess["pending_approval"] = False
        return {"ok": True, "action": act, "workdir": eff_workdir, "agent_mode": sess.get("agent_mode", "auto"), "response": msg_map[act]}

    if act == "status":
        st = get_cmdai_code_status()
        sess = _ACTIVE_CODE_SESSIONS.get(norm_wd) or {}
        live = _live_ipc()
        live_snap: Dict[str, Any] = {}
        if live:
            live_snap = _send_ipc_turn(live[0], live[1], action="status", timeout=15) or {}
        return {
            "ok": True,
            "action": "status",
            "installed": st["installed"],
            "window_open": bool(live),
            "workdir": eff_workdir,
            "active_session_id": live_snap.get("session_id") or sess.get("session_id", ""),
            "turns": int(sess.get("turns", 0)),
            "agent_mode": live_snap.get("agent_mode") or sess.get("agent_mode", "auto"),
            "is_generating": bool(live_snap.get("is_generating", False)),
            "todo_list": live_snap.get("todo_list") if "todo_list" in live_snap else sess.get("todo_list", []),
            "modified_files": live_snap.get("modified_files") if "modified_files" in live_snap else sess.get("modified_files", []),
            "pending_approval": bool(live_snap.get("pending_approval", sess.get("pending_approval", False))),
            "last_tool_calls": sess.get("tool_calls", []),
        }

    if act == "open":
        bridge_dir = os.path.join(tempfile.gettempdir(), "cmdai_code_bridge")
        log_path = os.path.join(bridge_dir, f"sess_{abs(hash(norm_wd)) & 0xFFFFFFFF:08x}.log")
        pid = _ensure_monitor_window(eff_workdir, log_path)
        if pid:
            return {"ok": True, "action": "open", "pid": pid, "workdir": eff_workdir, "mode": "tui_bridge"}
        return launch_cmdai_code_window(workdir=eff_workdir, initial_task="")

    task_clean = (task or "").strip()
    if act in ("run", "command") and task_clean:
        act, task_clean = _enrich_cmdai_code_prompt(task_clean, act, eff_workdir)
    if not task_clean:
        return {"ok": False, "error": f"Missing task/command for cmdai_code action='{act}'."}

    sess = _ACTIVE_CODE_SESSIONS.get(norm_wd) or {}
    if new_session:
        keep_proc = sess.get("_monitor_proc")
        keep_ipc = sess.get("_ipc_dir")
        sess = {}
        _ACTIVE_CODE_SESSIONS.pop(norm_wd, None)
        if keep_proc is not None and keep_proc.poll() is None and keep_ipc:
            _ACTIVE_CODE_SESSIONS[norm_wd] = {"_monitor_proc": keep_proc, "_ipc_dir": keep_ipc}
            sess = _ACTIVE_CODE_SESSIONS[norm_wd]

    turn_num = int(sess.get("turns", 0)) + 1
    should_continue = turn_num > 1 and not new_session

    bridge_dir = os.path.join(tempfile.gettempdir(), "cmdai_code_bridge")
    os.makedirs(bridge_dir, exist_ok=True)
    log_path = sess.get("log_path") or os.path.join(bridge_dir, f"sess_{abs(hash(norm_wd)) & 0xFFFFFFFF:08x}.log")

    if open_window:
        _ensure_monitor_window(eff_workdir, log_path)

    live = _live_ipc()
    res: Optional[Dict[str, Any]] = None
    if live:
        res = _send_ipc_turn(
            ipc_dir=live[0],
            proc=live[1],
            task=task_clean,
            new_session=new_session,
            timeout=timeout,
            action=act,
            mode=mode_clean,
            decision_handler=decision_handler,
            todo_handler=todo_handler,
        )

    if res is None:
        res = run_cmdai_code_headless(
            task=task_clean,
            workdir=eff_workdir,
            continue_session=should_continue,
            timeout=timeout,
        )

    eff_mode = res.get("agent_mode") or mode_clean or sess.get("agent_mode", "auto")
    sess_entry = _ACTIVE_CODE_SESSIONS.setdefault(norm_wd, {})
    sess_entry.update(
        {
            "session_id": res.get("session_id") or sess.get("session_id", ""),
            "turns": turn_num,
            "log_path": log_path,
            "agent_mode": eff_mode,
            "tool_calls": res.get("tool_calls", []),
            "todo_list": res.get("todo_list", sess.get("todo_list", [])),
            "modified_files": res.get("modified_files", sess.get("modified_files", [])),
            "pending_approval": bool(res.get("pending_approval", False)),
            "workdir": eff_workdir,
        }
    )

    if todo_handler is not None and sess_entry["todo_list"]:
        try:
            todo_handler(list(sess_entry["todo_list"]))
        except Exception:
            pass

    return {
        "ok": bool(res.get("ok")),
        "action": act,
        "turn": turn_num,
        "continued": should_continue,
        "session_id": sess_entry["session_id"],
        "agent_mode": eff_mode,
        "response": res.get("response", ""),
        "tool_calls": res.get("tool_calls", []),
        "todo_list": sess_entry["todo_list"],
        "modified_files": sess_entry["modified_files"],
        "diff_stats": res.get("diff_stats", {}),
        "pending_approval": sess_entry["pending_approval"],
        "decisions": res.get("decisions", []),
        "workdir": eff_workdir,
        "error": res.get("error", ""),
    }
