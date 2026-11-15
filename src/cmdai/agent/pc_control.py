"""Windows PC control tools for CMDAI (`<tool:system>`)."""

from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import subprocess
import time
import webbrowser
from typing import Any, Dict, List

import psutil

from ..core.notifier import send_desktop_notification


SUPPORTED_SYSTEM_ACTIONS = (
    "sysinfo",
    "processes",
    "open_app",
    "open_url",
    "clipboard_get",
    "clipboard_set",
    "notify",
    "volume",
    "kill_process",
)


def _format_bytes_gb(b: int) -> float:
    return round(float(b) / (1024 ** 3), 2)


def _query_gpu_summary() -> str:
    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        try:
            proc = subprocess.run(
                [
                    nvidia,
                    "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
        except Exception:
            pass
    return "N/A"


def _action_sysinfo() -> str:
    vm = psutil.virtual_memory()
    cpu_pct = psutil.cpu_percent(interval=0.1)
    boot_ts = psutil.boot_time()
    uptime_h = max(0.0, (time.time() - boot_ts) / 3600.0)

    disks: List[Dict[str, Any]] = []
    for part in psutil.disk_partitions(all=False):
        try:
            usage = psutil.disk_usage(part.mountpoint)
            disks.append(
                {
                    "mount": part.mountpoint,
                    "free_gb": _format_bytes_gb(usage.free),
                    "total_gb": _format_bytes_gb(usage.total),
                    "percent_used": usage.percent,
                }
            )
        except Exception:
            continue

    info = {
        "os": platform.platform(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_percent": cpu_pct,
        "ram_total_gb": _format_bytes_gb(vm.total),
        "ram_available_gb": _format_bytes_gb(vm.available),
        "ram_used_percent": vm.percent,
        "uptime_hours": round(uptime_h, 2),
        "gpu": _query_gpu_summary(),
        "disks": disks,
    }
    return json.dumps(info, indent=2, ensure_ascii=False)


def _action_processes(target: str = "", limit: int = 25) -> str:
    needle = (target or "").strip().lower()
    rows: List[Dict[str, Any]] = []
    for proc in psutil.process_iter(["pid", "name", "memory_info", "status"]):
        try:
            pinfo = proc.info
            name = str(pinfo.get("name") or "")
            if needle and needle not in name.lower() and needle != str(pinfo.get("pid")):
                continue
            mem_info = pinfo.get("memory_info")
            rss_mb = round(float(mem_info.rss) / (1024 * 1024), 1) if mem_info else 0.0
            rows.append(
                {
                    "pid": pinfo.get("pid"),
                    "name": name,
                    "mem_mb": rss_mb,
                    "status": pinfo.get("status") or "",
                }
            )
        except Exception:
            continue

    rows.sort(key=lambda r: r["mem_mb"], reverse=True)
    return json.dumps(rows[:limit], indent=2, ensure_ascii=False)


def _action_open_app(target: str) -> str:
    clean = (target or "").strip()
    if not clean:
        return "Error: `target` executable, app name, or file path is required for open_app."
    try:
        if os.name == "nt" and hasattr(os, "startfile"):
            try:
                os.startfile(clean)
                return f"Launched application/path: {clean}"
            except OSError:
                resolved = shutil.which(clean)
                if resolved:
                    subprocess.Popen([resolved])
                    return f"Launched application: {resolved}"
                raise
        subprocess.Popen([clean])
        return f"Launched application: {clean}"
    except Exception as exc:
        return f"Error opening '{clean}': {exc}"


def _action_open_url(target: str) -> str:
    clean = (target or "").strip()
    if not (clean.startswith("http://") or clean.startswith("https://")):
        return "Error: `open_url` only accepts http:// or https:// URLs."
    try:
        opened = webbrowser.open(clean)
        return f"Opened URL in browser: {clean}" if opened else f"Dispatched URL open: {clean}"
    except Exception as exc:
        return f"Error opening URL '{clean}': {exc}"


def _action_clipboard_get() -> str:
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-Clipboard -Raw"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=4,
        )
        if proc.returncode != 0:
            return f"Error reading clipboard: {proc.stderr.strip()}"
        return proc.stdout if proc.stdout else "(clipboard is empty)"
    except Exception as exc:
        return f"Error reading clipboard: {exc}"


def _action_clipboard_set(text: str) -> str:
    if text is None:
        text = ""
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Set-Clipboard -Value $input"],
            input=str(text),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=4,
        )
        if proc.returncode != 0:
            return f"Error writing clipboard: {proc.stderr.strip()}"
        return f"Copied {len(str(text))} character(s) to clipboard."
    except Exception as exc:
        return f"Error writing clipboard: {exc}"


def _action_volume(target: str, value: str) -> str:
    if os.name != "nt":
        return "Error: volume control is supported on Windows only."

    mode = (value or target or "").strip().lower()
    if not mode:
        return "Error: specify `value` as 'mute', 'unmute', 'up', 'down', or integer steps (e.g. 'up')."

    KEYEVENTF_EXTENDEDKEY = 0x0001
    KEYEVENTF_KEYUP = 0x0002
    VK_VOLUME_MUTE = 0xAD
    VK_VOLUME_DOWN = 0xAE
    VK_VOLUME_UP = 0xAF

    def _tap(vk: int, times: int = 1) -> None:
        user32 = ctypes.windll.user32
        for _ in range(times):
            user32.keybd_event(vk, 0, KEYEVENTF_EXTENDEDKEY, 0)
            user32.keybd_event(vk, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)

    try:
        if mode in ("mute", "toggle", "unmute"):
            _tap(VK_VOLUME_MUTE, 1)
            return f"Executed Windows volume '{mode}'."
        if mode == "up":
            _tap(VK_VOLUME_UP, 3)
            return "Increased Windows system volume."
        if mode == "down":
            _tap(VK_VOLUME_DOWN, 3)
            return "Decreased Windows system volume."
        if mode.isdigit():
            pct = max(0, min(100, int(mode)))
            # Reset to 0 using 50 down-steps (~2% per step on Windows), then step up cleanly.
            _tap(VK_VOLUME_DOWN, 50)
            steps = round(pct / 2)
            if steps > 0:
                _tap(VK_VOLUME_UP, steps)
            return f"Adjusted Windows system volume to ~{pct}%."
        return "Error: invalid volume mode. Use 'mute', 'up', 'down', or '0'-'100'."
    except Exception as exc:
        return f"Error adjusting volume: {exc}"


def _action_kill_process(target: str) -> str:
    clean = (target or "").strip()
    if not clean:
        return "Error: `target` PID or process name is required for kill_process."

    if clean.isdigit():
        pid = int(clean)
        if pid <= 4:
            return f"Refused: PID {pid} is a protected Windows system process."
        try:
            proc = psutil.Process(pid)
            name = proc.name()
            proc.terminate()
            return f"Terminated process {name} (PID {pid})."
        except psutil.NoSuchProcess:
            return f"Error: no process found with PID {pid}."
        except Exception as exc:
            return f"Error terminating PID {pid}: {exc}"

    needle = clean.lower()
    killed: List[str] = []
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            pid = int(proc.info["pid"])
            name = str(proc.info.get("name") or "")
            if pid <= 4:
                continue
            if name.lower() == needle or os.path.splitext(name.lower())[0] == os.path.splitext(needle)[0]:
                proc.terminate()
                killed.append(f"{name} (PID {pid})")
        except Exception:
            continue

    if not killed:
        return f"No running process matched {clean!r}."
    return "Terminated: " + ", ".join(killed)


def tool_system(ctx: Any, action: str, target: str = "", value: str = "") -> str:
    """Execute a Windows PC inspection or control action for `<tool:system>`."""
    act = (action or "").strip().lower()
    if act == "sysinfo":
        return _action_sysinfo()
    if act == "processes":
        return _action_processes(target=target)
    if act == "open_app":
        return _action_open_app(target=target or value)
    if act == "open_url":
        return _action_open_url(target=target or value)
    if act == "clipboard_get":
        return _action_clipboard_get()
    if act == "clipboard_set":
        return _action_clipboard_set(value if value else target)
    if act == "notify":
        title = target.strip() or "CMDAI"
        msg = value.strip() or target.strip() or "Notification from CMDAI"
        send_desktop_notification(title, msg)
        return f"Sent desktop notification: [{title}] {msg}"
    if act == "volume":
        return _action_volume(target=target, value=value)
    if act == "kill_process":
        return _action_kill_process(target=target)

    return (
        f"Error: unknown system action {action!r}. "
        f"Supported actions: {', '.join(SUPPORTED_SYSTEM_ACTIONS)}"
    )
