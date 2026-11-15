import argparse
import os
import sys
import traceback
from typing import Any, Dict, List

REPO_URL = "https://github.com/DevP10TR/CMDAI"
FIXED_ROOT_NAME = "CMDAI"


def _resolve_app_root() -> str:
    """Locate the CMDAI installation root.

    Priority: CMDAI_ROOT env -> repo layout next to this file ->
    fixed user location (~/CMDAI) -> fallback to file-based root.
    """
    override = os.environ.get("CMDAI_ROOT")
    if override and os.path.isdir(override):
        return os.path.abspath(override)
    here_based = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for marker in ("cmdai.py", "config.example.json", "requirements.txt"):
        if os.path.exists(os.path.join(here_based, marker)):
            return here_based
    fixed = os.path.join(os.path.expanduser("~"), FIXED_ROOT_NAME)
    if os.path.isdir(fixed):
        return fixed
    return here_based


APP_ROOT = _resolve_app_root()
SRC_PATH = os.path.join(APP_ROOT, "src")
if os.path.isdir(SRC_PATH) and SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from cmdai.core.launcher import install_global_launcher
from cmdai.core.settings import get_settings

if APP_ROOT not in sys.path:
    sys.path.append(APP_ROOT)

from cmdai.tui.app import CMDAICodeTUI


def setup_crash_logger():
    os.makedirs(os.path.join(APP_ROOT, "logs"), exist_ok=True)
    crash_log = os.path.join(APP_ROOT, "logs", "cmdai_crash.log")

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        with open(crash_log, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 60 + "\n")
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)

        print("\n[!] CMDAI encountered an unexpected error.")
        print(f"[!] Crash details saved to: {crash_log}")

        if os.environ.get("RUN_AI_KEEP_OPEN") == "1":
            input("\nPress Enter to exit...")

    sys.excepthook = handle_exception


_ANIM_MODULE = None


def _load_animator():
    """Load tools/install_anim.py as a module (shared step animation)."""
    global _ANIM_MODULE
    if _ANIM_MODULE is not None:
        return _ANIM_MODULE
    import importlib.util
    path = os.path.join(APP_ROOT, "tools", "install_anim.py")
    spec = importlib.util.spec_from_file_location("install_anim", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _ANIM_MODULE = mod
    return mod


def _animated_step(label, work_fn, min_time=1.0, timeout=300.0, header="", gap=0):
    """Run work_fn in a background thread while animating the step.

    work_fn returns (status, detail) with status OK/FAIL.
    Returns (status, detail). Falls back to plain output when the
    animator module is unavailable.
    """
    import tempfile
    import threading
    try:
        anim = _load_animator()
    except Exception:
        anim = None
    fd, flag = tempfile.mkstemp(prefix="cmdai-update-", suffix=".done")
    try:
        os.close(fd)
    except OSError:
        pass
    try:
        os.remove(flag)
    except OSError:
        pass
    result = {}

    def _work():
        try:
            result["status"], result["detail"] = work_fn()
        except Exception as e:
            result["status"], result["detail"] = ("FAIL", str(e))
        try:
            with open(flag, "w", encoding="utf-8") as f:
                f.write(result["status"])
        except OSError:
            pass

    thread = threading.Thread(target=_work, daemon=True)
    thread.start()
    try:
        if anim is not None:
            status = anim.animate(
                label=label, wait_file=flag,
                min_time=min_time, timeout=timeout,
                header=header, gap=gap,
            )
        else:
            print("... " + label + " ...")
            thread.join(timeout if timeout > 0 else None)
            status = result.get("status", "TIMEOUT")
    except KeyboardInterrupt:
        print("\nCancelled.")
        return ("FAIL", "cancelled by user")
    thread.join(timeout=10)
    try:
        os.remove(flag)
    except OSError:
        pass
    status = result.get("status", status)
    return (status, result.get("detail", ""))


class _CliReporter:
    """Draws the update steps in a terminal, the way install.bat does.

    Holds the only animation the update path has ever used. The TUI gets its
    own reporter over the same `core.updater.run_update`, which is what keeps
    `cmdai code update` and `/update` behaving identically.
    """

    def step(self, label, work_fn, min_time=1.0, timeout=300.0, header="", gap=0):
        return _animated_step(label, work_fn, min_time=min_time, timeout=timeout,
                              header=header, gap=gap)

    def message(self, text: str) -> None:
        print(text)

    def finish(self, text: str) -> None:
        try:
            _load_animator().print_footer(text)
        except Exception:
            print(text)


def handle_update():
    """Updates CMDAI CODE from GitHub repository with CC branding and animated progress bar."""
    from cmdai.core.updater import run_cli_update

    return run_cli_update(APP_ROOT)


def handle_mcp(argv: List[str]):
    """`cmdai mcp <subcommand>` - manage MCP servers from the command line.

    Servers are added here or by editing mcp_config.json directly. The model
    has no way to add one: a model that can register its own tool endpoints
    is a model that can be talked into loading something the user never
    reviewed. Adding stays a deliberate human action.
    """
    from cmdai.mcp.client import MCPClient, sanitize_name

    client = MCPClient()
    path = client.config_file(APP_ROOT)
    sub = (argv[0].lower() if argv else "list")
    args = argv[1:]

    if sub in ("list", "ls", "status"):
        if not os.path.isfile(path):
            print(f"[i] No MCP config yet at {path}")
            print("    Add one with:  cmdai mcp add <name> <command> [args...]")
            print("               or:  cmdai mcp add <name> --url http://127.0.0.1:8765/mcp")
            return 0
        print(f"\n[*] {path}\n")
        cfg = client.load_config(APP_ROOT)
        if client.load_error:
            # A file that exists but will not parse must not look the same as
            # an empty one, or the user goes looking for servers that are
            # sitting right there in the file.
            print(f"[!] Could not read the config: {client.load_error}")
            print("    Common cause: a byte-order mark from PowerShell")
            print("    `Set-Content -Encoding UTF8`. Rewrite it without:")
            print('      python -c "import json,io;d=json.load(io.open(r\'')
            print(f"{path}'" + r''',encoding='utf-8-sig'));io.open(r''')
            print(f"{path}'" + r''','w',encoding='utf-8').write(json.dumps(d,indent=2))"''')
            return 1
        if not cfg:
            print("[i] Config is empty.")
            return 0
        print(f"[*] {len(cfg)} server(s) configured\n")
        for name in sorted(cfg):
            entry = cfg[name] or {}
            kind = "http" if entry.get("url") else "stdio"
            if entry.get("type"):
                kind = entry["type"]
            flag = "" if entry.get("enabled", True) else "  [disabled]"
            if kind == "stdio":
                target = " ".join([entry.get("command", "")] + list(entry.get("args") or []))
            else:
                target = entry.get("url", "")
            auto = "" if entry.get("autostart", True) else "  [no autostart]"
            print(f"  {name}")
            print(f"      {kind}: {target}{flag}{auto}")
        print("\n[i] Connect state is reported by the app; run /mcp inside CMDAI CODE.")
        return 0

    if sub == "add":
        # Split out only the flags this command owns. Stripping every token
        # starting with "-" would eat the server's own options, so
        # `cmdai mcp add s python -c "print(1)"` would silently lose "-c".
        raw = list(args)
        url = ""
        env: Dict[str, str] = {}
        autostart = True
        enabled = True
        positional: List[str] = []

        index = 0
        while index < len(raw):
            token = raw[index]
            if token == "--url":
                if index + 1 >= len(raw):
                    print("[!] --url needs a value.")
                    return 2
                url = raw[index + 1]
                index += 2
                continue
            if token == "--env":
                if index + 1 >= len(raw) or "=" not in raw[index + 1]:
                    print("[!] --env needs KEY=value")
                    return 2
                key, _, value = raw[index + 1].partition("=")
                env[key] = value
                index += 2
                continue
            if token == "--no-autostart":
                autostart = False
                index += 1
                continue
            if token == "--disabled":
                enabled = False
                index += 1
                continue
            positional.append(token)
            index += 1

        if not positional:
            print("[!] Usage: cmdai mcp add <name> <command> [args...]")
            print("    or:     cmdai mcp add <name> --url <url>")
            return 2

        name = sanitize_name(positional[0])
        if name == "server":
            print(f"[!] '{positional[0]}' is not a usable server name.")
            return 2
        cfg = client.load_config(APP_ROOT)
        if name in cfg:
            print(f"[!] '{name}' already exists. Remove it first: cmdai mcp remove {name}")
            return 2

        rest = positional[1:]
        entry: Dict[str, Any] = {}
        if url:
            if rest:
                print("[!] Give either --url or a command, not both.")
                return 2
            entry = {"type": "http", "url": url}
        elif rest:
            entry = {"command": rest[0], "args": rest[1:]}
        else:
            print("[!] Give either a command to run or --url <url>.")
            print("    Usage: cmdai mcp add <name> <command> [args...]")
            print("           cmdai mcp add <name> --url <url>")
            return 2

        if env:
            entry["env"] = env
        if not autostart:
            entry["autostart"] = False
        if not enabled:
            entry["enabled"] = False

        cfg[name] = entry
        if not client.save_config(cfg, APP_ROOT):
            print(f"[!] Could not write {path}")
            return 1
        print(f"[OK] Added '{name}' to {path}")
        if name != positional[0]:
            print(f"     stored as '{name}' (tool names use underscores)")
        if env:
            print("     Values written as ${VAR} are expanded at launch, so tokens can")
            print("     stay in your environment instead of this file.")
        return 0

    if sub in ("remove", "rm", "delete"):
        if not args:
            print("[!] Usage: cmdai mcp remove <name>")
            return 2
        name = sanitize_name(args[0])
        cfg = client.load_config(APP_ROOT)
        if name not in cfg:
            known = ", ".join(sorted(cfg)) or "(none)"
            print(f"[!] No server named '{name}'. Configured: {known}")
            return 2
        cfg.pop(name)
        if not client.save_config(cfg, APP_ROOT):
            print(f"[!] Could not write {path}")
            return 1
        print(f"[OK] Removed '{name}'")
        return 0

    if sub in ("enable", "disable"):
        if not args:
            print(f"[!] Usage: cmdai mcp {sub} <name>")
            return 2
        name = sanitize_name(args[0])
        cfg = client.load_config(APP_ROOT)
        if name not in cfg:
            known = ", ".join(sorted(cfg)) or "(none)"
            print(f"[!] No server named '{name}'. Configured: {known}")
            return 2
        entry = cfg[name] or {}
        entry["enabled"] = (sub == "enable")
        cfg[name] = entry
        if not client.save_config(cfg, APP_ROOT):
            print(f"[!] Could not write {path}")
            return 1
        print(f"[OK] '{name}' is now {sub}d")
        return 0

    print(f"[!] Unknown: cmdai mcp {sub}")
    print("    Subcommands: list, add, remove, enable, disable")
    return 2


def handle_add_local_model():
    """Opens Windows File Explorer dialog to pick a local model (*.gguf) and imports it into models/."""
    import shutil
    import json

    print("\n" + "=" * 65)
    print("  CMDAI CODE - Add Local Model (GGUF)")
    print("=" * 65 + "\n")
    print("[*] Opening Windows File Explorer to select a model file...")

    selected_file = ""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected_file = filedialog.askopenfilename(
            title="Select Local Model File for CMDAI CODE",
            filetypes=[
                ("GGUF Model Files (*.gguf)", "*.gguf"),
                ("Binary Model Files (*.bin)", "*.bin"),
                ("All Files (*.*)", "*.*"),
            ],
        )
        root.destroy()
    except Exception:
        try:
            import subprocess
            ps_cmd = (
                "[System.Reflection.Assembly]::LoadWithPartialName('System.windows.forms') | Out-Null; "
                "$f = New-Object System.Windows.Forms.OpenFileDialog; "
                "$f.Filter = 'GGUF Model Files (*.gguf)|*.gguf|All Files (*.*)|*.*'; "
                "$f.Title = 'Select Local Model File (GGUF)'; "
                "$res = $f.ShowDialog(); "
                "if ($res -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $f.FileName }"
            )
            out = subprocess.check_output(["powershell", "-NoProfile", "-Command", ps_cmd], text=True)
            selected_file = out.strip()
        except Exception as pe:
            print(f"[!] Failed to open File Explorer dialog: {pe}")
            return 1

    if not selected_file or not os.path.exists(selected_file):
        print("[!] Model file selection cancelled.")
        return 0

    model_filename = os.path.basename(selected_file)
    from cmdai.core.model_paths import resolve_models_dir
    # Was hardcoded to APP_ROOT/models, ignoring both CMDAI_MODELS_DIR and
    # config["models_dir"] - so a user who configured a different directory
    # had `cmdai addlocal` drop the model somewhere the loader would not look.
    models_dir = resolve_models_dir("")
    os.makedirs(models_dir, exist_ok=True)
    dest_path = os.path.join(models_dir, model_filename)

    file_size_mb = os.path.getsize(selected_file) / (1024 * 1024)
    file_size_gb = file_size_mb / 1024

    print(f"\n[+] Selected model: {model_filename}")
    if file_size_gb >= 1.0:
        print(f"    Size: {file_size_gb:.2f} GB")
    else:
        print(f"    Size: {file_size_mb:.1f} MB")
    print(f"    Target path: models/{model_filename}")

    if os.path.abspath(selected_file) == os.path.abspath(dest_path):
        print("\n[i] File is already located in the models/ directory!")
    else:
        print("[*] Copying model file to models/ directory (please wait)...")
        try:
            shutil.copy2(selected_file, dest_path)
            print("[OK] Copying completed successfully!")
        except Exception as ce:
            print(f"[!] Copy error: {ce}")
            return 1

    cfg_file = os.path.join(APP_ROOT, "config.json")
    if not os.path.exists(cfg_file) and os.path.exists(os.path.join(APP_ROOT, "config.example.json")):
        shutil.copy2(os.path.join(APP_ROOT, "config.example.json"), cfg_file)

    if os.path.exists(cfg_file):
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["default_provider"] = "local_gguf"
            cfg["default_model"] = model_filename
            cfg.setdefault("settings", {})["default_provider"] = "local_gguf"
            with open(cfg_file, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
            print(f"[OK] Updated config.json - model '{model_filename}' set as default local model!")
        except Exception as e:
            print(f"[!] Warning updating config: {e}")

    print("\n[OK] Success: Model is ready for use in CMDAI CODE.")
    print("    Launch the assistant by running: cmdai.bat\n")
    return 0


CLI_HELP = r"""
  [bold]Usage[/]
    cmdai                          launch the Pitch Black TUI (default)
    cmdai run <model> \[prompt]     chat REPL or single-shot prompt (Ollama style)
    cmdai pull <repo|query> \[q]    download a .gguf model from Hugging Face
    cmdai list | ls                list installed local GGUF models
    cmdai ps                       show active loaded model & server status
    cmdai rm <model>               remove an installed local GGUF model
    cmdai serve \[--port 11434]     start Ollama + OpenAI-compatible HTTP server
    cmdai code \[task...]           open CMDAI-CODE or run a coding task
    cmdai addlocal                 import a local .gguf file via file dialog
    cmdai mcp <subcommand>         manage MCP servers
    cmdai update                   pull latest CMDAI updates from GitHub
    cmdai help | -h | --help       show this help

  [bold]Options[/]
    --workdir <path>     working directory (default: current)
    --provider <id>      model provider (default: local_gguf)
    --model <id>         model filename or provider model ID
    --port <int>         HTTP server port for `cmdai serve` (default: 11434)
    --host <addr>        HTTP server bind address (default: 127.0.0.1)
    --quant <tag>        quantization hint for `cmdai pull` (e.g. Q4_K_M)
    --install-launcher   install the global `cmdai` PATH shortcut

  [bold]In-app Slash Commands[/]
    /installmodels       browse Hugging Face GGUF hub with live RAM fit
    /server              inspect/toggle Ollama & OpenAI API server (/api/* & /v1/*)
    /code                CMDAI-CODE controller bridge modal
    /models              switch local GGUF or API providers
    /mode                auto / chat / ask / plan
    /unload, /ps         unload or inspect active GGUF runtime
    /help                interactive command & shortcut explorer

  [dim]{repo}[/]
"""


def _cli_match_local_model(name: str) -> str:
    """Resolve exact or partial local `.gguf` filename inside CMDAI models/."""
    from .core.engine import SimpleGGUFLoader
    from .core.model_paths import resolve_models_dir

    loader = SimpleGGUFLoader(resolve_models_dir())
    models = loader.list_models()
    low = (name or "").strip().lower()
    if not low:
        return ""
    for m in models:
        if m["id"].lower() == low or m["name"].lower() == low:
            return m["id"]
    for m in models:
        if low in m["id"].lower():
            return m["id"]
    return ""


def handle_list_models() -> int:
    from .core.engine import SimpleGGUFLoader
    from .core.hf_hub import format_bytes, parse_quant_tag
    from .core.model_paths import resolve_models_dir

    mdir = resolve_models_dir()
    loader = SimpleGGUFLoader(mdir)
    models = loader.list_models()
    print(f"Models directory: {mdir}")
    if not models:
        print("No local .gguf models installed yet.")
        print("Install one with:  cmdai pull Qwen/Qwen2.5-7B-Instruct-GGUF")
        return 0

    print(f"{'NAME':<46} {'QUANT':<8} {'SIZE':>10}")
    print("-" * 68)
    for m in models:
        fpath = m["path"]
        try:
            sz = os.path.getsize(fpath)
            sz_str = format_bytes(sz)
        except OSError:
            sz_str = f"{m.get('size_mb', 0):.1f} MB"
        quant = parse_quant_tag(m["id"])
        print(f"{m['id'][:45]:<46} {quant:<8} {sz_str:>10}")
    return 0


def handle_ps(host: str = "", port: int = 0) -> int:
    import requests
    from .core.engine import SimpleGGUFLoader
    from .core.hf_hub import format_bytes
    from .core.model_paths import resolve_models_dir

    settings = get_settings()
    srv_cfg = settings.config.get("server", {})
    eff_host = host or str(srv_cfg.get("host") or "127.0.0.1")
    eff_port = int(port or srv_cfg.get("port") or 11434)
    base_url = f"http://{eff_host}:{eff_port}"

    try:
        resp = requests.get(f"{base_url}/api/ps", timeout=1.5)
        if resp.ok:
            data = resp.json() or {}
            models = data.get("models") or []
            print(f"CMDAI server: RUNNING at {base_url}")
            if not models:
                print("Loaded models: none (idle)")
            else:
                print(f"{'NAME':<42} {'QUANT':<8} {'CTX':>7} {'SIZE':>10}")
                print("-" * 72)
                for m in models:
                    det = m.get("details") or {}
                    sz = format_bytes(int(m.get("size") or 0))
                    print(
                        f"{str(m.get('name', ''))[:41]:<42} "
                        f"{str(det.get('quantization_level', '-')):<8} "
                        f"{int(m.get('context_length') or 0):>7} "
                        f"{sz:>10}"
                    )
            return 0
    except Exception:
        pass

    local_cnt = len(SimpleGGUFLoader(resolve_models_dir()).list_models())
    print(f"CMDAI server: STOPPED (configured {base_url})")
    print(f"Installed local GGUF models: {local_cnt}")
    return 0


def handle_rm_model(target: str) -> int:
    from .core.engine import SimpleGGUFLoader
    from .core.model_paths import resolve_models_dir

    clean = (target or "").strip()
    if not clean:
        print("Usage: cmdai rm <model_filename>")
        return 1

    mdir = resolve_models_dir()
    models = SimpleGGUFLoader(mdir).list_models()
    low = clean.lower()
    match = next((m for m in models if m["id"].lower() == low), None)
    if match is None:
        partials = [m for m in models if low in m["id"].lower()]
        if len(partials) == 1:
            match = partials[0]
        elif len(partials) > 1:
            print(f"Ambiguous model name {clean!r}. Matches: {', '.join(m['id'] for m in partials)}")
            return 1

    if not match:
        print(f"Model not found in {mdir}: {clean}")
        return 1

    os.remove(match["path"])
    settings = get_settings()
    if settings.config.get("default_model") == match["id"]:
        settings.config["default_model"] = ""
        settings.save_config()
    print(f"Removed: {match['id']}")
    return 0


def handle_pull_model(target: str, quant: str = "") -> int:
    from .core.hf_hub import download_gguf_model, format_bytes, resolve_pull_target

    if not (target or "").strip():
        print("Usage: cmdai pull <owner/repo|query> [quant]")
        return 1

    try:
        print(f"Resolving Hugging Face GGUF for '{target}'...")
        repo_id, entry = resolve_pull_target(target, quant_hint=quant)
        fit = entry.get("ram_fit", {})
        print(
            f"Repo: {repo_id}\n"
            f"File: {entry['filename']} ({entry['quant']}, {entry['size_str']}, RAM fit: {fit.get('label', '?')})"
        )

        def _on_prog(info: Dict[str, Any]) -> None:
            pct = float(info.get("percent") or 0.0)
            done_s = format_bytes(int(info.get("downloaded_bytes") or 0))
            tot_b = int(info.get("total_bytes") or 0)
            tot_s = format_bytes(tot_b) if tot_b > 0 else "?"
            spd = float(info.get("speed_bps") or 0.0) / (1024 * 1024)
            sys.stdout.write(f"\r  Downloading: {pct:5.1f}%  {done_s} / {tot_s}  ({spd:.1f} MB/s)   ")
            sys.stdout.flush()

        dest = download_gguf_model(repo_id, entry["rfilename"], on_progress=_on_prog)
        print(f"\n[OK] Saved to {dest}")
        fname = os.path.basename(dest)
        settings = get_settings()
        if not settings.config.get("default_model"):
            settings.config["default_provider"] = "local_gguf"
            settings.config["default_model"] = fname
            settings.save_config()
            print(f"[OK] Set '{fname}' as default CMDAI model.")
        return 0
    except KeyboardInterrupt:
        print("\nDownload cancelled.")
        return 130
    except Exception as exc:
        print(f"\nPull failed: {exc}", file=sys.stderr)
        return 1


def handle_serve(host: str = "", port: int = 0) -> int:
    import time
    from .core.server import BackgroundHTTPServer

    settings = get_settings()
    srv_cfg = settings.config.get("server", {})
    eff_host = host or str(srv_cfg.get("host") or "127.0.0.1")
    eff_port = int(port or srv_cfg.get("port") or 11434)

    srv = BackgroundHTTPServer(port=eff_port, host=eff_host)
    if not srv.start():
        print(f"Failed to start CMDAI server on {eff_host}:{eff_port}: {srv.last_error}", file=sys.stderr)
        return 1

    print(f"CMDAI HTTP server listening on http://{srv.host}:{srv.port}")
    print(f"  Ollama endpoints : /api/tags, /api/ps, /api/chat, /api/generate, /api/pull, /api/show")
    print(f"  OpenAI endpoints : /v1/models, /v1/chat/completions")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping CMDAI server...")
        srv.stop()
        return 0


def handle_code_command(task_words: List[str], workdir: str) -> int:
    from .core.code_bridge import (
        install_cmdai_code,
        launch_cmdai_code_window,
        resolve_cmdai_code_script,
        run_cmdai_code_headless,
    )

    if not resolve_cmdai_code_script():
        try:
            ans = input("CMDAI-CODE is not installed. Install latest version from GitHub? [y/N]: ").strip().lower()
        except EOFError:
            ans = "n"
        if ans in ("y", "yes"):
            print("Installing latest CMDAI-CODE from GitHub...")
            inst = install_cmdai_code()
            if not inst.get("ok"):
                print(f"Installation failed: {inst.get('error')}", file=sys.stderr)
                return 1
            print(f"Installed CMDAI-CODE at {inst.get('script_path')}")
        else:
            print("CMDAI-CODE is required for `cmdai code`.", file=sys.stderr)
            return 1

    task_str = " ".join(task_words).strip()
    if not task_str:
        res = launch_cmdai_code_window(workdir=workdir)
        if res.get("ok"):
            print(f"Launched CMDAI-CODE terminal (PID {res.get('pid')}) in {res.get('workdir')}")
            return 0
        print(f"Failed to launch CMDAI-CODE: {res.get('error')}", file=sys.stderr)
        return 1

    print(f"Delegating to CMDAI-CODE in {workdir}...\n")
    res = run_cmdai_code_headless(task=task_str, workdir=workdir)
    if res.get("response"):
        print(res["response"])
    if not res.get("ok"):
        print(f"\nError: {res.get('error')}", file=sys.stderr)
        return 1
    return 0


def _run_interactive_repl(model_id: str) -> int:
    from .core.engine import SimpleGGUFLoader
    from .core.model_paths import resolve_models_dir

    loader = SimpleGGUFLoader(resolve_models_dir())
    if not model_id:
        models = loader.list_models()
        if not models:
            print("No local .gguf models found. Run `cmdai pull Qwen/Qwen2.5-7B-Instruct-GGUF` first.")
            return 1
        model_id = models[0]["id"]

    print(f"CMDAI Interactive Chat  ·  Model: {model_id}")
    print("Type '/bye' or 'exit' to quit.\n")
    history: List[Dict[str, str]] = []
    try:
        while True:
            try:
                user_msg = input(">>> ").strip()
            except EOFError:
                break
            if not user_msg:
                continue
            if user_msg.lower() in ("/bye", "/exit", "exit", "quit"):
                break
            history.append({"role": "user", "content": user_msg})
            chunks: List[str] = []

            def _on_tok(tok: str) -> None:
                chunks.append(tok)
                sys.stdout.write(tok)
                sys.stdout.flush()

            text, _ = loader.stream_chat(messages=history, model_filename=model_id, on_token=_on_tok)
            if not chunks and text:
                sys.stdout.write(text)
            print("\n")
            history.append({"role": "assistant", "content": text or "".join(chunks)})
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        loader.close()
    return 0


def _cmdaieditor_main() -> str:
    """Sciezka do `editor/main.py` albo `src/cmdai/editor/main.py`."""
    korzen = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    main_py = os.path.join(korzen, "editor", "main.py")
    if os.path.isfile(main_py):
        return main_py
    pkg_main = os.path.join(os.path.dirname(os.path.abspath(__file__)), "editor", "main.py")
    if os.path.isfile(pkg_main):
        return pkg_main
    return ""


def launch_cmdaieditor(args: list) -> int:
    """`cmdai editor [folder]` - uruchamia edytor CMDAI.

    Zwraca kod wyjścia procesu albo 1, gdy nie da się uruchomić.
    """
    main_py = _cmdaieditor_main()
    if not main_py:
        print("CMDAI EDITOR not found: expected editor/main.py next to src/.",
              file=sys.stderr)
        return 1

    # Ten sam interpreter, ktory uruchomił `cmdai`. Dzięki temu
    # edytor dostaje te same pakiety co aplikacja (a jego
    # `requirements-editor.txt` jest podzbiorem `requirements.txt`).
    polecenie = [sys.executable, main_py, *[str(a) for a in args if str(a)]]
    try:
        import subprocess

        # Bez `shell=True` - argumenty zostaja argumentami, a nazwa
        # folderu ze spacja albo cudzysłowem nie robi sie komenda.
        return subprocess.call(polecenie)
    except KeyboardInterrupt:
        return 130
    except OSError as exc:
        print(f"Cannot start CMDAI EDITOR: {exc}", file=sys.stderr)
        return 1


def print_cli_help() -> None:
    from .core.branding import get_hero_logo, get_plain_logo

    logo = get_hero_logo()
    # The ASCII logo already spells out CMDAI CODE, so only the tagline goes
    # under it - centred against the logo width instead of hardcoded spaces.
    plain_lines = get_plain_logo().splitlines()
    width = max((len(l) for l in plain_lines), default=0)
    tagline = "Local AI Runtime · Server · PC & CMDAI-CODE Controller"
    pad = " " * max(0, (width - len(tagline)) // 2)
    header = f"\n{pad}[bold white]{tagline}[/]\n"
    body = CLI_HELP.format(repo=REPO_URL)
    try:
        from rich.console import Console

        console = Console(highlight=False)
        for line in logo.splitlines():
            console.print(line)
        console.print(header, markup=True, highlight=False)
        console.print(body, markup=True, highlight=False)
    except Exception:
        print(get_plain_logo())
        print("\n" + tagline.center(width))
        print(body)


# Flags that consume the next argument. Everything else starting with "-" is
# a switch and takes no value.
_VALUE_FLAGS = {"--workdir", "--provider", "--model", "--port", "--host", "--quant"}

_HELP_TOKENS = ("help", "-h", "--help", "-?", "/?")


def _bare_words(argv: List[str]) -> List[str]:
    """The positional words in the arguments, in order.

    A subcommand is never reliably at position 0. The global launcher
    (`cmdai.cmd`) and the npm wrapper (`bin/cmdai.js`) both inject
    `--workdir <path>` ahead of whatever the user typed, so `cmdai update`,
    `cmdai code update` and `npx cmdai-code update` all arrive differently. The
    old `raw_args[0] == "update"` only ever matched one of those spellings, and
    the npm path fell through to argparse and tried to launch the TUI instead of
    updating.

    Flag values are skipped, or a directory called `update` would read as the
    update command.
    """
    words: List[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        low = str(arg).lower()
        if low.startswith("-"):
            if "=" not in low and low in _VALUE_FLAGS:
                skip_next = True
            continue
        words.append(low)
    return words


def _editor_target_args(argv: List[str]) -> List[str]:
    """Words meant for the editor: no flags, no flag values, no subcommand.

    `editor/main.py` is a separate application with its own options, so the
    launcher's own `--workdir` must not be forwarded to it.
    """
    return [a for a in _bare_words(argv) if a not in ("editor", "code", "launch")]


def main():
    setup_crash_logger()
    # Windows PL console (cp1250) can't encode box glyphs — force UTF-8 output.
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    raw_args = [a.lower() for a in sys.argv[1:]]
    words = _bare_words(sys.argv[1:])
    first = words[0] if words else ""

    # `help` / `-h` / `--help` print CMDAI help. Handled before
    # argparse so it never falls through to its own "usage: cmdai.py" output.
    if any(a in _HELP_TOKENS for a in raw_args):
        print_cli_help()
        return

    if first == "update":
        sys.exit(handle_update())

    if "addlocal" in words or "add-model" in words:
        sys.exit(handle_add_local_model())

    if "mcp" in words:
        index = sys.argv[1:].index("mcp")
        sys.exit(handle_mcp(sys.argv[index + 2:]))

    parser = argparse.ArgumentParser(prog="cmdai", add_help=False,
                                     description="CMDAI - Local AI Runtime & Controller")
    parser.add_argument("command", nargs="?", default="launch", help="Command")
    parser.add_argument("--workdir", default=os.getcwd(), help="Target project working directory")
    parser.add_argument("--provider", default="", help="LLM Provider ID")
    parser.add_argument("--model", default="", help="Model ID")
    parser.add_argument("--port", type=int, default=0, help="HTTP server port")
    parser.add_argument("--host", default="", help="HTTP server host")
    parser.add_argument("--quant", default="", help="Quantization tag hint")
    parser.add_argument("--install-launcher", action="store_true", help="Install global Windows PATH launcher")
    args, unknown = parser.parse_known_args()

    if args.install_launcher:
        install_global_launcher(silent=False)
        return

    try:
        install_global_launcher(silent=True)
    except Exception:
        pass

    if first in ("list", "ls"):
        sys.exit(handle_list_models())

    if first == "ps":
        sys.exit(handle_ps(host=args.host, port=args.port))

    if first in ("rm", "delete"):
        tail = [w for w in words[1:] if w]
        sys.exit(handle_rm_model(tail[0] if tail else ""))

    if first == "pull":
        pos = []
        skip = False
        for raw_a in sys.argv[1:]:
            if skip:
                skip = False
                continue
            low_a = raw_a.lower()
            if low_a.startswith("-"):
                if "=" not in low_a and low_a in _VALUE_FLAGS:
                    skip = True
                continue
            if not pos and low_a == "pull":
                continue
            pos.append(raw_a)
        tgt = pos[0] if pos else ""
        q_hint = args.quant or (pos[1] if len(pos) > 1 else "")
        sys.exit(handle_pull_model(tgt, quant=q_hint))

    if first == "serve":
        sys.exit(handle_serve(host=args.host, port=args.port))

    if first == "code":
        pos = []
        skip = False
        for raw_a in sys.argv[1:]:
            if skip:
                skip = False
                continue
            low_a = raw_a.lower()
            if low_a.startswith("-"):
                if "=" not in low_a and low_a in _VALUE_FLAGS:
                    skip = True
                continue
            if not pos and low_a == "code":
                continue
            pos.append(raw_a)
        sys.exit(handle_code_command(pos, os.path.abspath(args.workdir)))

    if first == "run" or args.command == "run":
        pos = []
        skip = False
        for raw_a in sys.argv[1:]:
            if skip:
                skip = False
                continue
            low_a = raw_a.lower()
            if low_a.startswith("-"):
                if "=" not in low_a and low_a in _VALUE_FLAGS:
                    skip = True
                continue
            if not pos and low_a == "run":
                continue
            pos.append(raw_a)

        matched_model = _cli_match_local_model(pos[0]) if pos else ""
        if not pos:
            sys.exit(_run_interactive_repl(args.model or str(get_settings().config.get("default_model") or "")))
        if matched_model and len(pos) == 1:
            sys.exit(_run_interactive_repl(matched_model))

        effective_model = args.model or matched_model
        effective_provider = args.provider or ("local_gguf" if matched_model else "")
        prompt_parts = pos[1:] if matched_model else pos
        prompt = " ".join(prompt_parts).strip()
        if not prompt:
            sys.exit(_run_interactive_repl(effective_model))

        from .core.session_runtime import SessionRuntime
        workdir = os.path.abspath(args.workdir)
        print(f"\n[CMDAI HEADLESS RUN] Target workspace: {workdir}")
        print(f"Task: {prompt}\n")
        runtime = SessionRuntime(
            workdir=workdir,
            provider=effective_provider,
            model=effective_model,
        )
        def _cli_token(t):
            sys.stdout.write(t)
            sys.stdout.flush()
        def _cli_tool_start(name, t_args):
            print(f"\n>> Executing tool [{name}]...")
        def _cli_tool_finish(name, t_args, res):
            status = "OK" if res.get("success", True) else "FAILED"
            print(f">> Finished tool [{name}]: {status}\n")

        runtime.run_turn(
            user_prompt=prompt,
            on_token=_cli_token,
            on_tool_start=_cli_tool_start,
            on_tool_finish=_cli_tool_finish,
        )
        print("\n\n[TASK COMPLETED]")
        sys.exit(0)

    if first == "editor":
        sys.exit(launch_cmdaieditor(_editor_target_args(sys.argv[1:])))

    if "editor" in words:
        sys.exit(launch_cmdaieditor(_editor_target_args(sys.argv[1:])))

    settings = get_settings()
    if args.provider:
        settings.config["default_provider"] = args.provider
    if args.model:
        settings.config["default_model"] = args.model

    workdir = os.path.abspath(args.workdir)
    try:
        sys.stdout.write("\x1b]11;#000000\x07")
        sys.stdout.flush()
    except Exception:
        pass

    try:
        app = CMDAICodeTUI(workdir=workdir)
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            sys.stdout.write("\x1b]111\x07")
            sys.stdout.flush()
        except Exception:
            pass


if __name__ == "__main__":
    main()
