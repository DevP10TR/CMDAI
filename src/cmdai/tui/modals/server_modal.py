"""Server status and Ollama/OpenAI API control modal for CMDAI."""

from __future__ import annotations

import subprocess
from typing import Any, Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from ...core.settings import get_settings


def _copy_text_clipboard(text: str) -> bool:
    if not text:
        return False
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Set-Clipboard -Value $input"],
            input=text,
            text=True,
            timeout=3,
            capture_output=True,
        )
        return proc.returncode == 0
    except Exception:
        return False


class ServerModal(ModalScreen[Optional[str]]):
    """Control and inspect the CMDAI Ollama & OpenAI-compatible HTTP server."""

    BINDINGS = [
        Binding("escape", "close_modal", "Close"),
    ]

    DEFAULT_CSS = """
    ServerModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.72);
    }
    ServerModal #modal-dialog {
        width: 84;
        max-width: 92%;
        height: auto;
        max-height: 82%;
    }
    ServerModal #srv-summary {
        color: #c9d1d9;
        height: auto;
        margin: 0 0 1 0;
        padding: 0 1;
        border: solid #21262d;
        background: #0d1117;
    }
    ServerModal #srv-port-input {
        display: none;
    }
    ServerModal #srv-port-input.visible {
        display: block;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-dialog"):
            with Horizontal(id="modal-header"):
                yield Static("CMDAI Local AI Server  ·  Ollama & OpenAI API", id="modal-title")
                yield Static("Esc", id="modal-esc")
            yield Static("", id="srv-summary", markup=True)
            yield Input(placeholder="Enter custom port (e.g. 11434 or 8080) and press Enter...", id="srv-port-input")
            yield OptionList(id="modal-list")
            with Horizontal(id="modal-footer"):
                yield Static("Enter: Execute action  ·  Esc: Close", id="modal-footer-right")

    def on_mount(self) -> None:
        self._refresh_view()
        self.query_one("#modal-list", OptionList).focus()

    def _get_server(self) -> Any:
        return getattr(self.app, "http_server", None)

    def _refresh_view(self) -> None:
        srv = self._get_server()
        summary = self.query_one("#srv-summary", Static)
        ol = self.query_one("#modal-list", OptionList)
        ol.clear_options()

        if srv is None:
            summary.update("[#f85149]HTTP server manager is not initialized.[/]")
            return

        st = srv.get_status()
        running = bool(st["running"])
        dot = "[#7ee787]● Running[/]" if running else "[#8b949e]○ Stopped[/]"
        base_url = st["base_url"]
        loaded = st["loaded_model"] or "[dim]None loaded[/]"
        err_line = f"\n[#f85149]Last error:[/] {st['last_error']}" if st.get("last_error") else ""

        summary.update(
            f"Status: {dot}    URL: [b #58a6ff]{base_url}[/]\n"
            f"Active Model: [b #e6edf3]{loaded}[/]  ({st['available_models']} local GGUF)\n"
            f"Requests Served: [b #e6edf3]{st['requests_served']}[/]\n"
            f"Ollama API: [dim]{base_url}/api/chat, /api/generate, /api/tags, /api/ps, /api/pull[/]\n"
            f"OpenAI API: [dim]{base_url}/v1/chat/completions, /v1/models[/]"
            f"{err_line}"
        )

        ol.add_option(
            Option(
                "[b #d29922]Stop Server[/]" if running else "[b #7ee787]Start Server[/]",
                id="toggle",
            )
        )
        if st["loaded_model"]:
            ol.add_option(Option(f"[#f85149]Unload Active Model[/]  [dim]({st['loaded_model']})[/]", id="unload"))
        ol.add_option(Option("Use Ollama Standard Port  [b #58a6ff]11434[/]", id="port_11434"))
        ol.add_option(Option("Use Alternate Port        [b #58a6ff]8080[/]", id="port_8080"))
        ol.add_option(Option("Set Custom Port...", id="port_custom"))
        ol.add_option(Option(f"Copy OpenAI Base URL      [dim]{base_url}/v1[/]", id="copy_url"))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        srv = self._get_server()
        if srv is None:
            return
        action = str(event.option.id or "")
        if action == "toggle":
            if srv.running:
                srv.stop()
                self.app.notify("CMDAI HTTP server stopped.", timeout=2.5)
            else:
                ok = srv.start()
                if ok:
                    self.app.notify(f"Server listening on http://{srv.host}:{srv.port}", timeout=3.0)
                else:
                    self.app.notify(f"Failed to start server: {srv.last_error}", severity="error", timeout=4.0)
            self._save_server_settings(srv.running, srv.port)
            self._refresh_view()
        elif action == "unload":
            srv.get_loader().unload()
            self.app.notify("Unloaded active GGUF model from memory.", timeout=2.5)
            self._refresh_view()
        elif action in ("port_11434", "port_8080"):
            new_port = 11434 if action == "port_11434" else 8080
            self._switch_port(new_port)
        elif action == "port_custom":
            inp = self.query_one("#srv-port-input", Input)
            inp.add_class("visible")
            inp.value = str(srv.port)
            inp.focus()
        elif action == "copy_url":
            url = f"http://{srv.host}:{srv.port}/v1"
            if _copy_text_clipboard(url):
                self.app.notify(f"Copied: {url}", timeout=2.5)
            else:
                self.app.notify(f"API Base URL: {url}", timeout=4.0)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "srv-port-input":
            return
        raw = event.value.strip()
        event.input.remove_class("visible")
        try:
            new_port = int(raw)
            if not (1 <= new_port <= 65535):
                raise ValueError("Port out of range 1..65535")
            self._switch_port(new_port)
        except ValueError as exc:
            self.app.notify(str(exc), severity="error", timeout=3.0)
        self.query_one("#modal-list", OptionList).focus()

    def _switch_port(self, new_port: int) -> None:
        srv = self._get_server()
        if srv is None:
            return
        was_running = srv.running
        if was_running:
            ok = srv.restart(port=new_port)
            if ok:
                self.app.notify(f"Server restarted on port {new_port}.", timeout=3.0)
            else:
                self.app.notify(f"Port {new_port} bind failed: {srv.last_error}", severity="error", timeout=4.0)
        else:
            srv.port = new_port
            self.app.notify(f"Configured server port: {new_port}.", timeout=2.5)
        self._save_server_settings(srv.running, new_port)
        self._refresh_view()

    def _save_server_settings(self, enabled: bool, port: int) -> None:
        try:
            settings = get_settings()
            settings.config.setdefault("server", {})["enabled"] = bool(enabled)
            settings.config["server"]["port"] = int(port)
            settings.save_config()
        except Exception:
            pass

    def action_close_modal(self) -> None:
        self.dismiss(None)
