"""CMDAI-CODE Controller Bridge modal for delegating coding tasks."""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from ...core.code_bridge import (
    get_cmdai_code_status,
    launch_cmdai_code_window,
    run_cmdai_code_headless,
)
from ...core.settings import get_settings


class CodeBridgeModal(ModalScreen[Optional[Dict[str, Any]]]):
    """Inspect CMDAI-CODE connection and launch or delegate coding tasks."""

    BINDINGS = [
        Binding("escape", "close_modal", "Close"),
    ]

    DEFAULT_CSS = """
    CodeBridgeModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.72);
    }
    CodeBridgeModal #modal-dialog {
        width: 92;
        max-width: 94%;
        height: auto;
        max-height: 85%;
    }
    CodeBridgeModal #cb-status {
        color: #c9d1d9;
        height: auto;
        margin: 0 0 1 0;
        padding: 0 1;
        border: solid #21262d;
    }
    CodeBridgeModal .cb-label {
        color: #6e7681;
        height: 1;
    }
    CodeBridgeModal #cb-output {
        height: auto;
        max-height: 12;
        color: #c9d1d9;
        border-top: solid #21262d;
        padding: 1 0 0 0;
        margin-top: 1;
        display: none;
    }
    CodeBridgeModal #cb-output.visible {
        display: block;
    }
    CodeBridgeModal #cb-path-input {
        display: none;
    }
    CodeBridgeModal #cb-path-input.visible {
        display: block;
    }
    """

    def __init__(self, default_workdir: str = ""):
        super().__init__()
        self.default_workdir = default_workdir or os.getcwd()
        self._busy = False

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-dialog"):
            with Horizontal(id="modal-header"):
                yield Static("CMDAI-CODE Controller Bridge", id="modal-title")
                yield Static("Esc", id="modal-esc")
            yield Static("", id="cb-status", markup=True)
            yield Static("Coding Task (for headless delegation):", classes="cb-label")
            yield Input(placeholder="Describe coding task for CMDAI-CODE...", id="cb-task")
            yield Static("Project Directory (--workdir):", classes="cb-label")
            yield Input(value=self.default_workdir, placeholder="Working directory...", id="cb-workdir")
            yield Input(placeholder="Path to CMDAI CODE cmdai.py...", id="cb-path-input")
            yield OptionList(id="modal-list")
            yield Static("", id="cb-output", markup=True)
            with Horizontal(id="modal-footer"):
                yield Static("Enter: Execute selected action  ·  Esc: Close", id="modal-footer-right")

    def on_mount(self) -> None:
        self._refresh_status()
        self.query_one("#cb-task", Input).focus()

    def _refresh_status(self) -> None:
        st = get_cmdai_code_status()
        status_w = self.query_one("#cb-status", Static)
        ol = self.query_one("#modal-list", OptionList)
        ol.clear_options()

        if st["installed"]:
            recent = st.get("recent_sessions") or []
            recent_txt = ""
            if recent:
                top_titles = ", ".join(s.get("title", "Untitled")[:28] for s in recent[:3])
                recent_txt = f"\nRecent Sessions: [dim]{top_titles}[/]"
            status_w.update(
                f"Status: [#7ee787]● Connected[/]    Path: [b #e6edf3]{st['script_path']}[/]\n"
                f"Python: [dim]{st['python_bin']}[/]{recent_txt}"
            )
        else:
            status_w.update(
                "Status: [#f85149]○ Not Installed[/]\n"
                "[dim]Install the latest CMDAI-CODE from GitHub or set a custom path below.[/]"
            )
            ol.add_option(Option("[b #7ee787]Install Latest CMDAI-CODE from GitHub[/]", id="install_code"))

        ol.add_option(Option("[b #7ee787]Run Task in CMDAI-CODE (Headless JSON)[/]", id="run_headless"))
        ol.add_option(Option("Continue Last CMDAI-CODE Session (--continue)", id="run_continue"))
        ol.add_option(Option("Open CMDAI-CODE Interactive Terminal", id="open_terminal"))
        ol.add_option(Option("Set Custom CMDAI-CODE cmdai.py Path...", id="set_path"))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if self._busy:
            return
        act = str(event.option.id or "")
        task = self.query_one("#cb-task", Input).value.strip()
        workdir = self.query_one("#cb-workdir", Input).value.strip() or self.default_workdir

        if act == "install_code":
            self._start_install()
            return
        if act in ("run_headless", "run_continue", "open_terminal") and not get_cmdai_code_status()["installed"]:
            from .ask import AskModal

            def _on_ask(ans: Optional[str]) -> None:
                if ans and ans.lower().startswith("yes"):
                    self._start_install()

            self.app.push_screen(
                AskModal(
                    "CMDAI-CODE is not installed. Install the latest version from GitHub (DevP10TR/CMDAI-CODE)?",
                    ["Yes, install latest CMDAI-CODE", "No, cancel"],
                ),
                _on_ask,
            )
            return

        if act in ("run_headless", "run_continue"):
            if not task:
                self.app.notify("Enter a coding task description first.", severity="warning", timeout=2.5)
                self.query_one("#cb-task", Input).focus()
                return
            self._start_headless(task, workdir, continue_session=(act == "run_continue"))
        elif act == "open_terminal":
            res = launch_cmdai_code_window(workdir=workdir, initial_task=task)
            if res.get("ok"):
                self.app.notify(f"Opened CMDAI-CODE window (PID {res.get('pid')}).", timeout=3.0)
            else:
                self.app.notify(f"Launch failed: {res.get('error')}", severity="error", timeout=4.0)
        elif act == "set_path":
            p_in = self.query_one("#cb-path-input", Input)
            p_in.add_class("visible")
            p_in.value = str(get_settings().config.get("cmdai_code_path") or "")
            p_in.focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "cb-path-input":
            new_path = event.value.strip()
            event.input.remove_class("visible")
            if new_path:
                settings = get_settings()
                settings.config["cmdai_code_path"] = new_path
                settings.save_config()
                self._refresh_status()
                self.app.notify("Saved cmdai_code_path.", timeout=2.5)
            self.query_one("#modal-list", OptionList).focus()
        elif event.input.id == "cb-task":
            self.query_one("#modal-list", OptionList).focus()

    def _start_headless(self, task: str, workdir: str, continue_session: bool = False) -> None:
        self._busy = True
        out_w = self.query_one("#cb-output", Static)
        out_w.add_class("visible")
        out_w.update("[#58a6ff]Running task inside CMDAI-CODE...[/]")

        def _worker() -> None:
            res = run_cmdai_code_headless(
                task=task,
                workdir=workdir,
                continue_session=continue_session,
            )
            self.app.call_from_thread(self._finish_headless, res)

        threading.Thread(target=_worker, daemon=True).start()

    def _finish_headless(self, res: Dict[str, Any]) -> None:
        self._busy = False
        out_w = self.query_one("#cb-output", Static)
        out_w.add_class("visible")
        if res.get("ok"):
            body = (res.get("response") or "").strip() or "(completed with empty response)"
            preview = body[:1000] + ("..." if len(body) > 1000 else "")
            out_w.update(
                f"[#7ee787]Done[/] [dim](session {res.get('session_id', '-')})[/]\n{preview}"
            )
        else:
            out_w.update(f"[#f85149]Error:[/] {res.get('error') or 'Unknown failure'}")

    def action_close_modal(self) -> None:
        self.dismiss(None)

    def _start_install(self) -> None:
        self._busy = True
        out_w = self.query_one("#cb-output", Static)
        out_w.add_class("visible")
        out_w.update("[#58a6ff]Installing latest CMDAI-CODE from GitHub...[/]")

        def _worker() -> None:
            from ...core.code_bridge import install_cmdai_code

            res = install_cmdai_code()
            self.app.call_from_thread(self._finish_install, res)

        threading.Thread(target=_worker, daemon=True).start()

    def _finish_install(self, res: Dict[str, Any]) -> None:
        self._busy = False
        out_w = self.query_one("#cb-output", Static)
        out_w.add_class("visible")
        if res.get("ok"):
            self._refresh_status()
            out_w.update(f"[#7ee787]Installed CMDAI-CODE[/]\nPath: {res.get('script_path')}")
            self.app.notify("CMDAI-CODE installed successfully.", timeout=3.0)
        else:
            out_w.update(f"[#f85149]Install failed:[/] {res.get('error') or 'Unknown error'}")
