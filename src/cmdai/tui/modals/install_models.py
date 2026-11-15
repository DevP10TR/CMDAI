"""LM Studio / Ollama style Hugging Face GGUF browser and downloader modals."""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from ...core.hf_hub import (
    format_bytes,
    get_curated_repos,
    get_system_ram_gb,
    list_repo_gguf_files,
    search_hf_models,
    download_gguf_model,
)


def _render_ascii_bar(percent: float, width: int = 28) -> str:
    pct = max(0.0, min(100.0, float(percent or 0.0)))
    filled = int(round((pct / 100.0) * width))
    empty = max(0, width - filled)
    return f"[#58a6ff]{'█' * filled}[/][#30363d]{'░' * empty}[/]"


class HFRepoFilesModal(ModalScreen[Optional[str]]):
    """Browse GGUF quantizations inside one Hugging Face repo and download one."""

    BINDINGS = [
        Binding("escape", "cancel_or_close", "Back / Cancel"),
    ]

    DEFAULT_CSS = """
    HFRepoFilesModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.72);
    }
    HFRepoFilesModal #modal-dialog {
        width: 96;
        max-width: 94%;
        height: auto;
        max-height: 82%;
    }
    HFRepoFilesModal #hf-subtitle {
        color: #8b949e;
        height: auto;
        margin: 0 0 1 0;
    }
    HFRepoFilesModal #hf-progress-panel {
        height: auto;
        border-top: solid #21262d;
        padding: 1 0 0 0;
        margin: 1 0 0 0;
        display: none;
    }
    HFRepoFilesModal #hf-progress-panel.visible {
        display: block;
    }
    HFRepoFilesModal #hf-progress-text {
        color: #e6edf3;
        height: auto;
    }
    HFRepoFilesModal #hf-progress-bar {
        color: #58a6ff;
        height: 1;
        margin-top: 1;
    }
    """

    def __init__(self, repo_id: str, repo_label: str = ""):
        super().__init__()
        self.repo_id = repo_id
        self.repo_label = repo_label or repo_id
        self.files: List[Dict[str, Any]] = []
        self._downloading = False
        self._cancelled = False

    def compose(self) -> ComposeResult:
        ram_gb = get_system_ram_gb()
        with Vertical(id="modal-dialog"):
            with Horizontal(id="modal-header"):
                yield Static(f"GGUF Files  ·  {self.repo_id}", id="modal-title")
                yield Static("Esc: Back", id="modal-esc")
            yield Static(
                f"System RAM: [#7ee787]{ram_gb:.1f} GB[/]  ·  Loading repository files...",
                id="hf-subtitle",
                markup=True,
            )
            yield OptionList(id="modal-list")
            with Vertical(id="hf-progress-panel"):
                yield Static("", id="hf-progress-text", markup=True)
                yield Static("", id="hf-progress-bar", markup=True)
            with Horizontal(id="modal-footer"):
                yield Static("Enter: Download .gguf to CMDAI models/  ·  Esc: Cancel", id="modal-footer-right")

    def on_mount(self) -> None:
        threading.Thread(target=self._load_files_worker, daemon=True).start()

    def _load_files_worker(self) -> None:
        try:
            files = list_repo_gguf_files(self.repo_id)
            self.app.call_from_thread(self._populate_files, files, None)
        except Exception as exc:
            self.app.call_from_thread(self._populate_files, [], str(exc))

    def _populate_files(self, files: List[Dict[str, Any]], error: Optional[str]) -> None:
        subtitle = self.query_one("#hf-subtitle", Static)
        ol = self.query_one("#modal-list", OptionList)
        ol.clear_options()

        ram_gb = get_system_ram_gb()
        if error:
            subtitle.update(f"[#f85149]Failed to load files: {error}[/]")
            return

        self.files = files
        if not files:
            subtitle.update(f"System RAM: {ram_gb:.1f} GB  ·  [#d29922]No .gguf files found in this repo.[/]")
            return

        subtitle.update(
            f"System RAM: [#7ee787]{ram_gb:.1f} GB[/]  ·  {len(files)} GGUF file(s) available"
        )
        for entry in files:
            quant = entry.get("quant", "GGUF")
            fname = entry.get("filename", "")
            sz = entry.get("size_str", "?")
            fit = entry.get("ram_fit", {})
            fit_label = fit.get("label", "Unknown")
            fit_color = fit.get("color", "#8b949e")
            q_color = "#d29922" if entry.get("is_mmproj") else "#58a6ff"
            label = (
                f"[b {q_color}]{quant:<7}[/]  {fname}   "
                f"[dim]{sz:>9}[/]   [{fit_color}]{fit_label}[/]"
            )
            ol.add_option(Option(label, id=entry["rfilename"]))
        ol.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if self._downloading:
            return
        selected_rfile = str(event.option.id or "")
        match = next((f for f in self.files if f.get("rfilename") == selected_rfile), None)
        if not match:
            return
        self._start_download(match)

    def _start_download(self, entry: Dict[str, Any]) -> None:
        self._downloading = True
        self._cancelled = False
        panel = self.query_one("#hf-progress-panel", Vertical)
        panel.add_class("visible")
        self.query_one("#hf-progress-text", Static).update(
            f"Connecting to Hugging Face for [b #e6edf3]{entry['filename']}[/]..."
        )
        self.query_one("#hf-progress-bar", Static).update(_render_ascii_bar(0.0))
        self.query_one("#modal-esc", Static).update("Esc: Cancel download")
        threading.Thread(target=self._download_worker, args=(entry,), daemon=True).start()

    def _download_worker(self, entry: Dict[str, Any]) -> None:
        def _on_prog(info: Dict[str, Any]) -> None:
            self.app.call_from_thread(self._update_progress_ui, info)

        try:
            dest_path = download_gguf_model(
                repo_id=self.repo_id,
                rfilename=entry["rfilename"],
                on_progress=_on_prog,
                is_cancelled=lambda: self._cancelled,
            )
            self.app.call_from_thread(self._finish_download, dest_path, None)
        except InterruptedError:
            self.app.call_from_thread(self._finish_download, None, "Download cancelled.")
        except Exception as exc:
            self.app.call_from_thread(self._finish_download, None, str(exc))

    def _update_progress_ui(self, info: Dict[str, Any]) -> None:
        pct = float(info.get("percent") or 0.0)
        done_s = format_bytes(int(info.get("downloaded_bytes") or 0))
        tot_b = int(info.get("total_bytes") or 0)
        tot_s = format_bytes(tot_b) if tot_b > 0 else "?"
        spd_b = float(info.get("speed_bps") or 0.0)
        spd_s = f"{spd_b / (1024 * 1024):.1f} MB/s" if spd_b > 0 else "-- MB/s"
        eta = int(info.get("eta_sec") or 0)
        eta_s = f"{eta // 60}m {eta % 60:02d}s" if eta >= 60 else (f"{eta}s" if eta > 0 else "--")
        self.query_one("#hf-progress-text", Static).update(
            f"Downloading [b #e6edf3]{info.get('filename', '')}[/]  ·  "
            f"{done_s} / {tot_s}  ·  [#7ee787]{spd_s}[/]  ·  ETA {eta_s}"
        )
        self.query_one("#hf-progress-bar", Static).update(
            f"{_render_ascii_bar(pct)}  [b #e6edf3]{pct:5.1f}%[/]"
        )

    def _finish_download(self, dest_path: Optional[str], error: Optional[str]) -> None:
        self._downloading = False
        if dest_path:
            self.app.notify(
                f"Downloaded {os.path.basename(dest_path)} to CMDAI models.",
                title="Model Installed",
                timeout=4.0,
            )
            self.dismiss(dest_path)
            return

        panel = self.query_one("#hf-progress-panel", Vertical)
        panel.add_class("visible")
        self.query_one("#hf-progress-text", Static).update(f"[#f85149]{error}[/]")
        self.query_one("#modal-esc", Static).update("Esc: Back")

    def action_cancel_or_close(self) -> None:
        if self._downloading:
            self._cancelled = True
            return
        self.dismiss(None)


class InstallModelsModal(ModalScreen[Optional[str]]):
    """LM Studio-style Hugging Face GGUF browser and local importer modal."""

    BINDINGS = [
        Binding("escape", "close_modal", "Close"),
        Binding("ctrl+f", "import_local", "Import Local .gguf"),
    ]

    DEFAULT_CSS = """
    InstallModelsModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.72);
    }
    InstallModelsModal #modal-dialog {
        width: 96;
        max-width: 94%;
        height: auto;
        max-height: 82%;
    }
    InstallModelsModal #install-status {
        color: #8b949e;
        height: auto;
        margin: 0 0 1 0;
    }
    """

    def __init__(self, initial_query: str = ""):
        super().__init__()
        self.initial_query = initial_query.strip()
        self._repos: List[Dict[str, Any]] = []
        self._search_timer = None
        self._search_seq = 0

    def compose(self) -> ComposeResult:
        ram_gb = get_system_ram_gb()
        with Vertical(id="modal-dialog"):
            with Horizontal(id="modal-header"):
                yield Static("Install Models  ·  Hugging Face GGUF Hub", id="modal-title")
                yield Static("Esc", id="modal-esc")
            yield Input(
                value=self.initial_query,
                placeholder="Search Hugging Face GGUF models (e.g. qwen 7b, llama 3.2, gemma, deepseek)...",
                id="modal-search",
            )
            yield Static(
                f"System RAM: [#7ee787]{ram_gb:.1f} GB[/]  ·  Popular GGUF models ready to download",
                id="install-status",
                markup=True,
            )
            yield OptionList(id="modal-list")
            with Horizontal(id="modal-footer"):
                yield Static("Enter: Browse quants  ·  Ctrl+F: Import local .gguf  ·  Esc: Close", id="modal-footer-right")

    def on_mount(self) -> None:
        search_input = self.query_one("#modal-search", Input)
        search_input.focus()
        if self.initial_query:
            self._trigger_remote_search(self.initial_query)
        else:
            self._render_repos(get_curated_repos(), source_note="Popular GGUF repositories")

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "modal-search":
            return
        query = event.value.strip()
        if self._search_timer is not None:
            try:
                self._search_timer.stop()
            except Exception:
                pass
            self._search_timer = None

        if not query:
            self._search_seq += 1
            self._render_repos(get_curated_repos(), source_note="Popular GGUF repositories")
            return

        # Show instant curated subset immediately while debouncing remote HF call.
        local_matches = get_curated_repos(query)
        if local_matches:
            self._render_repos(local_matches, source_note=f"Curated matches + searching Hugging Face for '{query}'...")
        else:
            self.query_one("#install-status", Static).update(
                f"Searching Hugging Face for [b #58a6ff]{query}[/]..."
            )
        self._search_timer = self.set_timer(0.35, lambda q=query: self._trigger_remote_search(q))

    def _trigger_remote_search(self, query: str) -> None:
        self._search_seq += 1
        seq = self._search_seq
        threading.Thread(
            target=self._search_worker,
            args=(query, seq),
            daemon=True,
        ).start()

    def _search_worker(self, query: str, seq: int) -> None:
        try:
            remote = search_hf_models(query=query, limit=25)
            curated = get_curated_repos(query)
            seen = {r["repo_id"].lower() for r in remote}
            merged = list(remote)
            for item in curated:
                if item["repo_id"].lower() not in seen:
                    merged.insert(0, item)
            self.app.call_from_thread(self._apply_search_results, seq, query, merged, None)
        except Exception as exc:
            fallback = get_curated_repos(query)
            self.app.call_from_thread(self._apply_search_results, seq, query, fallback, str(exc))

    def _apply_search_results(
        self,
        seq: int,
        query: str,
        repos: List[Dict[str, Any]],
        error: Optional[str],
    ) -> None:
        if seq != self._search_seq:
            return
        note = (
            f"Showing {len(repos)} GGUF repo(s) for '{query}'"
            if not error
            else f"[#d29922]Hugging Face unreachable ({error}); showing {len(repos)} curated match(es)[/]"
        )
        self._render_repos(repos, source_note=note)

    def _render_repos(self, repos: List[Dict[str, Any]], source_note: str) -> None:
        self._repos = repos
        ram_gb = get_system_ram_gb()
        self.query_one("#install-status", Static).update(
            f"System RAM: [#7ee787]{ram_gb:.1f} GB[/]  ·  {source_note}"
        )
        ol = self.query_one("#modal-list", OptionList)
        ol.clear_options()
        if not repos:
            ol.add_option(Option("[dim]No matching GGUF repositories found.[/]", id="__NONE__", disabled=True))
            return
        for item in repos:
            repo_id = item["repo_id"]
            dl = int(item.get("downloads") or 0)
            desc = item.get("description") or ""
            badge = "[#7ee787]★[/]" if item.get("curated") else "[#58a6ff]HF[/]"
            stat = f"[dim]{dl:,} dl[/]" if dl > 0 else f"[dim]{desc}[/]"
            ol.add_option(Option(f"{badge}  [b #e6edf3]{repo_id}[/]   {stat}", id=repo_id))

    def on_key(self, event) -> None:
        if event.key == "down":
            focused = self.app.focused
            if focused and focused.id == "modal-search":
                ol = self.query_one("#modal-list", OptionList)
                if ol.option_count > 0:
                    ol.focus()
                    event.prevent_default()
                    event.stop()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        repo_id = str(event.option.id or "")
        if not repo_id or repo_id == "__NONE__":
            return
        match = next((r for r in self._repos if r["repo_id"] == repo_id), None)
        label = match.get("label", repo_id) if match else repo_id
        self.app.push_screen(HFRepoFilesModal(repo_id, label), self._on_file_downloaded)

    def _on_file_downloaded(self, dest_path: Optional[str]) -> None:
        if dest_path:
            self.dismiss(dest_path)

    def action_import_local(self) -> None:
        self.dismiss("__IMPORT_LOCAL__")

    def action_close_modal(self) -> None:
        self.dismiss(None)
