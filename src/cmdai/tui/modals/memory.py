"""Modal for /memory: read-only/delete view of global cross-session memory."""

from __future__ import annotations

from rich.markup import escape
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import OptionList, Static
from textual.widgets.option_list import Option

from ...core.memory import (
    clear_memory,
    list_memory_entries,
    remove_memory_entry,
)


class MemoryModal(ModalScreen):
    """Modal displaying global cross-session memory entries (~/.cmdai/memory.md)."""

    BINDINGS = [
        Binding("escape", "dismiss_modal", "Close"),
        Binding("delete", "delete_selected", "Delete"),
        Binding("ctrl+delete", "clear_global", "Clear all"),
    ]

    def __init__(self, workdir: str = ".", **kwargs):
        super().__init__(**kwargs)
        self.workdir = workdir
        self._items: list[tuple[str, int, str]] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-dialog", classes="sessions-dialog"):
            with Horizontal(id="modal-header"):
                yield Static("Global Memory (~/.cmdai/memory.md)", id="modal-title")
                yield Static("[dim]esc[/]", id="modal-esc")

            yield OptionList(id="modal-list")

            with Horizontal(id="modal-footer"):
                yield Static("delete: Delete  ·  ctrl+del: Clear all  ·  esc: Close")

    def on_mount(self) -> None:
        self._reload_list()
        self.query_one("#modal-list", OptionList).focus()

    def _reload_list(self) -> None:
        data = list_memory_entries(self.workdir, scope="global")
        ol = self.query_one("#modal-list", OptionList)
        ol.clear_options()
        self._items = []
        g_list = data.get("global", [])

        if g_list:
            ol.add_option(Option(Text.from_markup("[bold #a371f7]  ── Global Cross-Session Memory (~/.cmdai/memory.md)[/]"), disabled=True))
            for idx, text in enumerate(g_list, 1):
                oid = f"global:{idx}"
                self._items.append(("global", idx, text))
                t_row = Text.from_markup(f"   [#a371f7]#{idx}[/]  [white]{escape(text)}[/]")
                t_row.no_wrap = True
                t_row.overflow = "ellipsis"
                ol.add_option(Option(t_row, id=oid))
        else:
            ol.add_option(
                Option(
                    Text.from_markup("  [dim]No global memories recorded yet.[/]"),
                    disabled=True,
                )
            )

    def action_dismiss_modal(self) -> None:
        self.dismiss(None)

    def action_delete_selected(self) -> None:
        ol = self.query_one("#modal-list", OptionList)
        if ol.highlighted is None or not self._items:
            return
        try:
            opt = ol.get_option_at_index(ol.highlighted)
            oid = str(getattr(opt, "id", "") or "")
            if ":" not in oid:
                return
            scope, idx_str = oid.split(":", 1)
            res = remove_memory_entry(self.workdir, index=int(idx_str), scope=scope)
            if res.get("success"):
                self._reload_list()
        except Exception:
            pass

    def action_clear_global(self) -> None:
        clear_memory(self.workdir, scope="global")
        self._reload_list()
