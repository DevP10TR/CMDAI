import os
import re
from typing import Any, Callable, Dict, List, Optional

import requests


class AgentContext:
    def __init__(self, workdir: str = "."):
        self._workdir = os.path.abspath(workdir)
        self.current_cwd = self._workdir
        self.session_id: str = ""
        self.todo_list: List[Dict[str, Any]] = []
        self.mode: str = "auto"
        self.pending_images: List[Dict[str, str]] = []
        self.on_before_file_modify: Optional[Callable[[str], None]] = None

    @property
    def workdir(self) -> str:
        return self._workdir

    @workdir.setter
    def workdir(self, value: str) -> None:
        new_w = os.path.abspath(value or ".")
        self._workdir = new_w
        self.current_cwd = new_w

    def resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        base = getattr(self, "current_cwd", None) or self.workdir
        cand = os.path.normpath(os.path.join(base, path))
        if os.path.exists(cand):
            return cand
        return os.path.normpath(os.path.join(self.workdir, path))

    def rel_path(self, path: str) -> str:
        try:
            return os.path.relpath(path, self.workdir)
        except Exception:
            return path


def tool_web(ctx: AgentContext, url_or_query: str) -> Dict[str, Any]:
    target = url_or_query.strip()
    if target.startswith("http://") or target.startswith("https://"):
        try:
            resp = requests.get(
                target,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                timeout=3.5,
            )
            html = resp.text
            html = re.sub(r"<script.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
            html = re.sub(r"<style.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            return {"success": True, "url": target, "content": text[:4000]}
        except Exception as e:
            return {"success": False, "url": target, "error": str(e)}
    else:
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

            with DDGS(timeout=4) as ddgs:
                results = list(ddgs.text(target, max_results=3))

            if not results:
                return {
                    "success": True,
                    "query": target,
                    "content": f'No web search results found for: "{target}"',
                    "results": [],
                }

            formatted_results = []
            output_parts = [f'Web search results for: "{target}":\n']
            for i, res in enumerate(results, 1):
                title = res.get("title", "No title").strip()
                href = res.get("href", "").strip()
                raw_body = res.get("body") or ""
                clean_body = re.sub(r"\s+", " ", raw_body).strip()
                if len(clean_body) > 220:
                    truncated = clean_body[:220].rsplit(" ", 1)[0].rstrip(".,;:-") + "..."
                else:
                    truncated = clean_body
                formatted_results.append({"title": title, "url": href, "snippet": truncated})
                output_parts.append(f"{i}. {title}\n   Link: {href}\n   Snippet: {truncated}\n")

            return {
                "success": True,
                "query": target,
                "content": "\n".join(output_parts),
                "results": formatted_results,
            }
        except Exception as e:
            return {"success": False, "query": target, "error": f"Web search timeout or error: {str(e)}"}


def _split_todo_items(raw: str) -> List[str]:
    if not raw or not str(raw).strip():
        return []
    raw_str = str(raw).strip()
    if "|" in raw_str or "\n" in raw_str:
        chunks = re.split(r"[|\r\n]+", raw_str)
    else:
        chunks = re.split(r"(?:\r?\n|\s+)(?=\d+[\.\)]\s+)", raw_str)
    items: List[str] = []
    for chunk in chunks:
        c = chunk.strip().rstrip(".;")
        c = re.sub(r"^(?:[-*•]+|\d+[\.\)])\s*", "", c)
        c = re.sub(
            r"^(?:plan\s*(?:prac)?|todo|zadania|step)\s*:\s*",
            "",
            c,
            flags=re.IGNORECASE,
        ).strip()
        if c:
            items.append(c)
    return items


def tool_scratch(
    ctx: AgentContext,
    action: str,
    note: str = "",
    index: Optional[int] = None,
    new_text: str = "",
    todos: Optional[str] = None,
) -> Dict[str, Any]:
    action_lower = (action or "add").lower().strip()
    todos_str = str(todos or "").strip()
    note_str = str(note or "").strip()
    new_text_str = str(new_text or "").strip()

    if todos_str and (not note_str or "|" in todos_str or "\n" in todos_str) and action_lower in ("add", "push"):
        if not ctx.todo_list:
            action_lower = "set"

    if action_lower in ("add", "push"):
        raw_input = note_str or todos_str or new_text_str
        for it in _split_todo_items(raw_input):
            ctx.todo_list.append({"text": it, "done": False})
    elif action_lower in ("set", "init"):
        raw_todos = todos_str or note_str or new_text_str
        items = _split_todo_items(raw_todos)
        if items:
            ctx.todo_list = [{"text": it, "done": False} for it in items]
    elif action_lower in ("edit", "update"):
        target_idx = index if index is not None else 0
        if 0 <= target_idx < len(ctx.todo_list):
            ctx.todo_list[target_idx]["text"] = new_text_str or note_str
        elif note_str and new_text_str:
            for item in ctx.todo_list:
                if note_str.lower() in item["text"].lower():
                    item["text"] = new_text_str
                    break
    elif action_lower in ("done", "complete"):
        if index is not None and len(ctx.todo_list) > 0:
            if index == 0:
                ctx.todo_list[0]["done"] = True
            elif 1 <= index <= len(ctx.todo_list):
                if index == len(ctx.todo_list) or not ctx.todo_list[index - 1].get("done", False):
                    ctx.todo_list[index - 1]["done"] = True
                else:
                    ctx.todo_list[index]["done"] = True
        elif note_str or todos_str:
            raw_done = note_str or todos_str
            if raw_done.strip().lower() in ("all", "wszystkie", "wszystko", "*", "done", "complete"):
                for item in ctx.todo_list:
                    item["done"] = True
            else:
                low_done = raw_done.lower()
                matched_any = False
                for item in ctx.todo_list:
                    it_low = str(item.get("text", "")).lower()
                    if it_low and (low_done in it_low or it_low in low_done):
                        item["done"] = True
                        matched_any = True
                if not matched_any:
                    if not ctx.todo_list:
                        for it in _split_todo_items(raw_done):
                            ctx.todo_list.append({"text": it, "done": True})
                    else:
                        for item in ctx.todo_list:
                            if not item.get("done"):
                                item["done"] = True
                                break
        elif ctx.todo_list:
            marked = False
            for item in ctx.todo_list:
                if not item.get("done"):
                    item["done"] = True
                    marked = True
                    break
            if not marked:
                ctx.todo_list[-1]["done"] = True
    elif action_lower in ("remove", "delete"):
        target_idx = index if index is not None else 0
        if 0 <= target_idx < len(ctx.todo_list):
            ctx.todo_list.pop(target_idx)
    elif action_lower in ("clear", "reset"):
        ctx.todo_list.clear()

    return {
        "success": True,
        "action": action,
        "visible_steps": list(ctx.todo_list),
        "total_remaining": len([t for t in ctx.todo_list if not t.get("done")]),
    }


tool_todo = tool_scratch


SCREENSHOT_MAX_BYTES = 2_000_000
SCREENSHOT_MAX_DIM = 1920


def _screenshot_dir() -> str:
    import tempfile

    d = os.path.join(tempfile.gettempdir(), "cmdai_screenshots")
    os.makedirs(d, exist_ok=True)
    return d


def _find_window_rect(title_sub: str = "") -> Optional[Dict[str, int]]:
    if os.name != "nt":
        return None
    try:
        import win32gui
    except ImportError:
        return None
    needle = (title_sub or "").strip().lower()
    found: Dict[str, int] = {}

    def _cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        try:
            text = win32gui.GetWindowText(hwnd) or ""
        except Exception:
            return
        if needle and needle not in text.lower():
            return
        try:
            l, t, r, b = win32gui.GetWindowRect(hwnd)
        except Exception:
            return
        if r - l > 50 and b - t > 50 and not found:
            found.update({"left": l, "top": t, "width": r - l, "height": b - t})

    try:
        if not needle:
            import win32gui as _wg

            hwnd = _wg.GetForegroundWindow()
            l, t, r, b = _wg.GetWindowRect(hwnd)
            if r - l > 50 and b - t > 50:
                return {"left": l, "top": t, "width": r - l, "height": b - t}
            return None
        win32gui.EnumWindows(_cb, None)
    except Exception:
        return None
    return found or None


def tool_screenshot(ctx: AgentContext, target: str = "active_window") -> Dict[str, Any]:
    import uuid

    target = (target or "active_window").strip()
    rect = _find_window_rect("" if target.lower() == "active_window" else target)
    try:
        import mss
    except ImportError:
        return {"success": False, "error": "mss package not installed (pip install mss)"}
    try:
        out = os.path.join(_screenshot_dir(), f"{uuid.uuid4().hex}.png")
        with mss.mss() as sct:
            monitor = (
                {"left": rect["left"], "top": rect["top"], "width": rect["width"], "height": rect["height"]}
                if rect
                else sct.monitors[0]
            )
            shot = sct.grab(monitor)
            try:
                from PIL import Image

                img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
                img.thumbnail((SCREENSHOT_MAX_DIM, SCREENSHOT_MAX_DIM))
                img.save(out, "PNG")
            except ImportError:
                import mss.tools

                mss.tools.to_png(shot.rgb, shot.size, output=out)
        size = os.path.getsize(out)
        if size > SCREENSHOT_MAX_BYTES:
            try:
                from PIL import Image

                img = Image.open(out)
                img.thumbnail((1280, 1280))
                img.save(out, "PNG", optimize=True)
                size = os.path.getsize(out)
            except ImportError:
                pass
        res = {"success": True, "path": out, "size_bytes": size, "target": target}
        if "." in target and any(
            target.lower().endswith(ext)
            for ext in (".html", ".htm", ".py", ".js", ".ts", ".css", ".md", ".txt", ".json")
        ):
            res["note"] = (
                f"Captured desktop because '{target}' is a file name, not an OS window title. "
                "To screenshot a web page, open it in a browser window first and capture that window."
            )
        return res
    except Exception as e:
        return {"success": False, "error": f"screenshot failed: {e}"}


def tool_vision(ctx: AgentContext, image_path: str = "", question: str = "") -> Dict[str, Any]:
    p = (image_path or "").strip()
    if re.fullmatch(r"\[?\s*image\s+\d+\s*\]?", p, flags=re.IGNORECASE):
        return {
            "success": True,
            "path": p,
            "question": (question or "").strip(),
            "note": "This attached image is already embedded in your conversation prompt — inspect it directly and answer the user without calling <tool:vision> again.",
        }
    if not p or not os.path.exists(p):
        return {"success": False, "error": f"image not found: {image_path}"}
    if os.path.getsize(p) > SCREENSHOT_MAX_BYTES * 2:
        return {"success": False, "error": "image too large (>4MB)"}
    entry = {"path": os.path.abspath(p), "question": (question or "").strip()}
    ctx.pending_images.append(entry)
    return {
        "success": True,
        "path": entry["path"],
        "question": entry["question"],
        "note": "image queued, it will be attached to the next model request",
    }


def tool_memory(
    ctx: AgentContext,
    action: str = "list",
    content: str = "",
    scope: str = "global",
    index: int = 0,
    query: str = "",
) -> Dict[str, Any]:
    """Read or update global cross-session persistent memory (~/.cmdai/memory.md)."""
    from ..core.memory import (
        add_memory_entry,
        clear_memory,
        list_memory_entries,
        remove_memory_entry,
    )

    act = (action or "list").strip().lower()
    sc = "global"
    if act == "list":
        data = list_memory_entries(ctx.workdir, scope=sc)
        return {"success": True, "action": "list", "entries": data}
    if act == "add":
        return add_memory_entry(ctx.workdir, content=content, scope=sc)
    if act == "remove":
        return remove_memory_entry(ctx.workdir, index=index, query=query, scope=sc)
    if act == "clear":
        return clear_memory(ctx.workdir, scope=sc)
    return {"success": False, "error": f"Unknown memory action: {action}. Use add, list, remove, or clear."}
