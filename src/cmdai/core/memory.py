"""Global cross-session persistent memory (~/.cmdai_app/memory.md)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional


_MAX_ENTRY_CHARS = 500


def get_memory_file_path(
    workdir: str = ".",
    scope: str = "global",
    global_path: Optional[str] = None,
) -> Path:
    """Return the path to the global cross-session memory markdown file."""
    if global_path:
        return Path(global_path)
    env_override = os.environ.get("CMDAI_GLOBAL_MEMORY_PATH", "").strip()
    if env_override:
        return Path(env_override)
    return Path.home() / ".cmdai_app" / "memory.md"


def _read_entries(path: Path) -> List[str]:
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    out: List[str] = []
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("- "):
            s = s[2:].strip()
        elif s.startswith("* "):
            s = s[2:].strip()
        if s:
            out.append(s)
    return out


def _write_entries(path: Path, entries: List[str], header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not entries:
        path.write_text(f"# {header}\n", encoding="utf-8")
        return
    body = "\n".join(f"- {e}" for e in entries)
    path.write_text(f"# {header}\n\n{body}\n", encoding="utf-8")


def list_memory_entries(
    workdir: str = ".",
    scope: str = "global",
    global_path: Optional[str] = None,
) -> Dict[str, List[str]]:
    """Return global cross-session memory entries."""
    return {"global": _read_entries(get_memory_file_path(workdir, "global", global_path))}


def add_memory_entry(
    workdir: str,
    content: str,
    scope: str = "global",
    global_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Append a concise, deduplicated memory entry to global memory."""
    sc = "global"
    cleaned = " ".join(str(content or "").strip().split())
    if not cleaned:
        return {"success": False, "error": "Memory content cannot be empty."}
    if len(cleaned) > _MAX_ENTRY_CHARS:
        cleaned = cleaned[: _MAX_ENTRY_CHARS - 3].rstrip() + "..."

    path = get_memory_file_path(workdir, sc, global_path)
    entries = _read_entries(path)
    low = cleaned.lower()
    if any(e.lower() == low for e in entries):
        return {
            "success": True,
            "scope": sc,
            "path": str(path),
            "entry": cleaned,
            "duplicate": True,
            "count": len(entries),
        }
    entries.append(cleaned)
    _write_entries(path, entries, "Global User Memory")
    return {
        "success": True,
        "scope": sc,
        "path": str(path),
        "entry": cleaned,
        "duplicate": False,
        "count": len(entries),
    }


def remove_memory_entry(
    workdir: str,
    index: int = 0,
    query: str = "",
    scope: str = "global",
    global_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Remove a global memory entry by 1-based `index` or substring `query`."""
    sc = "global"
    path = get_memory_file_path(workdir, sc, global_path)
    entries = _read_entries(path)
    if not entries:
        return {"success": False, "scope": sc, "error": "No memory entries found."}

    removed: List[str] = []
    if index and int(index) >= 1:
        idx0 = int(index) - 1
        if idx0 >= len(entries):
            return {"success": False, "scope": sc, "error": f"Index {index} out of range (1..{len(entries)})."}
        removed.append(entries.pop(idx0))
    elif (query or "").strip():
        q_low = query.strip().lower()
        kept: List[str] = []
        for e in entries:
            if q_low in e.lower():
                removed.append(e)
            else:
                kept.append(e)
        entries = kept
        if not removed:
            return {"success": False, "scope": sc, "error": f"No memory entry matching '{query}'."}
    else:
        return {"success": False, "scope": sc, "error": "Provide either 1-based index or query."}

    _write_entries(path, entries, "Global User Memory")
    return {"success": True, "scope": sc, "path": str(path), "removed": removed, "count": len(entries)}


def clear_memory(
    workdir: str,
    scope: str = "global",
    global_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Clear all entries from global cross-session memory."""
    path = get_memory_file_path(workdir, "global", global_path)
    if path.is_file():
        _write_entries(path, [], "Global User Memory")
    return {"success": True, "cleared": ["global"]}


def format_memory_hint(
    workdir: str = ".",
    global_path: Optional[str] = None,
) -> str:
    """Return a lightweight hint when global cross-session memory exists."""
    data = list_memory_entries(workdir, scope="global", global_path=global_path)
    g_cnt = len(data.get("global", []))
    if g_cnt == 0:
        return ""
    return (
        f"Global cross-session memory is available ({g_cnt} entries). "
        'Call <tool:memory action="list" /> when prior user facts or preferences are needed.'
    )


def format_memory_for_prompt(
    workdir: str = ".",
    max_chars: int = 4000,
    global_path: Optional[str] = None,
) -> str:
    """Format global memory entries for inclusion in the system prompt."""
    data = list_memory_entries(workdir, scope="global", global_path=global_path)
    g_entries = data.get("global", [])
    if not g_entries:
        return ""

    parts: List[str] = ["[GLOBAL USER MEMORY (~/.cmdai_app/memory.md)]"]
    parts.extend(f"- {e}" for e in g_entries)

    text = "\n".join(parts)
    if len(text) > max_chars:
        text = text[: max_chars - 30].rstrip() + "\n... [memory truncated]"
    return text
