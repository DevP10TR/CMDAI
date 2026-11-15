"""Single source of truth for the XML agent tools reference.

Used by the /tools window and kept next to the system prompt so the visible
reference and the model-facing description stay in sync.
"""

from __future__ import annotations

from typing import List, NamedTuple


class ToolDoc(NamedTuple):
    name: str
    summary: str
    syntax: str
    example: str


TOOL_DOCS: List[ToolDoc] = [
    ToolDoc(
        "models",
        "Inspect, load, or unload local GGUF models",
        '<tool:models action="list|ps|load|unload" model="FILENAME.gguf" />',
        '<tool:models action="load" model="qwen2.5-7b-q4_k_m.gguf" />',
    ),
    ToolDoc(
        "cmdai_code",
        "Control live CMDAI-CODE TUI autonomous software engineer (send detailed natural-language tasks, slash commands, switch modes, inspect status, approve/reject diffs)",
        '<tool:cmdai_code action="run|command|set_mode|approve|reject|interrupt|status|open|reset|close" workdir="PATH" mode="auto|plan|code">DETAILED_TASK_OR_SLASH_CMD</tool:cmdai_code>',
        '<tool:cmdai_code action="run" workdir="E:\\Project">Inspect all files and tests in this project, identify why the authentication unit tests are failing, fix the root cause in the source code, and verify the full test suite passes.</tool:cmdai_code>',
    ),
    ToolDoc(
        "web",
        "Search the web or fetch a web page over HTTP GET",
        '<tool:web query="QUERY_OR_URL" />',
        '<tool:web query="llama.cpp GGUF quantization guide" />',
    ),
    ToolDoc(
        "memory",
        "Save, list, remove, or clear global cross-session memory (~/.cmdai/memory.md)",
        '<tool:memory action="add|list|remove|clear" content="FACT" />',
        '<tool:memory action="add" content="User prefers concise Polish replies" />',
    ),
    ToolDoc(
        "ask",
        "Ask the user a clarifying question with optional pipe-separated choices",
        '<tool:ask question="QUESTION" options="Opt1|Opt2" />',
        '<tool:ask question="Terminate notepad.exe?" options="Yes|No" />',
    ),
    ToolDoc(
        "mcp",
        "List configured MCP servers, open a server, or invoke mcp__SERVER__TOOL",
        '<tool:mcp open server="NAME" />',
        '<tool:mcp />',
    ),
    ToolDoc(
        "screenshot",
        "Capture a window or desktop screenshot to PNG (vision models)",
        '<tool:screenshot target="active_window" />',
        '<tool:screenshot target="desktop" />',
    ),
    ToolDoc(
        "vision",
        "Analyze an image or screenshot with the active vision-capable model",
        '<tool:vision image_path="PATH.png" question="WHAT_TO_CHECK" />',
        '<tool:vision image_path="screen.png" question="Read the error dialog" />',
    ),
    ToolDoc(
        "todo",
        "Manage the assistant task checklist (add / done / set)",
        '<tool:todo action="add|done|set" note="STEP" />',
        '<tool:todo action="set" todos="Check RAM|Load model|Verify server" />',
    ),
]

TOOL_DOCS_BY_NAME = {t.name: t for t in TOOL_DOCS}
