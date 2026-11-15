"""Centralized Tool Specifications and JSON Schemas for CMDAI.

Single source of truth for tool calling:
- Generates JSON Schema tool definitions for native tool-calling APIs (OpenAI, Anthropic, Gemini, OpenRouter)
- Generates XML prompts for local/GGUF models
- Provides unified parsing and conversion between native tool calls and XML representations.
"""

from typing import Any, Dict, List


TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "web",
        "description": "Search the web or retrieve documentation and information from a URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query or URL"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "todo",
        "description": "Manage assistant task checklist: 'set' replaces checklist, 'add' appends, 'done' marks complete.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "done", "set"], "description": "Action type"},
                "note": {"type": "string", "description": "Checklist step description"},
                "todos": {"type": "string", "description": "Pipe-separated steps for 'set' action, e.g. 'step1|step2'"},
            },
            "required": ["action"],
        },
    },
    {
        "name": "memory",
        "description": "Save, list, remove, or clear global cross-session user memory (~/.cmdai/memory.md).",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "list", "remove", "clear"],
                    "description": "Global memory operation to perform.",
                },
                "content": {"type": "string", "description": "Fact or preference to remember across sessions (for 'add')."},
                "index": {"type": "integer", "description": "1-based entry index to remove."},
                "query": {"type": "string", "description": "Substring match to remove."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "cmdai_code",
        "description": (
            "Persistent MCP/subagent-style controller for the autonomous CMDAI-CODE AI software engineer in its live TUI window. "
            "CMDAI-CODE already has its own file, search, edit, and terminal tools. Keeps an ongoing session per directory, "
            "can switch modes, run TUI/verification commands, inspect status, approve/reject diffs, interrupt, or reset/close."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["run", "command", "set_mode", "approve", "reject", "interrupt", "status", "open", "reset", "close"],
                    "description": (
                        "'run' sends a detailed, multi-sentence natural-language engineering task (default), "
                        "'command' ONLY runs CMDAI-CODE TUI slash commands (like '/diff', '/git', '/commit') or explicit test commands (never bare 'dir'/'ls'), "
                        "'set_mode' switches mode ('auto'|'plan'|'code'), 'approve'/'reject' confirms or reverts pending diffs, "
                        "'interrupt' aborts generation, 'status' inspects live state, 'open' opens TUI, 'reset' starts /new, 'close' closes window."
                    ),
                },
                "task": {
                    "type": "string",
                    "description": (
                        "Comprehensive, multi-sentence natural-language engineering prompt specifying the goal, context, files/requirements, "
                        "and expected deliverable (for 'run') — NEVER send bare shell commands like 'dir', 'ls', 'cat', or 'type'; "
                        "or a TUI slash command like '/diff' (for 'command')."
                    ),
                },
                "workdir": {"type": "string", "description": "Target directory path where CMDAI-CODE works (e.g. 'E:/Projects/my_app')."},
                "mode": {
                    "type": "string",
                    "enum": ["auto", "plan", "code"],
                    "description": "Optional CMDAI-CODE mode ('auto', 'plan', or 'code') for 'set_mode' or 'run'.",
                },
                "new_session": {"type": "boolean", "description": "Set true to force a fresh session instead of continuing the existing one."},
            },
            "required": ["workdir"],
        },
    },
    {
        "name": "models",
        "description": "Manage local CMDAI GGUF models (list installed models, check loaded model status, load or unload).",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "ps", "load", "unload"],
                    "description": "Local model runtime action.",
                },
                "model": {"type": "string", "description": "Model filename or partial name (for 'load')."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "ask",
        "description": "Prompt the user with a clarifying question and optional pipe-separated choices.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Question to ask the user."},
                "options": {"type": "string", "description": "Optional pipe-separated choices (e.g. 'Yes|No|Custom')."},
            },
            "required": ["question"],
        },
    },
]


def get_native_tool_schemas() -> List[Dict[str, Any]]:
    """Return list of standard JSON Schema tool objects formatted for OpenAI/OpenRouter/Anthropic API calls."""
    schemas = []
    for tool in TOOL_DEFINITIONS:
        schemas.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        })
    return schemas


def format_tool_call_as_xml(tool_name: str, args: Dict[str, Any]) -> str:
    """Format tool call arguments into standard CMDAI XML tag for unified handling."""
    name = tool_name.lower().strip()
    if name == "todo":
        action = args.get("action", "add")
        if action == "set":
            todos = args.get("todos", "")
            return f'<tool:todo action="set" todos="{todos}" />'
        elif action == "done":
            note = args.get("note", "")
            return f'<tool:todo action="done" note="{note}" />'
        else:
            note = args.get("note", "")
            return f'<tool:todo action="add" note="{note}" />'
    elif name in ("cmdai_code", "cmdai-code", "code_agent"):
        act = str(args.get("action") or "").strip()
        task = args.get("task", "") or args.get("prompt", "")
        workdir = args.get("workdir", "")
        mode = str(args.get("mode") or "").strip()
        new_sess = str(args.get("new_session", "")).lower() in ("true", "1", "yes")
        attrs = []
        if act:
            attrs.append(f'action="{act}"')
        if workdir:
            attrs.append(f'workdir="{workdir}"')
        if mode:
            attrs.append(f'mode="{mode}"')
        if new_sess:
            attrs.append('new_session="true"')
        attr_str = (" " + " ".join(attrs)) if attrs else ""
        if task:
            return f'<tool:cmdai_code{attr_str}>{task}</tool:cmdai_code>'
        return f'<tool:cmdai_code{attr_str} />'
    else:
        attr_strs = [f'{k}="{v}"' for k, v in args.items() if isinstance(v, (str, int, float, bool))]
        attrs = " " + " ".join(attr_strs) if attr_strs else ""
        return f'<tool:{name}{attrs} />'
