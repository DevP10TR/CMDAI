import html
import json
import os
import re
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..mcp.client import (
    PERM_DESTRUCTIVE,
    PERM_READ,
    PERM_WRITE,
    TOOL_NAME_PREFIX,
    block_texts,
    image_blocks,
)
from .render import render_tool_header
from .tools import (
    AgentContext,
    tool_memory,
    tool_scratch,
    tool_screenshot,
    tool_vision,
    tool_web,
)
from .pc_control import tool_system


_HALLUCINATED_TURN_RE = re.compile(
    r'(?:^|\n)\s*(?:</?ASSISTANT\s*>|<USER\s*>(?=\s*(?:\n|<tool_response\b))|<tool_response(?:\s+name=[^>]*)?>)',
    re.IGNORECASE,
)


def _truncate_hallucinated_turns(text: str) -> str:
    """Cut off model output at the first hallucinated turn or tool_response delimiter."""
    if not text:
        return ""
    m = _HALLUCINATED_TURN_RE.search(text)
    if not m:
        return text
    prefix = text[:m.start()]
    open_writes = len(re.findall(r'<tool:(?:write|edit)\b[^>]*>', prefix, re.IGNORECASE))
    close_writes = len(re.findall(r'</tool:(?:write|edit)\s*>', prefix, re.IGNORECASE))
    if open_writes <= close_writes:
        return prefix
    return text


def _parse_json_body(body: str, tool_name: str) -> Dict[str, Any]:
    """Parse an MCP tool's JSON argument object.

    Returns flat key/value pairs that fit the rest of the dispatch, which is
    typed as Dict[str, str]. A malformed body is reported back to the model
    as a readable sentence naming the tool, because a generic "invalid call"
    is the kind of message a small local model retries the exact same way
    twice before trying anything else.

    Nested values are kept as JSON text rather than flattened away: a server
    that expects {"filter": {"a": 1}} must receive that structure, and the
    client re-parses before sending.
    """
    text = (body or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except Exception as exc:
        return {"__json_error__": f"Arguments for {tool_name} are not valid JSON: {exc}"}
    if not isinstance(data, dict):
        return {"__json_error__": (
            f"Arguments for {tool_name} must be a JSON object with named fields, "
            f"got {type(data).__name__}.")}
    out: Dict[str, Any] = {}
    for key, value in data.items():
        out[str(key)] = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return out


def _all_tools_failed(events: List[Any]) -> bool:
    """True when a subagent ran real work and every single call errored.

    agent_note and scratch are bookkeeping, not evidence: a subagent that only
    wrote notes and had every real tool fail still got nowhere. Requiring one
    non-bookkeeping event keeps that case out of the "everything broke" verdict
    while still catching a run where the workspace was never actually read.
    """
    real = [
        e for e in (events or [])
        if isinstance(e, dict)
        and bool(str(e.get("tool", "") or "").strip())
        and e.get("type") not in ("text", "thinking")
        and str(e.get("tool", "")).strip().lower() not in ("agent_note", "scratch", "todo", "tasks")
    ]
    if not real:
        return False
    return all(not e.get("ok", False) for e in real)

    # ---- end of _all_tools_failed ----


def _parse_only(text: str) -> List[Tuple[str, Any]]:
    """Parse a model turn into ordered ("text"|"tool", payload) segments.

    Module-level so the subagent orchestrator and the TUI window can share the
    one parser. The orchestrator used to keep its own copy of the patterns and
    it drifted: it only knew <tool:name>, so <|tool_call>call:ls path="." />
    from a local Gemma was never seen as a call.
    """
    # Parsing never touches instance state - no engine, no context, no
    # callbacks - so one bare runner serves every caller. Built here rather
    # than at import time because the class is not defined yet at the top.
    if _BARE_RUNNER is None:
        globals()["_BARE_RUNNER"] = AgentRunner.__new__(AgentRunner)
    return _BARE_RUNNER.parse_chronological_segments(text)


def _strip_xml_tool_calls_clean(text: str) -> str:
    """Strip XML tool call tags, tool response artifacts, and JSON artifacts from model text."""
    if not text:
        return ""
    import re
    clean = re.sub(r'<(?:tool:\w+|tool_call|function_call)[\s\S]*?(?:/>|</(?:tool:)?\w+>|</tool_call>|</function_call>)', '', text, flags=re.IGNORECASE)
    clean = re.sub(r'</?(?:tool_response|tool_call|function_call)[^>]*>', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'<(?:tool:\w+|tool_call|function_call)[^>]*$', '', clean, flags=re.IGNORECASE)
    return clean


def clean_subagent_call(raw: Any) -> Dict[str, str]:
    """Normalize any subagent arguments dictionary or JSON payload."""
    import json
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = {"goal": raw}
    if not isinstance(raw, dict):
        return {"name": "Subagent", "role": "Subagent", "goal": str(raw or ""), "system": ""}

    # Unpack nested structures if present (e.g. live_code from bridge, arguments, parameters)
    for nest_key in ("live_code", "arguments", "parameters", "input"):
        val = raw.get(nest_key)
        if isinstance(val, str) and val.strip().startswith("{"):
            try:
                inner = json.loads(val)
                if isinstance(inner, dict):
                    raw = {**raw, **inner}
            except Exception:
                pass
        elif isinstance(val, dict):
            raw = {**raw, **val}

    role = str(
        raw.get("name")
        or raw.get("role")
        or raw.get("agent")
        or raw.get("type")
        or raw.get("subagent")
        or "Subagent"
    ).strip()
    if role.lower() in ("general", "subagent", "default", "agent"):
        role = "Subagent"
    role = role[:60] or "Subagent"

    goal = str(
        raw.get("goal")
        or raw.get("task")
        or raw.get("description")
        or raw.get("prompt")
        or raw.get("instruction")
        or raw.get("text")
        or raw.get("target")
        or ""
    ).strip()

    bg_val = raw.get("background")
    bg_str = str(bg_val if bg_val is not None else "").strip().lower()
    is_bg_true = bg_str in ("true", "1", "yes")
    is_bg_flag = bg_str in ("true", "1", "yes", "false", "0", "no", "")
    system = str(
        raw.get("system")
        or raw.get("system_prompt")
        or raw.get("context")
        or (bg_val if not is_bg_flag else "")
        or ""
    ).strip()

    if not goal and system:
        goal = system
    if not goal:
        goal = "Inspect the workspace and complete the task."
    if not system:
        system = f"You are the autonomous {role} subagent."

    res = dict(raw)
    res.update({
        "name": role,
        "role": role,
        "goal": goal,
        "task": goal,
        "system": system,
        "system_prompt": system,
        "background": "true" if is_bg_true else "false",
    })
    return res


_BARE_RUNNER: Optional["AgentRunner"] = None


class AgentRunner:

    def __init__(
        self,
        ctx: AgentContext,
        on_tool_start: Optional[Callable[[str, str], None]] = None,
        on_tool_finish: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
        on_todo_updated: Optional[Callable[[List[str]], None]] = None,
        ask_handler: Optional[Callable[[str, List[str]], str]] = None,
        mcp_client: Any = None,
        vision_capable: Optional[Callable[[], bool]] = None,
    ):
        self.ctx = ctx
        self.workdir = getattr(ctx, "workdir", ".")
        self.on_tool_start = on_tool_start
        self.on_tool_finish = on_tool_finish
        self.on_todo_updated = on_todo_updated
        self.ask_handler = ask_handler
        self.mcp = mcp_client
        self.permissions = getattr(ctx, "permissions", None)
        if not self.permissions:
            try:
                from ..core.settings import SettingsManager
                self.permissions = SettingsManager(getattr(ctx, "workdir", ".")).config.get("permissions")
            except Exception:
                self.permissions = None
        # Optional hook telling us whether the loaded model can accept images.
        # Defaults to False so an image from a tool is never silently attached
        # to a text-only model - the caller gets told instead.
        self.vision_capable = vision_capable
        # Permission gate state per server: "closed" -> "open" -> "write_granted".
        # Replaces a single boolean that either unlocked everything or nothing.
        self._mcp_state: Dict[str, str] = {}
        self.provider_id = "local_gguf"
        self.detach_requested: bool = False

    def request_background(self) -> bool:
        self.detach_requested = True
        return True

    # ------------------------------------------------------------- MCP gate

    def _mcp_state_of(self, server: str) -> str:
        return self._mcp_state.get(server, "closed")

    def _mcp_set_state(self, server: str, state: str) -> None:
        self._mcp_state[server] = state

    def _mcp_list_servers(self) -> Dict[str, Any]:
        """`<tool:mcp />` - always visible to the model, lists what exists.

        Sent as fields, not as the ASCII table `cmdai mcp list` prints. The
        table is for a person; a model handed it spends a turn working out
        that "-  xph  connected  10 tools" means a server with ten tools,
        and a model that gets this wrong tends to invent tool names.
        """
        if not self.mcp:
            return {"success": False, "servers": [],
                    "error": ("MCP is not configured, so there is nothing to list. "
                              "The user can add a server with `cmdai mcp add`. "
                              "Do not retry this tool and do not invent a server "
                              "name."),
                    "kind": "mcp_not_configured"}
        servers = self.mcp.model_servers()
        if not servers:
            # success=False on purpose. An empty list returned success=True, so
            # the tool card rendered blank and the failure looked like nothing
            # happening. There is no MCP here, and that is an error worth
            # saying out loud.
            return {"success": False, "servers": [],
                    "error": ("No MCP servers are configured, so no MCP tool can "
                              "be called. The user can add one with "
                              "`cmdai mcp add`. Do not retry and do not invent a "
                              "server or tool name."),
                    "kind": "mcp_no_servers"}
        usable = [s["name"] for s in servers if s.get("usable")]
        off = [s["name"] for s in servers if not s.get("enabled")]
        note = ("Call <tool:mcp open server=\"NAME\" /> for a server with usable=true. "
                "It returns that server's tools with their exact call syntax. "
                "Do not guess tool names.")
        if off:
            # Naming the switched-off servers here means the model can say
            # "xph is switched off" instead of reporting a missing tool,
            # which is the difference between a useful message and a wrong
            # one that sends the user looking for a typo.
            note += (" Servers with enabled=false are switched off by the user: "
                     + ", ".join(off) + ". They have no connection and no tools, "
                     "so do not call them and do not offer to fix them - the user "
                     "knows they are off.")
        if not usable:
            note += (" No server is currently usable, so no MCP tool can be "
                     "called at all this turn. Say that plainly instead of "
                     "pretending a tool failed.")
        return {"success": True, "servers": servers, "note": note}

    def _mcp_open_server(self, server_key: str) -> Dict[str, Any]:
        """`<tool:mcp open server="..." />` - the only unlock step.

        Tool schemas are returned as the result of this call, not injected
        into the system prompt. The prompt prefix therefore stays byte-identical
        between turns, which is what keeps llama.cpp's prompt cache working.
        """
        if not self.mcp:
            return {"success": False, "error": "MCP is not configured."}
        key = (server_key or "").strip().lower()
        target = self.mcp.servers.get(key)
        if target is None:
            known = ", ".join(self.mcp.server_names()) or "(none)"
            return {"success": False,
                    "error": f"No MCP server named '{key}'. Available: {known}"}
        # Switched off is not a fault and must not read like one. The
        # distinction decides what the model does next: a connection error
        # invites a retry, an off switch invites telling the user, and
        # there is nothing to retry here.
        if not target.enabled:
            return {"success": False, "error": (
                f"MCP server '{key}' is switched off in the user's configuration. "
                f"There is no connection to it and it has no tools, so nothing can "
                f"be called on it. The user has to turn it on themselves, in the "
                f"/mcp window. Do not retry and do not invent tool names for it."),
                "kind": "server_disabled", "server": key}
        if target.status != "connected":
            return {"success": False,
                    "error": f"MCP server '{key}' is not connected ({target.status})."
                             f" {target.error or ''}".strip(),
                    "kind": "server_not_connected", "server": key}

        defs = self.mcp.tool_definitions(only_servers=[key])
        if not defs:
            return {"success": False, "error": f"MCP server '{key}' exposed no tools."}

        self._mcp_set_state(key, "open")
        lines = [f"MCP server '{key}' is open. {len(defs)} tools available.", ""]
        for d in defs:
            schema = d.get("inputSchema") or {}
            required = schema.get("required") or []
            props = list((schema.get("properties") or {}).keys())
            arg_doc = ", ".join(props) if props else "none"
            lines.append(f"- {d['name']}  [perm: {d['permission']}]")
            if d.get("description"):
                lines.append(f"    {d['description']}")
            lines.append(f"    args: {arg_doc}; required: {', '.join(required) or 'none'}")
            lines.append("    call: <tool:%s>{\"arg\": \"value\"}</tool:%s>" % (d["name"], d["name"]))
        lines.append("")
        lines.append("Send arguments as a JSON object between the tags. "
                     "Write tools need the user's approval; you do not get to grant it yourself.")
        return {"success": True, "server": key, "tools": [d["name"] for d in defs],
                "schema_text": "\n".join(lines)}

    def _mcp_resolve(self, qualified: str) -> Optional[Tuple[str, str, str]]:
        """qualified name -> (server, real tool name, permission)."""
        if not self.mcp:
            return None
        try:
            tools = self.mcp.dynamic_tools()
        except Exception:
            return None
        found = tools.get((qualified or "").lower())
        if not found:
            return None
        server, real = found
        return server, real, self.mcp.permission_for(qualified)

    def _mcp_miss(self, name: str) -> Dict[str, Any]:
        """Explain a tool name that did not resolve, as precisely as we can.

        "No such tool" is true and useless when the real reason is that the
        server is switched off. Those two ask different things of the user -
        one asks them to check a spelling, the other tells them nothing to
        do - and only one of them is a real answer. A model that reports
        the wrong one sends the user off to fix something that is not
        broken.
        """
        base = {
            "success": False,
            "server": "",
            "tool": name,
        }
        if not self.mcp:
            base["error"] = "MCP is not configured, so no tool can be called."
            return base

        owner = self.mcp.server_for_tool_prefix(name)
        if owner and self.mcp.is_disabled(owner):
            base["server"] = owner
            base["error"] = (
                f"'{name}' belongs to MCP server '{owner}', which is switched off. "
                f"There is no connection to that server and it exposes no tools right "
                f"now, so this call cannot run. The user would have to switch the server "
                f"on themselves, in the /mcp window. Do not retry and do not try a "
                f"similar name."
            )
            return base

        if owner:
            base["server"] = owner
            state = self._mcp_state_of(owner)
            base["error"] = (
                f"MCP server '{owner}' is configured but has not been opened this "
                f"session (state: {state}), so its tool names are unknown to you. Call "
                f"<tool:mcp open server=\"{owner}\" /> and use the name it returns. "
                f"Do not guess it."
            )
            return base

        base["error"] = (
            f"No connected MCP tool named '{name}'. Call <tool:mcp /> to list servers, "
            f"then <tool:mcp open server=\"NAME\" /> for the exact tool names. Do not "
            f"guess them."
        )
        return base

    def _mcp_gate(self, server: str, tool: str, permission: str) -> Tuple[bool, str]:
        """Decide whether a call may proceed. Returns (allowed, reason)."""
        if permission == PERM_READ:
            return True, ""

        if str(getattr(self.ctx, "mode", "auto") or "auto").lower().strip() == "auto":
            return True, ""

        if permission == PERM_DESTRUCTIVE:
            # Asked every time, and answering yes does not grant anything:
            # deleting something cannot be undone, so no standing consent.
            if not self.ask_handler:
                return False, ("This tool is marked destructive and no confirmation "
                               "handler is available. Ask the user to run it themselves.")
            answer = self.ask_handler(
                f"MCP tool '{tool}' on server '{server}' is marked destructive. "
                f"This cannot be undone. Run it?", ["Yes", "No"])
            if str(answer).strip().lower().startswith("yes"):
                return True, ""
            return False, "User declined a destructive MCP call."

        # PERM_WRITE
        if self._mcp_state_of(server) == "write_granted":
            return True, ""
        if not self.ask_handler:
            return False, ("This tool writes, and no confirmation handler is available. "
                           "Ask the user to allow it first.")
        answer = self.ask_handler(
            f"MCP tool '{tool}' on server '{server}' writes data. "
            f"Allow it?", ["Yes", "Always", "No"])
        lowered = str(answer).strip().lower()
        if lowered.startswith("always"):
            self._mcp_set_state(server, "write_granted")
            return True, "granted"
        if lowered.startswith("yes"):
            return True, ""
        return False, "User declined the write."

    def _mcp_legacy_status(self) -> Dict[str, Any]:
        return self._mcp_list_servers()

    def _validate_mcp_args(self, server: str, tool: str,
                           args: Dict[str, Any]) -> Optional[str]:
        """Check arguments against the tool's own inputSchema.

        Cheap, and it happens before the call rather than after, so a
        malformed request does not reach a server that might act on half of
        it. Returns a human-readable problem, or None when fine.
        """
        schema = None
        for candidate in self.mcp.servers[server].tools:
            if str(candidate.get("name") or "") == tool:
                schema = candidate.get("inputSchema")
                break
        if not isinstance(schema, dict):
            return None
        try:
            import jsonschema  # already a project dependency
        except Exception:
            return None  # validation is a bonus, not a hard gate
        try:
            jsonschema.validate(instance=dict(args), schema=schema)
        except jsonschema.ValidationError as exc:
            where = "/".join(str(p) for p in exc.absolute_path) or "(root)"
            return f"Invalid arguments for {tool}: at '{where}', {exc.message}"
        except Exception:
            return None
        return None

    def _execute_mcp_tool(self, name: str, args: Dict[str, Any],
                          notify: bool) -> Dict[str, Any]:
        """Run one MCP tool, behind the permission gate."""
        if notify and self.on_tool_start:
            self.on_tool_start(name, "")

        def finish(res: Dict[str, Any], target: str = "") -> Dict[str, Any]:
            if notify and self.on_tool_finish:
                self.on_tool_finish(name, target, res)
            return res

        json_error = str(args.pop("__json_error__", "") or "")
        if json_error:
            return finish({"success": False, "error": json_error})

        resolved = self._mcp_resolve(name)
        if resolved is None:
            return finish(self._mcp_miss(name))
        server, real_tool, permission = resolved

        # Checked here, not only during resolution. A server can be
        # switched off between the tool name being resolved and the call
        # going out - the user presses a key in another window - and the
        # resolution table is a snapshot. Without this the call would run
        # against a server the user has just turned off, which is the one
        # outcome they were trying to prevent.
        if self.mcp.is_disabled(server):
            return finish({
                "success": False,
                "server": server,
                "tool": real_tool,
                "error": (
                    f"MCP server '{server}' is switched off, so there is no connection "
                    f"to it and '{real_tool}' cannot run. The user switched it off; do "
                    f"not retry and do not look for a different tool name."
                ),
                "kind": "server_disabled",
            })

        # Plan mode is read-only. An MCP write would be a change to the world
        # made behind the gate's back: the turn would be "planning" while the
        # server wrote something. Refused here, before the permission prompt,
        # because asking would already make it look allowed.
        if getattr(self.ctx, "mode", "auto") == "plan" and permission != PERM_READ:
            return finish({
                "success": False,
                "server": server,
                "tool": real_tool,
                "error": (
                    f"PLAN MODE ACTIVE: MCP tool '{real_tool}' on '{server}' is marked "
                    f"'{permission}', so it cannot run while planning. Plan mode only "
                    f"reads; write the action into the plan as a step instead."
                ),
            })

        # "open" and "write_granted" both count as open. Testing for the
        # literal "open" here would lock the server out again the moment the
        # user granted write access, which is the opposite of what granting
        # is supposed to do.
        if self._mcp_state_of(server) == "closed":
            target = self.mcp.servers.get(server)
            if target and target.status == "connected":
                self._mcp_set_state(server, "open")
            else:
                return finish({
                    "success": False,
                    "error": f"Server '{server}' is not open yet. Call "
                             f"<tool:mcp open server=\"{server}\" /> first and use the tool "
                             f"name it returns.",
                })

        allowed, reason = self._mcp_gate(server, real_tool, permission)
        if not allowed:
            return finish({"success": False, "error": reason})

        problem = self._validate_mcp_args(server, real_tool, args)
        if problem:
            # One retry is worth the round trip: the message names the field
            # and says what is wrong, which small local models act on well.
            return finish({"success": False, "error": problem})

        try:
            out = self.mcp.call_tool(server, real_tool, args)
        except Exception as exc:
            return finish({"success": False, "error": str(exc)[:500]})

        result = (out or {}).get("result") or {}
        texts = block_texts(result.get("content"))
        images = image_blocks(result)
        structured = result.get("structuredContent")

        if images and not (self.vision_capable and self.vision_capable()):
            # Say so. Dropping the image silently leaves the model claiming it
            # saw something it never received.
            texts.insert(0, (
                f"[{len(images)} image(s) returned by this tool, but the loaded model "
                f"is not vision-capable, so they were not attached. Switch to a "
                f"multimodal model to use them."
            ))

        payload: Dict[str, Any] = {"success": True, "server": server, "tool": real_tool}
        if texts:
            payload["text"] = "\n".join(texts)
        if structured is not None:
            payload["structured"] = structured
        if images and self.vision_capable and self.vision_capable():
            saved = self._queue_mcp_images(images, real_tool)
            if saved:
                payload["images"] = saved
                payload["text"] = (payload.get("text", "") +
                                   f"\n{len(saved)} image(s) from this tool were saved and "
                                   f"will be attached to the next request: "
                                   f"{', '.join(saved)}").strip()
            else:
                payload["text"] = (payload.get("text", "") +
                                   "\nImages were returned but could not be written to "
                                   "disk, so they were not attached.").strip()
        return finish(payload)

    def _queue_mcp_images(self, images: List[Dict[str, Any]], tool: str) -> List[str]:
        """Write tool-returned images to disk and queue them for the next turn.

        Reuses the same pending_images queue as the vision tool, so images
        from an MCP server and images from a screenshot take one path to the
        request instead of two that can drift apart.
        """
        saved: List[str] = []
        for index, item in enumerate(images):
            try:
                directory = os.path.join(tempfile.gettempdir(), "cmdaicode-mcp")
                os.makedirs(directory, exist_ok=True)
                ext = ".png"
                mime = str(item.get("mimeType") or "")
                if "jpeg" in mime or "jpg" in mime:
                    ext = ".jpg"
                elif "webp" in mime:
                    ext = ".webp"
                path = os.path.join(
                    directory, f"mcp_{re.sub(r'[^a-z0-9]+', '_', tool.lower())}_{index}{ext}")
                with open(path, "wb") as handle:
                    handle.write(item["data"])
                saved.append(path)
            except Exception:
                continue
        for path in saved:
            try:
                self.ctx.pending_images.append({"path": path, "question": ""})
            except Exception:
                pass
        return saved

    def parse_chronological_segments(self, text: str) -> List[Tuple[str, Any]]:
        if not text:
            return []

        text = _truncate_hallucinated_turns(text)

        def _inside_inline_code(pos: int) -> bool:
            """True when `pos` sits inside an inline `...` span on the same line."""
            line_start = text.rfind("\n", 0, pos) + 1
            line_end = text.find("\n", pos)
            if line_end == -1:
                line_end = len(text)
            line = text[line_start:line_end]
            rel = pos - line_start
            # Skip triple-backtick fences here; JSON extractor handles fenced blocks.
            if "```" in line:
                return False
            before = line[:rel]
            after = line[rel:]
            return (before.count("`") % 2 == 1) and ("`" in after)

        def _parse_attrs(attrs_str: str) -> Dict[str, str]:
            if not attrs_str:
                return {}
            attrs: Dict[str, str] = {}
            s = attrs_str.strip()
            i = 0
            n = len(s)
            key_re = re.compile(r'(\w+)\s*=\s*')
            next_key_re = re.compile(r'\s+\w+\s*=')
            while i < n:
                while i < n and s[i].isspace():
                    i += 1
                if i >= n:
                    break
                km = key_re.match(s, i)
                if not km:
                    i += 1
                    continue
                key = km.group(1).lower()
                i = km.end()
                if i >= n:
                    attrs[key] = ""
                    break
                q = s[i]
                if q in ('"', "'"):
                    i += 1
                    start_val = i
                    # Find closing quote followed by end-of-attrs or ` next_key=`.
                    end_val = -1
                    search_pos = start_val
                    while search_pos < n:
                        q_idx = s.find(q, search_pos)
                        if q_idx == -1:
                            break
                        rest = s[q_idx + 1:]
                        if not rest.strip() or next_key_re.match(rest):
                            end_val = q_idx
                            i = q_idx + 1
                            break
                        search_pos = q_idx + 1
                    if end_val == -1:
                        q_idx = s.find(q, start_val)
                        if q_idx != -1:
                            end_val = q_idx
                            i = q_idx + 1
                        else:
                            end_val = n
                            i = n
                    attrs[key] = html.unescape(s[start_val:end_val])
                else:
                    m_bare = re.match(r'([^\s>]+)', s[i:])
                    if m_bare:
                        attrs[key] = html.unescape(m_bare.group(1))
                        i += len(m_bare.group(1))
                    else:
                        i += 1
            return attrs

        def _normalize_attrs(t_name: str, a_dict: Dict[str, str], b_text: str = "") -> None:
            for p_alias in ("targetfile", "target_file", "filepath", "file_path", "filename", "file_name", "absolutepath", "absolute_path", "dest", "destination", "file", "target"):
                if p_alias in a_dict and "path" not in a_dict:
                    a_dict["path"] = a_dict[p_alias]
            for c_alias in ("codecontent", "code_content", "code", "replacementcontent", "replacement_content", "new_content"):
                if c_alias in a_dict and "content" not in a_dict:
                    a_dict["content"] = a_dict[c_alias]
            for o_alias in ("targetcontent", "target_content", "old_str", "old_content", "search", "original"):
                if o_alias in a_dict and "old" not in a_dict:
                    a_dict["old"] = a_dict[o_alias]
            for n_alias in ("replacementcontent", "replacement_content", "new_str", "new_content", "replace"):
                if n_alias in a_dict and "new" not in a_dict:
                    a_dict["new"] = a_dict[n_alias]
            for cmd_alias in ("commandline", "command_line", "command", "script", "exec"):
                if cmd_alias in a_dict and "cmd" not in a_dict:
                    a_dict["cmd"] = a_dict[cmd_alias]

            if t_name in ("write", "edit", "read") and not a_dict.get("path") and b_text:
                p_m = re.search(r'<(?:path|file|filename|target|target_file|targetfile|dest)>([^<>\r\n]+)</(?:path|file|filename|target|target_file|targetfile|dest)>', b_text, re.IGNORECASE)
                if p_m:
                    a_dict["path"] = p_m.group(1).strip()
                    b_text = re.sub(r'<(?:path|file|filename|target|target_file|targetfile|dest)>[^<>\r\n]+</(?:path|file|filename|target|target_file|targetfile|dest)>', '', b_text, flags=re.IGNORECASE).strip()
                else:
                    lines = [l.strip() for l in b_text.splitlines() if l.strip()]
                    if lines and "." in lines[0] and len(lines[0]) < 100 and not any(k in lines[0] for k in ("<", "{", "(", "import", "def", "class", "=", " ", "\t")):
                        a_dict["path"] = lines[0]
                        b_text = "\n".join(b_text.splitlines()[1:]).strip()

            def _strip_code_fence(s: str, target_path: str) -> str:
                if not s:
                    return s
                if str(target_path or "").lower().endswith((".md", ".markdown", ".mdx")):
                    return s
                st = s.strip()
                if st.startswith("```"):
                    first_nl = st.find("\n")
                    if first_nl != -1:
                        inner = st[first_nl + 1:]
                        if inner.rstrip().endswith("```"):
                            inner = inner.rstrip()[:-3].rstrip("\r\n")
                        return inner
                return s

            if t_name in ("scratch", "todo", "tasks"):
                for t_alias in ("items", "steps", "tasks", "list"):
                    if t_alias in a_dict and "todos" not in a_dict:
                        a_dict["todos"] = a_dict[t_alias]
                for n_alias in ("text", "step", "task", "item", "description", "content", "value"):
                    if n_alias in a_dict and "note" not in a_dict:
                        a_dict["note"] = a_dict[n_alias]
                if b_text and "note" not in a_dict:
                    a_dict["note"] = b_text
            elif t_name == "web":
                if "query" not in a_dict and b_text:
                    a_dict["query"] = b_text
            elif t_name == "ask":
                if "question" not in a_dict and b_text:
                    a_dict["question"] = b_text
            elif t_name == "subagent":
                if not a_dict.get("goal") and not a_dict.get("task") and b_text:
                    a_dict["goal"] = b_text
            elif t_name in ("cmdai_code", "cmdai-code", "code_agent"):
                if not a_dict.get("task") and not a_dict.get("prompt") and not a_dict.get("content") and b_text:
                    a_dict["task"] = b_text
            elif t_name in ("system", "pc", "windows", "os"):
                if not a_dict.get("value") and not a_dict.get("target") and b_text:
                    a_dict["value"] = b_text

        _NORM_NAMES = {
            "cmdai-code": "cmdai_code", "code_agent": "cmdai_code",
            "web_search": "web", "search_web": "web", "fetch_web_page": "webfetch", "read_url_content": "webfetch", "read_browser_page": "webfetch",
        }

        raw_matches = []

        gemma_pattern = re.compile(
            r'<\|tool_call>call:(\w+)(?:\s+([^>]*?))?(?:\s*/>|>(.*?)</\|tool_call>)',
            re.DOTALL | re.IGNORECASE,
        )
        for match in gemma_pattern.finditer(text):
            if _inside_inline_code(match.start()):
                continue
            tool_name = _NORM_NAMES.get(match.group(1).lower(), match.group(1).lower())
            attrs_str = match.group(2) or ""
            body = (match.group(3) or "").strip()
            attrs = _parse_attrs(attrs_str)
            _normalize_attrs(tool_name, attrs, body)
            raw_matches.append((match.start(), match.end(), tool_name, attrs))

        body_pattern = re.compile(
            r'<tool:(\w+)(?:\s+([^>]*?))?\s*/?>(.*?)</(?:tool:)?(?:\1|tool)>',
            re.DOTALL | re.IGNORECASE,
        )
        for match in body_pattern.finditer(text):
            if _inside_inline_code(match.start()):
                continue
            tool_name = _NORM_NAMES.get(match.group(1).lower(), match.group(1).lower())
            attrs_str = (match.group(2) or "").rstrip("/").strip()
            body = match.group(3).strip()
            attrs = _parse_attrs(attrs_str)
            _normalize_attrs(tool_name, attrs, body)
            if tool_name.startswith(TOOL_NAME_PREFIX) and body:
                attrs.update(_parse_json_body(body, tool_name))
            raw_matches.append((match.start(), match.end(), tool_name, attrs))

        unclosed_body_pattern = re.compile(
            r'<tool:(subagent|cmdai_code|code_agent)\b(?:\s+([^>]*?))?\s*/?>\s*((?:(?!</?(?:tool:|tool_call|\|tool_call)).)+)$',
            re.DOTALL | re.IGNORECASE,
        )
        for match in unclosed_body_pattern.finditer(text):
            if _inside_inline_code(match.start()):
                continue
            tool_name = _NORM_NAMES.get(match.group(1).lower(), match.group(1).lower())
            attrs_str = (match.group(2) or "").rstrip("/").strip()
            body = match.group(3).strip()
            # Ignore stray placeholder mentions at end of prose such as "<tool:command>).".
            if body and (attrs_str or not re.fullmatch(r'[\s\)\.,;:\'"`>]+', body)):
                attrs = _parse_attrs(attrs_str)
                _normalize_attrs(tool_name, attrs, body)
                raw_matches.append((match.start(), match.end(), tool_name, attrs))

        single_tag_pattern = re.compile(r'<tool:(\w+)(?:\s+([^>]*?))?\s*(/?)>', re.DOTALL | re.IGNORECASE)
        for match in single_tag_pattern.finditer(text):
            if _inside_inline_code(match.start()):
                continue
            attrs_str = (match.group(2) or "").strip()
            is_self_closing = (match.group(3) == "/")
            # Bare non-self-closing tags without attributes (e.g. "np. <tool:write>, <tool:ls>") are prose mentions, not tool calls.
            if not is_self_closing and not attrs_str:
                continue
            tool_name = _NORM_NAMES.get(match.group(1).lower(), match.group(1).lower())
            attrs = _parse_attrs(attrs_str)
            _normalize_attrs(tool_name, attrs, "")
            raw_matches.append((match.start(), match.end(), tool_name, attrs))

        tool_call_block = re.compile(
            r'<tool_call>(?:call:)?(\w+)(?:\s+([^>]*?))?(?:\s*/>|>(.*?)</tool_call>)',
            re.DOTALL | re.IGNORECASE,
        )
        for match in tool_call_block.finditer(text):
            tool_name = _NORM_NAMES.get(match.group(1).lower(), match.group(1).lower())
            attrs_str = match.group(2) or ""
            body = (match.group(3) or "").strip()
            attrs = _parse_attrs(attrs_str)
            _normalize_attrs(tool_name, attrs, body)
            raw_matches.append((match.start(), match.end(), tool_name, attrs))
        dec = json.JSONDecoder()
        scan_pos = 0
        while scan_pos < len(text):
            idx_obj = text.find("{", scan_pos)
            idx_arr = text.find("[", scan_pos)
            candidates = [p for p in (idx_obj, idx_arr) if p != -1]
            if not candidates:
                break
            start_idx = min(candidates)
            try:
                parsed, end_offset = dec.raw_decode(text[start_idx:])
                full_end = start_idx + end_offset
                calls = parsed if isinstance(parsed, list) else [parsed]
                has_tool = False
                for call_obj in calls:
                    if not isinstance(call_obj, dict):
                        continue
                    raw_name = call_obj.get("tool_name") or call_obj.get("name") or call_obj.get("tool") or ""
                    if not raw_name and isinstance(call_obj.get("function"), dict):
                        raw_name = call_obj["function"].get("name", "")
                    elif not raw_name and isinstance(call_obj.get("function"), str):
                        raw_name = call_obj["function"]
                    if not raw_name:
                        continue
                    norm_name = _NORM_NAMES.get(str(raw_name).lower().strip(), str(raw_name).lower().strip())
                    args: Dict[str, str] = {}
                    raw_args = call_obj.get("arguments") or call_obj.get("args") or call_obj.get("parameters")
                    if isinstance(raw_args, str):
                        try:
                            raw_args = json.loads(raw_args)
                        except Exception:
                            raw_args = {}
                    if isinstance(raw_args, dict):
                        args.update({str(k).lower(): str(v) if not isinstance(v, (dict, list)) else json.dumps(v, ensure_ascii=False) for k, v in raw_args.items()})
                    for k, v in call_obj.items():
                        k_low = str(k).lower()
                        if k_low not in ("tool_name", "name", "tool", "tool_call_id", "id", "type", "function", "arguments", "args", "parameters"):
                            if k_low not in args:
                                args[k_low] = str(v) if not isinstance(v, (dict, list)) else json.dumps(v, ensure_ascii=False)

                    # Normalize common parameter aliases
                    for p_alias in ("targetfile", "target_file", "filepath", "file_path", "filename", "file_name", "absolutepath", "absolute_path", "dest", "destination"):
                        if p_alias in args and "path" not in args:
                            args["path"] = args[p_alias]
                    for c_alias in ("codecontent", "code_content", "code", "replacementcontent", "replacement_content", "new_content"):
                        if c_alias in args and "content" not in args:
                            args["content"] = args[c_alias]
                    for o_alias in ("targetcontent", "target_content", "old_str", "old_content", "search", "original"):
                        if o_alias in args and "old" not in args:
                            args["old"] = args[o_alias]
                    for n_alias in ("replacementcontent", "replacement_content", "new_str", "new_content", "replace"):
                        if n_alias in args and "new" not in args:
                            args["new"] = args[n_alias]
                    for cmd_alias in ("commandline", "command_line", "command", "script", "exec"):
                        if cmd_alias in args and "cmd" not in args:
                            args["cmd"] = args[cmd_alias]

                    m_start = start_idx
                    m_end = full_end
                    pre_text = text[:start_idx].rstrip()
                    if pre_text.endswith("```json") or pre_text.endswith("```"):
                        fence_idx = pre_text.rfind("```")
                        if fence_idx != -1:
                            m_start = fence_idx
                    post_text = text[full_end:].lstrip()
                    if post_text.startswith("```"):
                        m_end = full_end + text[full_end:].find("```") + 3

                    raw_matches.append((m_start, m_end, norm_name, args))
                    has_tool = True
                scan_pos = full_end if has_tool else (start_idx + 1)
            except Exception:
                scan_pos = start_idx + 1

        raw_matches.sort(key=lambda m: (m[0], -m[1]))

        non_overlapping = []
        last_end = 0
        for start, end, tool_name, attrs in raw_matches:
            if start >= last_end:
                non_overlapping.append((start, end, tool_name, attrs))
                last_end = end

        segments: List[Tuple[str, Any]] = []
        pos = 0
        last_tool_sig: Optional[Tuple[str, str, str]] = None
        for start, end, tool_name, attrs in non_overlapping:
            if start > pos:
                raw_chunk = text[pos:start]
                clean_chunk = self.strip_xml_tool_calls(raw_chunk).strip()
                clean_chunk = re.sub(r'</?(?:tool_response|tool_call|function_call)[^>]*>', '', clean_chunk, flags=re.IGNORECASE).strip()
                if clean_chunk:
                    segments.append(("text", clean_chunk))
                    last_tool_sig = None

            # Deduplicate immediate identical tool calls (e.g. repeated write calls for same file)
            cur_path = str(attrs.get("path") or attrs.get("targetfile") or attrs.get("file") or attrs.get("cmd") or "").strip()
            cur_content = str(attrs.get("content") or attrs.get("code") or attrs.get("codecontent") or "").strip()
            cur_sig = (tool_name, cur_path, cur_content[:100])
            if last_tool_sig and cur_sig == last_tool_sig:
                pos = end
                continue

            last_tool_sig = cur_sig
            segments.append(("tool", (tool_name, attrs)))
            pos = end

        if pos < len(text):
            raw_chunk = text[pos:]
            clean_chunk = self.strip_xml_tool_calls(raw_chunk).strip()
            clean_chunk = re.sub(r'</?(?:tool_response|tool_call|function_call)[^>]*>', '', clean_chunk, flags=re.IGNORECASE).strip()
            if clean_chunk:
                segments.append(("text", clean_chunk))

        return segments

    def parse_xml_tool_calls(self, text: str) -> List[Tuple[str, Dict[str, str]]]:
        segments = self.parse_chronological_segments(text)
        return [data for kind, data in segments if kind == "tool"]

    def check_tool_permission(self, tool_name: str, args: Dict[str, Any], mode: str) -> str:
        """Evaluate granular permission: returns 'allow', 'ask', or 'deny'."""
        name = tool_name.lower().strip()
        mode_low = str(mode or "auto").lower().strip()
        if mode_low == "chat":
            return "deny"
        if name in ("system", "pc", "windows", "os"):
            act = str((args or {}).get("action") or "sysinfo").strip().lower()
            perms = getattr(self, "permissions", None) or {}
            sys_rule = perms.get("system")
            dec = "ask" if act in ("kill_process", "shutdown", "restart") else "allow"
            if isinstance(sys_rule, str):
                dec = sys_rule
            elif isinstance(sys_rule, dict):
                dec = str(sys_rule.get(act, sys_rule.get("*", dec))).lower().strip()
            if act in ("kill_process", "shutdown", "restart"):
                # Destructive process/power actions require explicit "allow" or stay "ask" even in auto mode.
                return dec if dec in ("allow", "deny") else "ask"
            if mode_low == "ask" and act not in ("sysinfo", "processes", "clipboard_get"):
                return "ask"
            if mode_low == "auto" and dec == "ask":
                return "allow"
            return dec if dec in ("allow", "ask", "deny") else "ask"
        if name in ("models", "model", "ollama"):
            act = str((args or {}).get("action") or "list").strip().lower()
            if mode_low == "ask" and act in ("load", "unload"):
                return "ask"
            return "allow"
        if mode_low == "ask" and name in ("cmdai_code", "cmdai-code", "code_agent"):
            return "ask"
        if name == "memory":
            return "allow"

        def _resolve_decision(dec: str) -> str:
            d = str(dec or "").lower().strip()
            if d not in ("allow", "ask", "deny"):
                d = "ask"
            if mode_low == "auto" and d == "ask":
                return "allow"
            return d

        perms = getattr(self, "permissions", None)
        if not perms or not isinstance(perms, dict):
            return "allow"

        rule = perms.get(name)
        if rule is None:
            return "allow"
        if isinstance(rule, str):
            return _resolve_decision(rule)
        return "allow"

    def execute_tool(self, tool_name: str, args: Dict[str, str], ctx: Any = None,
                   ask: bool = True, notify: bool = True) -> Dict[str, Any]:
        """Run one tool.

        ctx:  override the AgentContext. Subagents get their own instance so
              their scratch/todo list does not land on the lead agent's
              checklist. Defaults to the runner context.
        ask:    set False for subagent calls. Subagents work without popping the
                approval dialog (variant C: the commit modal stays the gate).
        notify: set False for subagent calls. Without this a subagent's own tool
                activity fires on_tool_start/on_tool_finish and lands in the lead
                agent's chat as extra markers next to the subagent marker.
        """
        name = tool_name.lower().strip()
        # Without this every tool that takes a context (ls, read, write, glob,
        # search...) got None and died on "NoneType has no attribute
        # resolve_path". The subagent window showed it twice in a row.
        ctx = ctx if ctx is not None else self.ctx

        mode = getattr(ctx, "mode", "auto")
        if mode == "plan" and name in ("cmdai_code", "cmdai-code", "code_agent"):
            return {
                "success": False,
                "error": (
                    f"PLAN MODE ACTIVE: '{name}' is not available in Plan mode - the app "
                    "refuses it, so do not call it again in this turn. Record your plan "
                    "with <tool:todo action=\"set\" todos=\"step one|step two\" /> and present "
                    "the plan out for the user: Goal / Actions / Steps / How to verify."
                ),
            }

        perm_decision = self.check_tool_permission(name, args, mode)
        if perm_decision == "deny" and mode != "plan":
            return {
                "success": False,
                "error": f"Permission denied: execution of tool '{name}' is forbidden by permissions configuration.",
            }
        should_ask = (perm_decision == "ask")
        if should_ask and self.ask_handler and ask:
            target = str(args.get("target") or args.get("action") or args.get("model") or args.get("task") or name)
            details = {
                "action": name,
                "target": target,
                "diff_info": str(args.get("value") or args.get("task") or ""),
            }
            try:
                approved = self.ask_handler(
                    f"Allow agent to execute '{name}' ({target})?",
                    ["Yes, proceed", "No, skip"],
                    details,
                )
            except TypeError:
                approved = self.ask_handler(
                    f"Allow agent to execute '{name}' ({target})?",
                    ["Yes, proceed", "No, skip"],
                )
            if approved not in ("Yes, proceed", "allow", "yes"):
                return {"success": False, "error": f"User rejected execution of '{name}': {target}"}

        try:
            if name in ("task", "tasks", "bg_task", "background_task"):
                from .tasks import BackgroundTaskManager
                mgr = BackgroundTaskManager.get_instance()
                action = str(args.get("action", "list")).lower()
                tid = str(args.get("task_id", "") or args.get("id", "")).strip()
                if notify and self.on_tool_start:
                    self.on_tool_start("task", f"{action} {tid}".strip())
                if action == "kill":
                    ok = mgr.kill_task(tid)
                    res = {"success": ok, "task_id": tid, "action": "kill"}
                elif action in ("status", "logs", "read"):
                    t = mgr.get_task(tid)
                    if t:
                        logs = t.read_logs()
                        res = {"success": True, "task_id": tid, "running": t.is_running(), "status": "running" if t.is_running() else f"finished (code {t.poll()})", "logs": logs}
                    else:
                        res = {"success": False, "error": f"Task '{tid}' not found."}
                else:
                    tasks_list = mgr.list_tasks()
                    res = {"success": True, "tasks": tasks_list}
                if notify and self.on_tool_finish:
                    self.on_tool_finish("task", f"{action} {tid}".strip(), res)
                return res


            elif name in ("web", "webfetch", "fetch", "web_fetch"):
                target = args.get("query") or args.get("url") or args.get("q") or ""
                if not target:
                    err_res = {
                        "success": False,
                        "error": "missing required attribute 'query' or 'url'",
                        "missing": "query",
                        "example": '<tool:web query="search query" /> or <tool:webfetch url="https://example.com" />',
                    }
                    if notify and self.on_tool_start:
                        self.on_tool_start("web", "")
                    if notify and self.on_tool_finish:
                        self.on_tool_finish("web", "", err_res)
                    return err_res
                if notify and self.on_tool_start:
                    self.on_tool_start("web", target)
                res = tool_web(ctx, target)
                if notify and self.on_tool_finish:
                    self.on_tool_finish("web", target, res)
                return res

            elif name in ("scratch", "todo", "tasks"):
                action = args.get("action", "add")
                note = (
                    args.get("note")
                    or args.get("text")
                    or args.get("step")
                    or args.get("task")
                    or args.get("item")
                    or args.get("description")
                    or args.get("content")
                    or ""
                )
                # A plan is recorded in one call:
                #   <tool:todo action="set" todos="step one|step two" />
                todos = (
                    args.get("todos")
                    or args.get("items")
                    or args.get("steps")
                    or args.get("tasks")
                    or ""
                )
                new_text = args.get("new_text", "") or args.get("newtext", "")
                raw_index = args.get("index")
                index_val: Optional[int] = None
                if raw_index not in (None, ""):
                    try:
                        index_val = int(str(raw_index).strip())
                    except (TypeError, ValueError):
                        index_val = None
                target_str = f"{action}: {note}" if note else action
                if todos:
                    target_str = f"{action}: {todos}"
                if notify and self.on_tool_start:
                    self.on_tool_start("todo", target_str)
                res = tool_scratch(
                    ctx,
                    action,
                    note,
                    index=index_val,
                    new_text=new_text,
                    todos=todos,
                )
                if self.on_todo_updated:
                    self.on_todo_updated(list(res.get("visible_steps", [])))
                if notify and self.on_tool_finish:
                    self.on_tool_finish("todo", target_str, res)
                return res

            elif name == "ask":
                question = args.get("question") or args.get("prompt") or args.get("note") or "Agent confirmation requested"
                options_str = args.get("options", "")
                if options_str:
                    if "|" in options_str:
                        options = [o.strip() for o in options_str.split("|") if o.strip()]
                    elif "," in options_str:
                        options = [o.strip() for o in options_str.split(",") if o.strip()]
                    else:
                        options = [options_str.strip()]
                else:
                    options = ["Yes", "No", "Always", "Cancel"]

                if notify and self.on_tool_start:
                    self.on_tool_start("ask", question)

                answer = ""
                if self.ask_handler:
                    answer = self.ask_handler(question, options)
                else:
                    answer = "Interactive confirmation modal unavailable"

                res = {"success": True, "question": question, "answer": answer}
                if notify and self.on_tool_finish:
                    self.on_tool_finish("ask", question, res)
                return res

            elif name == "screenshot":
                if not (self.vision_capable and self.vision_capable()):
                    err_res = {
                        "success": False,
                        "error": "The active model does not support vision or image analysis. Screenshot tool is disabled.",
                        "note": "Switch to a vision-capable multimodal model (e.g. Gemini, GPT-4o, Claude) to capture and inspect screenshots.",
                    }
                    if notify and self.on_tool_start:
                        self.on_tool_start("screenshot", str(args.get("target", "")))
                    if notify and self.on_tool_finish:
                        self.on_tool_finish("screenshot", str(args.get("target", "")), err_res)
                    return err_res
                target = args.get("target", "active_window")
                if notify and self.on_tool_start:
                    self.on_tool_start("screenshot", target)
                res = tool_screenshot(ctx, target)
                if notify and self.on_tool_finish:
                    self.on_tool_finish("screenshot", target, res)
                return res

            elif name == "vision":
                if not (self.vision_capable and self.vision_capable()):
                    err_res = {
                        "success": False,
                        "error": "The active model does not support vision or image analysis. Vision tool is disabled.",
                        "note": "Switch to a vision-capable multimodal model (e.g. Gemini, GPT-4o, Claude) to use vision.",
                    }
                    if notify and self.on_tool_start:
                        self.on_tool_start("vision", str(args.get("image_path") or args.get("path", "")))
                    if notify and self.on_tool_finish:
                        self.on_tool_finish("vision", str(args.get("image_path") or args.get("path", "")), err_res)
                    return err_res
                image_path = args.get("image_path") or args.get("path", "")
                question = args.get("question", "")
                if notify and self.on_tool_start:
                    self.on_tool_start("vision", image_path)
                res = tool_vision(ctx, image_path, question)
                if notify and self.on_tool_finish:
                    self.on_tool_finish("vision", image_path, res)
                return res

            elif name == "memory":
                act = str(args.get("action") or "list").strip()
                scope = "global"
                content = str(args.get("content") or args.get("text") or "")
                idx = int(args.get("index") or 0) if str(args.get("index") or "").isdigit() else 0
                query = str(args.get("query") or "")
                target = f"{act} (global)"
                if notify and self.on_tool_start:
                    self.on_tool_start("memory", target)
                res = tool_memory(self.ctx, action=act, content=content, scope=scope, index=idx, query=query)
                if notify and self.on_tool_finish:
                    self.on_tool_finish("memory", target, res)
                return res

            elif name == "mcp":
                wanted = (args.get("server") or args.get("name") or "").strip()
                if notify and self.on_tool_start:
                    # The server name has to be known *before* the callback:
                    # the start hook decides whether this call draws a card,
                    # and with an empty target it could not tell a bare
                    # listing from `<tool:mcp open server="X" />` and silenced
                    # both, hiding the tool list the model just fetched.
                    self.on_tool_start("mcp", wanted)
                res = self._mcp_open_server(wanted) if wanted else self._mcp_list_servers()
                if notify and self.on_tool_finish:
                    self.on_tool_finish("mcp", wanted, res)
                return res

            elif name == "subagent":
                return self.execute_subagents([args])

            elif name == "agent_note":
                # Only reachable if a subagent forgot to route through
                # execute_subagents. There is no lead agent to read a note.
                res = {
                    "success": False,
                    "error": "agent_note is only available inside a subagent. Use <tool:scratch action=\"add\" /> for your own checklist.",
                }
                if notify and self.on_tool_finish:
                    self.on_tool_finish("agent_note", "", res)
                return res

            elif name in ("cmdai_code", "cmdai-code", "code_agent"):
                from ..core.code_bridge import (
                    _enrich_cmdai_code_prompt,
                    decide_for_cmdai_code,
                    install_cmdai_code,
                    resolve_cmdai_code_script,
                    send_to_cmdai_code,
                )
                act = str(args.get("action") or "run").strip().lower()
                task_txt = str(args.get("task") or args.get("prompt") or args.get("command") or args.get("content") or "").strip()
                raw_wd = (
                    args.get("workdir")
                    or args.get("directory")
                    or args.get("dir")
                    or args.get("path")
                    or args.get("folder")
                    or args.get("cwd")
                    or getattr(ctx, "workdir", "")
                    or self.workdir
                )
                target_wd = os.path.abspath(os.path.expanduser(str(raw_wd).strip() or "."))
                if act in ("run", "command") and task_txt:
                    act, task_txt = _enrich_cmdai_code_prompt(task_txt, act, target_wd)
                mode_arg = str(args.get("mode") or "").strip().lower()
                new_sess = str(args.get("new_session") or "").lower() in ("true", "1", "yes")
                summary_lbl = f"{act}: {task_txt[:55]} ({target_wd})" if task_txt else f"{act} ({target_wd})"
                if notify and self.on_tool_start:
                    self.on_tool_start("cmdai_code", summary_lbl)
                if not resolve_cmdai_code_script():
                    installed_ok = False
                    install_err = "CMDAI-CODE is not installed. Run /code to install the latest version from GitHub."
                    if self.ask_handler:
                        choice = self.ask_handler(
                            "CMDAI-CODE is not installed. Install the latest version from GitHub (DevP10TR/CMDAI-CODE)?",
                            ["Yes, install latest CMDAI-CODE", "No, cancel"],
                        )
                        if str(choice).lower().startswith("yes"):
                            inst = install_cmdai_code()
                            if inst.get("ok"):
                                installed_ok = True
                            else:
                                install_err = str(inst.get("error") or install_err)
                    if not installed_ok:
                        res = {"success": False, "error": install_err}
                        if notify and self.on_tool_finish:
                            self.on_tool_finish("cmdai_code", summary_lbl, res)
                        return res

                def _llm_decide(q: str, opts: List[str]) -> str:
                    if str(q or "").startswith("Allow agent to execute "):
                        for o in opts:
                            if o.lower() in ("yes, proceed", "allow", "yes"):
                                return o
                        return opts[0] if opts else "Yes, proceed"
                    try:
                        from ..core.providers import stream_chat_completion
                        from ..core.settings import get_settings

                        s = get_settings()
                        prov_id = str(getattr(self, "provider_id", "") or s.config.get("default_provider") or "").strip()
                        mod_id = str(s.config.get("default_model") or "").strip()
                        if not prov_id or not mod_id or prov_id == "local_gguf":
                            return opts[0] if opts else "Yes, proceed"
                        api_key = s.get_api_key(prov_id)
                        opts_text = ", ".join(opts) if opts else "Yes, proceed, No, skip"
                        prompt_msg = (
                            f"You are CMDAI supervising CMDAI-CODE in '{target_wd}' for task: {task_txt or '(active task)'}.\n"
                            f"CMDAI-CODE asked: {q}\n"
                            f"Options: {opts_text}\n"
                            "Reply with ONLY the exact chosen option or a concise direct answer."
                        )
                        out = stream_chat_completion(
                            provider_id=prov_id,
                            model_id=mod_id,
                            messages=[{"role": "user", "content": prompt_msg}],
                            api_key=api_key,
                            generation_params={"temperature": 0.1, "max_tokens": 64, "reasoning_level": "off"},
                        )
                        ans = str(out[0] if isinstance(out, tuple) else getattr(out, "content", "")).strip()
                        if ans and not ans.startswith(("⚠", "Connection error:", "Unknown provider:")):
                            return ans.splitlines()[0].strip()
                    except Exception:
                        pass
                    return opts[0] if opts else "Yes, proceed"

                def _cmdai_decision_handler(question: str, options: List[str], details: Dict[str, Any]) -> str:
                    chosen = decide_for_cmdai_code(
                        question,
                        options,
                        task=task_txt,
                        workdir=target_wd,
                        details=details,
                        llm_decide=_llm_decide,
                    )
                    if notify and self.on_tool_start:
                        self.on_tool_start("cmdai_code", f"decided: {chosen} ({str(question)[:45]})")
                    return chosen

                r = send_to_cmdai_code(
                    task=task_txt,
                    workdir=target_wd,
                    action=act,
                    mode=mode_arg,
                    new_session=new_sess,
                    decision_handler=_cmdai_decision_handler,
                )
                res = {
                    "success": bool(r.get("ok")),
                    "action": r.get("action", act),
                    "turn": r.get("turn", 0),
                    "turns": r.get("turns", r.get("turn", 0)),
                    "continued": r.get("continued", False),
                    "session_id": r.get("session_id") or r.get("active_session_id", ""),
                    "agent_mode": r.get("agent_mode", mode_arg or "auto"),
                    "response": r.get("response", ""),
                    "tool_calls": r.get("tool_calls", []),
                    "todo_list": r.get("todo_list", []),
                    "modified_files": r.get("modified_files", []),
                    "pending_approval": bool(r.get("pending_approval", False)),
                    "decisions": r.get("decisions", []),
                    "workdir": r.get("workdir", target_wd),
                }
                if "installed" in r:
                    res["installed"] = r["installed"]
                if not r.get("ok") and r.get("error"):
                    res["error"] = r["error"]
                if notify and self.on_tool_finish:
                    self.on_tool_finish("cmdai_code", summary_lbl, res)
                return res

            elif name in ("system", "pc", "windows", "os"):
                action = str(args.get("action") or "sysinfo").strip()
                target_val = str(args.get("target") or "").strip()
                val_arg = str(args.get("value") or args.get("content") or "").strip()
                summary_lbl = f"{action} {target_val}".strip()
                if notify and self.on_tool_start:
                    self.on_tool_start("system", summary_lbl)
                out = tool_system(ctx, action=action, target=target_val, value=val_arg)
                ok = not out.startswith(("Error:", "Refused:"))
                res = {"success": ok, "action": action, "output": out}
                if not ok:
                    res["error"] = out
                if notify and self.on_tool_finish:
                    self.on_tool_finish("system", summary_lbl, res)
                return res

            elif name in ("models", "model", "ollama"):
                from ..core.engine import SimpleGGUFLoader
                from ..core.model_paths import resolve_models_dir
                action = str(args.get("action") or "list").strip().lower()
                model_req = str(args.get("model") or args.get("name") or "").strip()
                summary_lbl = f"{action} {model_req}".strip()
                if notify and self.on_tool_start:
                    self.on_tool_start("models", summary_lbl)
                loader = getattr(self, "_gguf_loader", None)
                if loader is None:
                    loader = SimpleGGUFLoader(resolve_models_dir())
                    self._gguf_loader = loader
                if action in ("list", "ls"):
                    res = {"success": True, "models_dir": loader.models_dir, "models": loader.list_models()}
                elif action == "ps":
                    loaded = os.path.basename(loader.current_model_path) if getattr(loader, "llm", None) and loader.current_model_path else ""
                    res = {"success": True, "loaded_model": loaded, "n_ctx": getattr(loader, "current_n_ctx", 0), "loader": getattr(loader, "current_loader", "")}
                elif action == "unload":
                    loader.unload()
                    res = {"success": True, "status": "unloaded"}
                elif action == "load":
                    if not model_req:
                        res = {"success": False, "error": "Specify `model` filename for action='load'."}
                    else:
                        ok = loader.load_model(model_req)
                        res = {"success": bool(ok), "loaded_model": os.path.basename(loader.current_model_path or model_req)}
                        if not ok:
                            res["error"] = f"Failed to load {model_req}"
                else:
                    res = {"success": False, "error": f"Unknown models action {action!r}. Use list, ps, load, unload."}
                if notify and self.on_tool_finish:
                    self.on_tool_finish("models", summary_lbl, res)
                return res

            elif name.startswith(TOOL_NAME_PREFIX):
                return self._execute_mcp_tool(name, args, notify)

            display_name = tool_name or name
            target_str = str(args.get("path") or args.get("prompt") or args.get("cmd") or args.get("target") or args.get("url") or "")
            has_vis = bool(self.vision_capable and self.vision_capable())
            allowed_tools = "models, system, cmdai_code, web, memory, todo, ask, subagent, mcp"
            if has_vis:
                allowed_tools += ", screenshot, vision"
            err_res = {
                "success": False,
                "error": f"Tool '{display_name}' does not exist and cannot be executed.",
                "note": (
                    f"This tool is not available in CMDAI. You must only use the officially available tools: {allowed_tools}. "
                    "For coding or file editing tasks in a repository, delegate to <tool:cmdai_code>."
                ),
            }
            if notify and self.on_tool_start:
                self.on_tool_start(display_name, target_str)
            if notify and self.on_tool_finish:
                self.on_tool_finish(display_name, target_str, err_res)
            return err_res
        except Exception as e:
            err_res = {"success": False, "error": str(e)}
            if notify and self.on_tool_finish:
                self.on_tool_finish(name, str(args.get("path") or args.get("cmd") or ""), err_res)
            return err_res

    def execute_subagents(
        self,
        calls: List[Dict[str, str]],
        on_progress: Optional[Callable[[Any], None]] = None,
        is_local: Optional[bool] = None,
        model_call: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
        max_turns: int = 25,
    ) -> Dict[str, Any]:
        """Run a batch of <tool:subagent ... /> calls.

        One task per call, so N calls produce N markers in the chat. A local
        model runs them one after another; an API model runs them all at once
        (SubagentOrchestrator.execute_tasks branches on is_local).
        """
        from .subagents.models import SubagentTask
        from .subagents.orchestrator import SUBAGENT_AVAILABLE_TOOLS, SubagentOrchestrator
        from .tools import AgentContext

        calls = [clean_subagent_call(c) for c in (calls or []) if c]
        if not calls:
            return {"success": False, "error": "subagent was called without a task", "subagents": []}

        if getattr(self.ctx, "workdir", None):
            self.workdir = self.ctx.workdir
        mode = getattr(self.ctx, "mode", "auto")
        orch = SubagentOrchestrator(engine=None)
        tasks: List[SubagentTask] = []

        for raw in calls:
            role = raw["role"]
            goal = raw["goal"]
            system = raw["system"]
            call_workdir = (raw.get("workdir") or self.workdir)
            task = SubagentTask(
                role=role,
                goal=goal,
                system_prompt=system,
                # Explicit, not the dataclass default: SubagentTask.tools
                # defaults to a 4-item read-only list, and run_subagent only
                # falls back to SUBAGENT_AVAILABLE_TOOLS when tools is empty.
                tools=list(SUBAGENT_AVAILABLE_TOOLS),
                context={"workdir": call_workdir, "subagent_id": raw.get("_sid", "")},
            )
            tasks.append(task)

        # SubagentOrchestrator calls tool_executor_factory(role), not
        # factory(task), and it does so once per task in list order. Hand out
        # the matching task so two subagents with the same role stay separate.
        pending: Dict[str, List[SubagentTask]] = {}
        for t in tasks:
            pending.setdefault(t.role, []).append(t)

        def _make_executor(role: str):
            queue = pending.get(role) or []
            task = queue.pop(0) if queue else None
            # Per-subagent context so scratch/todo entries and queued images do
            # not bleed into the lead agent's context.
            sub_w = (task.context.get("workdir") if task and isinstance(task.context, dict) else None) or self.workdir
            sub_ctx = AgentContext(workdir=sub_w)
            sub_ctx.mode = mode
            sub_ctx.is_subagent = True

            def _exec(name: str, args: Dict[str, str]) -> Dict[str, Any]:
                n = (name or "").lower().strip()
                if n in ("shell", "terminal", "bash", "cmd", "sh", "run_command", "exec"):
                    n = "command"
                elif n in ("file_read", "read_file"):
                    n = "read"
                elif n in ("file_write", "write_file"):
                    n = "write"
                elif n in ("file_edit", "edit_file"):
                    n = "edit"
                if n in ("subagent", "spawn_subagent"):
                    return {"success": False, "error": "Subagents cannot spawn other subagents. Do your work directly."}
                if n in ("task", "tasks", "bg_task", "background_task"):
                    return {"success": False, "error": "Subagents cannot manage or spawn background tasks. Inspect files directly."}
                if n in ("todo", "tasks"):
                    n = "scratch"
                if n == "agent_note":
                    text = str(args.get("text") or args.get("note") or "").strip()
                    if text and task is not None:
                        task.notes.append(text)
                    return {"success": True, "note": text}
                if n == "ask":
                    # The user cannot answer mid-run and the lead agent is busy
                    # waiting, so ask would only pop a modal in the main window
                    # that the subagent then cannot read. Fail with the way out.
                    return {
                        "success": False,
                        "error": "You cannot ask anyone anything. Decide on your own and "
                                 "continue. If the lead agent needs to know something, put "
                                 "it in <tool:agent_note text=\"...\" />.",
                    }
                # notify=False: a subagent's own tools must not add markers to
                # the lead agent's chat or touch the main app's todo bar. Only the
                # one subagent marker shows.
                return self.execute_tool(n, args, ctx=sub_ctx, ask=False, notify=False)

            return _exec

        if is_local is None:
            provider = getattr(self, "provider_id", "local_gguf") or "local_gguf"
            is_local = provider in ("local_gguf", "local", "llama_cpp", "ollama")

        if is_local and (max_turns is None or max_turns > 6):
            max_turns = 6

        results = orch.execute_tasks(
            tasks,
            on_progress=on_progress,
            is_local=is_local,
            model_call=model_call,
            tool_executor=None,
            tool_executor_factory=_make_executor,
            max_turns=max_turns,
        )

        by_role: Dict[str, Any] = {t.role: t for t in tasks}
        agents: List[Dict[str, Any]] = []
        for r in results:
            task = by_role.get(r.role)
            history = r.history or []
            tool_count = sum(
                1 for m in history
                if m.get("role") == "user" and str(m.get("content", "")).startswith("[tool:")
            )
            server_tools_count = sum(
                1 for ev in (r.events or [])
                if isinstance(ev, dict)
                and bool(str(ev.get("tool", "") or "").strip())
                and ev.get("type") not in ("text", "thinking")
            )
            tool_count = max(tool_count, server_tools_count)
            all_failed = _all_tools_failed(r.events or [])
            # A subagent whose every tool errored did not do its task, whatever
            # its closing text claims. Trusting that text let "done, all fine"
            # reach the lead agent over a run where nothing was read.
            status = "failed" if (not r.success or all_failed) else "completed"
            if all_failed and r.success:
                summary = ("Every tool this subagent ran failed, so its report "
                           "could not be based on the workspace.\n"
                           + str(r.summary or "").strip())
            else:
                summary = r.summary
            agents.append({
                "role": r.role,
                "goal": getattr(task, "goal", "") if task else "",
                "status": status,
                "summary": summary,
                "findings": list(r.findings or []),
                "notes": list(r.notes or []),
                "tool_count": tool_count,
                "error": "" if status == "completed" else summary,
                "history": history,
                "events": list(r.events or []),
                "elapsed": float(r.elapsed or 0.0),
                "tools_all_failed": all_failed,
                "system_prompt": r.system_prompt,
                "tokens": int(r.tokens_in or 0) + int(r.tokens_out or 0),
            })

        total_tools = sum(a["tool_count"] for a in agents)
        total_notes = sum(len(a["notes"]) for a in agents)
        failed_n = sum(1 for a in agents if a["status"] == "failed")
        out: Dict[str, Any] = {
            "success": failed_n < len(agents),
            "subagents": agents,
            "count": len(agents),
            "tool_count": total_tools,
            "notes": [f"{a['role']}: {n}" for a in agents for n in a["notes"]],
            "is_local": bool(is_local),
        }
        if failed_n:
            # Only set the key when it carries a message. format_tool_result_for_llm
            # treats the mere presence of "error" as a tool failure.
            out["error"] = f"{failed_n} of {len(agents)} subagent(s) failed."
        return out

    def strip_xml_tool_calls(self, text: str) -> str:
        if not text:
            return ""
        clean = _truncate_hallucinated_turns(text)
        # Protect inline code mentions like `<tool:write>` from being stripped as real calls.
        inline_spans: List[str] = []
        def _stash_inline(m: re.Match) -> str:
            token = f"\x00INL{len(inline_spans)}\x00"
            inline_spans.append(m.group(0))
            return token
        clean = re.sub(r'`[^`\n]*<tool:[^`\n]*`', _stash_inline, clean, flags=re.IGNORECASE)
        try:
            dec = json.JSONDecoder()
            scan_pos = 0
            json_ranges = []
            while scan_pos < len(clean):
                idx_obj = clean.find("{", scan_pos)
                idx_arr = clean.find("[", scan_pos)
                candidates = [p for p in (idx_obj, idx_arr) if p != -1]
                if not candidates:
                    break
                start_idx = min(candidates)
                try:
                    parsed, end_offset = dec.raw_decode(clean[start_idx:])
                    full_end = start_idx + end_offset
                    calls = parsed if isinstance(parsed, list) else [parsed]
                    is_tool_chunk = False
                    for call_obj in calls:
                        if isinstance(call_obj, dict) and any(k in call_obj for k in ("tool_name", "tool", "name", "tool_call_id", "function")):
                            is_tool_chunk = True
                            break
                    if is_tool_chunk:
                        m_start = start_idx
                        m_end = full_end
                        pre = clean[:start_idx].rstrip()
                        if pre.endswith("```json") or pre.endswith("```"):
                            fence = pre.rfind("```")
                            if fence != -1:
                                m_start = fence
                        post = clean[full_end:].lstrip()
                        if post.startswith("```"):
                            m_end = full_end + clean[full_end:].find("```") + 3
                        json_ranges.append((m_start, m_end))
                        scan_pos = m_end
                    else:
                        scan_pos = start_idx + 1
                except Exception:
                    scan_pos = start_idx + 1
            for r_start, r_end in reversed(sorted(json_ranges, key=lambda r: r[0])):
                clean = clean[:r_start] + clean[r_end:]
        except Exception:
            pass

        clean = re.sub(
            r'<\|tool_call>call:\w+(?:\s+[^>]*?)?(?:\s*/>|>(.*?)</\|tool_call>)',
            '',
            clean,
            flags=re.DOTALL | re.IGNORECASE,
        )
        clean = re.sub(
            r'<tool:(\w+)(?:\s+[^>]*?)?\s*/?>.*?</(?:tool:)?(?:\1|tool)>',
            '',
            clean,
            flags=re.DOTALL | re.IGNORECASE,
        )
        clean = re.sub(r'<tool:\w+(?:\s+[^>]*?)?\s*/>', '', clean, flags=re.IGNORECASE)
        # A server-side subagent reports back as prose, not as an event: the
        # model writes `<subagent sessionID="..." state="completed">` into its
        # answer, and without this it was printed verbatim in the chat as if it
        # were something the user was meant to read. Same for the server's
        # `task` tool. The completion itself still closes the marker - that
        # travels as a `session.tool.success` event, not as this text.
        clean = re.sub(r'<subagent(?:\s+[^>]*?)?\s*/>', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'<subagent(?:\s+[^>]*?)?>.*?</subagent>', '', clean, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r'<task(?:\s+[^>]*?)?\s*/>', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'<task(?:\s+[^>]*?)?>.*?</task>', '', clean, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r'<tool_call(?:\s+[^>]*?)?(?:\s*/>|>.*?</tool_call>)', '', clean, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(
            r'<tool_response(?:\s+[^>]*?)?>[\s\S]*?(?:</tool_response\s*>|</toolIO\s*>|</IO\s*>|</ANTI\s*>|$)',
            '',
            clean,
            flags=re.IGNORECASE,
        )
        clean = re.sub(r'<tool:\w+(?:\s+[^>]*=\s*[^>]*>|\s*>\s*\n)[\s\S]*$', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'<\|tool_call.*$', '', clean, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r'<tool_call.*$', '', clean, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r'<(?:tool:?\w*|\|tool_call\w*|tool_call\w*|tool_response\w*)[^>]*$', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'</?tool:\w+[^>]*>', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'</?\|?tool_call[^>]*>', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'<subagent[^>]*$', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'<task[^>]*$', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'</?subagent[^>]*>', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'</?task[^>]*>', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'</?(?:tool_response|ASSISTANT|USER|ANTI|IO|toolIO|antml)[^>]*>', '', clean, flags=re.IGNORECASE)
        for idx, orig in enumerate(inline_spans):
            clean = clean.replace(f"\x00INL{idx}\x00", orig)
        return clean.strip()

    def format_tool_result_for_llm(self, tool_name: str, res: Dict[str, Any]) -> str:
        try:
            return self._format_tool_result_impl(tool_name, res)
        except Exception as exc:
            import json
            name = tool_name.lower().strip()
            try:
                dumped = json.dumps(res, default=str, ensure_ascii=False)
            except Exception:
                dumped = str(res)
            return f'<tool_response name="{name}">\n{dumped}\n</tool_response>\n[Proceed with the next step or final answer based on the tool result. Do NOT repeat previous messages.]'

    def _format_tool_result_impl(self, tool_name: str, res: Dict[str, Any]) -> str:
        name = tool_name.lower().strip()

        # A subagent batch always goes through the full formatter below, even
        # when it failed: its per-agent block carries the reports, the notes and
        # the reason, and "ERROR: 1 of 1 subagent(s) failed." told the lead
        # agent nothing it could act on.
        if name == "subagent":
            pass
        elif not res.get("success", True) or "error" in res:
            err = res.get("error") or res.get("stderr") or "Execution failed"
            missing = res.get("missing")
            example = res.get("example")
            if missing and example:
                return f'<tool_response name="{name}">ERROR: missing {missing}. Use: {example}</tool_response>'
            elif missing:
                return f'<tool_response name="{name}">ERROR: missing {missing}</tool_response>'
            lines = [f'<tool_response name="{name}">']
        if name == "web":
            results = res.get("results", [])
            if results:
                lines.append(f"Web search top results for: \"{res.get('query', '')}\"")
                for i, r in enumerate(results[:5], 1):
                    title = r.get("title", "")
                    url = r.get("url", "")
                    snippet = r.get("snippet", "").strip()
                    if len(snippet) > 180:
                        snippet = snippet[:177] + "..."
                    lines.append(f"{i}. {title}\n   URL: {url}\n   Snippet: {snippet}")
            elif res.get("content"):
                text = str(res["content"]).strip()
                if len(text) > 1200:
                    text = text[:1150] + "\n... [Remaining web content truncated to conserve context tokens]"
                lines.append(text)
        elif name == "screenshot":
            if res.get("path"):
                lines.append(f"Screenshot saved: {res['path']} ({res.get('size_bytes', 0)} bytes)")
                lines.append("To analyze it, call <tool:vision ... /> with this image_path.")
            else:
                lines.append(f"Screenshot failed: {res.get('error', 'unknown')}")
        elif name == "vision":
            if res.get("path"):
                lines.append(f"Image queued: {res['path']}")
                if res.get("question"):
                    lines.append(f"Question: {res['question']}")
            else:
                lines.append(f"Vision failed: {res.get('error', 'unknown')}")
        elif name == "subagent":
            if res.get("background"):
                roles_str = ", ".join(res.get("roles") or ["Subagent"])
                lines.append(
                    f"Subagent(s) [{roles_str}] are now running in the background (task_id: {res.get('task_id', '')}).\n"
                    "Inform the user that the subagent is working in the background and continue with other work or finish your turn. "
                    "As soon as the background subagent finishes, its report will be automatically delivered to you so you can present the findings to the user."
                )
            else:
                # The subagent's final report comes back as the tool result - that is
                # the data channel, the same shape OpenCode's task tool uses. What
                # stays out is the transcript and the raw tool output, which is what
                # actually keeps the lead agent's context small: a subagent that
                # read forty files hands over its conclusions, not the files.
                agents = res.get("subagents", [])
                if not agents:
                    lines.append("No subagent produced a result.")
                else:
                    names = [str(a.get("role", "Subagent")) for a in agents]
                    failed = [str(a.get("role", "Subagent")) for a in agents
                              if str(a.get("status", "")) == "failed"]
                    lines.append(f"{len(agents)} subagent(s) finished: {', '.join(names)}.")
                    if failed:
                        lines.append(f"Failed: {', '.join(failed)}.")
                    notes = res.get("notes") or []
                    if notes:
                        lines.append("Notes they left for you:")
                        for nt in notes:
                            lines.append(f"  - {nt}")
                    for a in agents:
                        summary = str(a.get("summary", "") or "").strip()
                        if not summary:
                            continue
                        role = str(a.get("role", "Subagent"))
                        if len(summary) > 1800:
                            summary = summary[:1800] + "\n... [report truncated]"
                        lines.append(f"\nReport from {role}:")
                        lines.append(summary)
                    if not any(str(a.get("summary", "") or "").strip() for a in agents):
                        lines.append("None of them wrote a report. Their work is not in your context: "
                                     "do not assume what they found, check the files yourself.")
                    lines.append("Present the subagent findings clearly to the user in your own words now. "
                                 "Do NOT spawn the same subagent again and do NOT call any more tools unless a file edit is needed.")
        elif name == "mcp":
            if res.get("error"):
                lines.append(f"Error: {res.get('error')}")
            servers = res.get("servers", [])
            if servers:
                lines.append("Available MCP servers:")
                for s in servers:
                    if isinstance(s, dict):
                        status = s.get("status", "unknown")
                        count = s.get("tool_count", 0)
                        usable = s.get("usable", False)
                        perms = s.get("tools_by_permission") or {}
                        perm_str = f" [read: {perms.get('read', 0)}, write: {perms.get('write', 0)}, destructive: {perms.get('destructive', 0)}]" if perms else ""
                        lines.append(f"- Server '{s.get('name')}': status={status}, usable={usable}, {count} tools{perm_str}")
                    else:
                        lines.append(f"- {s}")
            if res.get("schema_text"):
                # The whole point: schemas live in the transcript, so the
                # prompt prefix stays identical between turns.
                lines.append(str(res["schema_text"]))
            if res.get("note"):
                lines.append(str(res["note"]))
        elif name.startswith(TOOL_NAME_PREFIX):
            if res.get("text"):
                lines.append(res["text"])
            if res.get("structured") is not None:
                lines.append(json.dumps(res["structured"], ensure_ascii=False, indent=2))
            if res.get("images"):
                lines.append(f"({len(res['images'])} image(s) attached to this request.)")
            if not res.get("text") and res.get("structured") is None and not res.get("images"):
                lines.append(f"{res.get('tool')} returned no content.")
        else:
            clean_res = {k: v for k, v in res.items() if k not in ("diff_entries",)}
            lines.append(json.dumps(clean_res, indent=2, ensure_ascii=False))

        lines.append('</tool_response>')
        lines.append('[Proceed with the next step or final answer based on the tool result. Do NOT repeat previous messages.]')
        return "\n".join(str(l) if not isinstance(l, str) else l for l in lines)


# The checklist rule is the same in both writing modes, so it is written once.
# It is what turns a plan into work: without it a plan made in PLAN mode sits in
# the context and the agent starts the next task from scratch.
_CHECKLIST_RULE = (
    "- If a checklist exists (from PLAN mode), it is your work order: execute it step "
    "by step, mark each finished step with <tool:todo action=\"done\" index=\"N\" />, "
    "and do not end the turn while steps remain.\n"
)

# The tool list used to be the same in every mode, which told a model in PLAN
# mode that Write File was available. It called it, the app refused, and the turn
# ended in an apology instead of a plan - the symptom was a model that described
# the app it wanted to build instead of planning it. The list now matches what
# `execute_tool` actually allows, and names what is blocked.
_PLAN_TOOLS_BLOCK = (
    "AVAILABLE TOOLS IN THIS MODE (use these):\n"
    "1. Web Search: <tool:web query=\"docs or query or https://url\" />\n"
    "2. Local Models Inspect: <tool:models action=\"list|ps\" />\n"
    "3. Global Memory Read: <tool:memory action=\"list\" />\n"
    "4. Plan/Todo Tracker: <tool:todo action=\"set\" todos=\"step one|step two|step three\" /> - "
    "that is how you record the plan. 'set' replaces the checklist, 'add' appends one "
    "step, 'done' marks a finished step.\n"
    "5. CMDAI-CODE Inspect/Plan: <tool:cmdai_code action=\"status|set_mode|run\" mode=\"plan\" workdir=\"path\">Write a comprehensive multi-sentence prompt describing what to inspect and analyze (NEVER send bare shell commands like 'dir' or 'ls').</tool:cmdai_code>\n"
    "6. MCP Servers: <tool:mcp /> / <tool:mcp open server=\"NAME\" />\n"
)

_PLAN_DIRECTIVES_BLOCK = (
    "\nEXECUTION DIRECTIVES:\n"
    "- CRITICAL (ANTI-STALLING): You are an autonomous agent, NOT a chatbot. Never output only a statement of intent without immediately invoking the required tool in the same response.\n"
    "- INTERLEAVED EXPLANATIONS & ACTIONS: say in one sentence what you are looking at "
    "before each tool call, so the user can follow the investigation.\n"
    "- STRUCTURE: Explain briefly what you are checking before each <tool:...>, then summarize what you found and present the plan. Do NOT dump a wall of tools without explanations, and never output literal bracket placeholders.\n"
    "- ONE CHECKLIST CALL: record the plan with a single <tool:todo action=\"set\" ... /> "
    "call, after the investigation and before your summary.\n"
    "- STOP AFTER THE PLAN: no tool call after the checklist, and no second checklist.\n"
)


def _full_tools_block(vision: bool) -> str:
    """The complete tool list for CMDAI (Ollama runtime & full CMDAI-CODE controller)."""
    return (
        "AVAILABLE TOOLS:\n"
        "1. Local Models Runtime: <tool:models action=\"list|ps|load|unload\" model=\"filename.gguf\" /> (inspect, load, or unload local GGUF models)\n"
        "2. CMDAI-CODE Full Controller (persistent autonomous AI software engineer in its live TUI window): "
        "<tool:cmdai_code action=\"run|command|set_mode|approve|reject|interrupt|status|open|reset|close\" workdir=\"path\" mode=\"auto|plan|code\">Detailed natural-language engineering task or TUI slash command</tool:cmdai_code> "
        "(CMDAI-CODE already has its own file, search, edit, and terminal tools: ls, read, glob, search, edit, write, command. "
        "CRITICAL: For 'run', ALWAYS write a rich, multi-sentence natural-language engineering prompt describing the full goal, context, files/requirements, and expected verification — "
        "NEVER send bare shell commands like 'dir', 'ls', 'cat', 'type', or 'tree' as a task. "
        "Use 'command' ONLY for TUI slash commands such as '/diff', '/git', '/commit' or explicit test runners like 'pytest'; "
        "'set_mode' switches auto/plan/code, 'approve'/'reject' confirms or reverts diffs, 'interrupt' stops generation, 'status' inspects modified_files/todo_list/state, 'open'/'reset'/'close' manages the window/session)\n"
        "3. Web Search: <tool:web query=\"docs or query or https://url\" />\n"
        "4. Global Memory: <tool:memory action=\"add|list|remove|clear\" content=\"concise fact\" /> (saves cross-session user facts and preferences globally in ~/.cmdai/memory.md)\n"
        "5. Plan/Todo Tracker: <tool:todo action=\"add|done|set\" note=\"step description\" /> - "
        "'set' replaces the checklist, 'add' appends one step, 'done' marks a finished step.\n"
        "6. Ask User: <tool:ask question=\"clarifying question\" options=\"Opt1|Opt2\" />\n"
        "7. MCP Servers: <tool:mcp /> lists the configured servers. Then "
        "<tool:mcp open server=\"NAME\" /> returns that server's tools, their arguments "
        "and their exact call names. Call those tools as "
        "<tool:mcp__NAME__TOOL>{\"arg\": \"value\"}</tool:mcp__NAME__TOOL> with a JSON "
        "object between the tags. Only the two `mcp` tools are listed here on purpose: "
        "server tools appear in context only after you open a server, so you never have to "
        "guess a name.\n"
        + (
            "8. Screenshot: <tool:screenshot target=\"active_window\" /> (captures an OS window title or 'active_window'/'desktop' to PNG)\n"
            "9. Vision: <tool:vision image_path=\"path.png\" question=\"what to check\" /> (attaches the image to your next request)\n"
            if vision else ""
        )
    )


_FULL_DIRECTIVES_BLOCK = (
    "\nEXECUTION DIRECTIVES:\n"
    "- CRITICAL (ANTI-STALLING): You are an autonomous execution engine, NOT a conversational chatbot. Never output only an announcement of what you plan to do without immediately emitting the corresponding <tool:...> call in the exact same response.\n"
    "- INTERLEAVED EXPLANATIONS & ACTIONS: Always include a brief 1-2 sentence explanation before and between your tool actions, followed by the tool call.\n"
    "- STRUCTURE: Write natural prose before each <tool:...> call, then explain the next finding or final outcome. Do NOT dump a wall of tools without explanations, and never output literal bracket placeholders.\n"
    "- CODING DELEGATION & CONTROL: CMDAI acts as the orchestrator and uses `<tool:cmdai_code>` as its persistent coding subagent in a live CMDAI-CODE TUI window. "
    "ALWAYS give `<tool:cmdai_code action=\"run\">` a well-structured, detailed, multi-sentence natural-language engineering prompt with full goals, context, and deliverables — NEVER delegate bare shell commands like `dir`, `ls`, `cat`, or `type`. "
    "Follow-up `run` calls in the same `workdir` continue the same conversation; use `status` to inspect `modified_files` and `todo_list`, `command` ONLY for `/diff` or explicit test commands, `set_mode` for `auto`/`plan`/`code`, and `approve`/`reject` when reviewing changes.\n"
    "- Be fast, direct, and proactive: execute the necessary tool calls to complete the user request.\n"
)


def load_project_instructions(workdir: str = ".") -> str:
    """Read global cross-session memory and enabled plugin rules."""
    sections: List[str] = []
    try:
        from ..core.memory import format_memory_hint

        mem_hint = format_memory_hint(workdir or ".")
        if mem_hint:
            sections.append(f"\n\n### GLOBAL PERSISTENT MEMORY:\n{mem_hint}\n")
    except Exception:
        pass
    try:
        from ..core.plugins import get_enabled_plugin_rules

        plug_block = get_enabled_plugin_rules(workdir or ".")
        if plug_block:
            sections.append(f"\n\n### ENABLED PLUGIN RULES:\n{plug_block}\n")
    except Exception:
        pass
    return "".join(sections)


def get_agent_system_prompt(workdir: str = ".", mode: str = "auto", vision: bool = False) -> str:
    mode_clean = (mode or "auto").strip().lower()

    if mode_clean == "chat":
        return ""

    if mode_clean == "ask":
        mode_section = (
            "CURRENT EXECUTION MODE: ASK\n"
            "- In ASK mode, inspect safely and ask user confirmation before loading/unloading models or delegating CMDAI-CODE tasks.\n"
            + _CHECKLIST_RULE
        )
        tools_block = _full_tools_block(vision)
        directives_block = _FULL_DIRECTIVES_BLOCK
    else:
        mode_section = (
            "CURRENT EXECUTION MODE: AUTO\n"
            "- In AUTO mode, you autonomously assist the user, control local GGUF models (`<tool:models>`), and orchestrate coding tasks in CMDAI-CODE (`<tool:cmdai_code>`).\n"
            + _CHECKLIST_RULE
        )
        tools_block = _full_tools_block(vision)
        directives_block = _FULL_DIRECTIVES_BLOCK

    extra_rules = load_project_instructions(workdir)
    return (
        f"\n\n### CMDAI AUTONOMOUS ASSISTANT & CONTROLLER ({mode_clean.upper()} MODE)\n"
        f"{mode_section}\n\n"
        f"{tools_block}"
        f"{directives_block}"
        f"{extra_rules}"
    )
