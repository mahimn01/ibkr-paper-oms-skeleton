from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]
    call_id: str | None = None


@dataclass(frozen=True)
class ChatModelReply:
    assistant_message: str
    tool_calls: list[ToolCall]


def parse_chat_model_reply(text: str) -> ChatModelReply:
    """
    Expected JSON:
      {
        "assistant_message": "string",
        "tool_calls": [
          {"id": "optional", "name": "tool_name", "args": {...}}
        ]
      }
    If parsing fails, treat entire text as `assistant_message` with no tool calls.
    """
    raw = str(text or "").strip()
    raw = _strip_code_fences(raw)
    try:
        obj = json.loads(raw)
    except Exception:
        return ChatModelReply(assistant_message=str(text or ""), tool_calls=[])

    if not isinstance(obj, dict):
        return ChatModelReply(assistant_message=str(text or ""), tool_calls=[])

    msg = obj.get("assistant_message")
    assistant_message = str(msg) if msg is not None else ""
    tool_calls_raw = obj.get("tool_calls") or []
    tool_calls: list[ToolCall] = []
    if isinstance(tool_calls_raw, list):
        for item in tool_calls_raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            args = item.get("args")
            if not isinstance(args, dict):
                args = {}
            call_id = item.get("id")
            tool_calls.append(ToolCall(name=name, args=args, call_id=str(call_id) if call_id is not None else None))

    return ChatModelReply(assistant_message=assistant_message, tool_calls=tool_calls)


def format_tool_result_for_model(*, call: ToolCall, ok: bool, result: Any) -> str:
    payload = {
        "tool_result": {
            "id": call.call_id,
            "name": call.name,
            "ok": bool(ok),
            "result": result,
        }
    }
    return json.dumps(payload, sort_keys=True)


def _strip_code_fences(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return text

