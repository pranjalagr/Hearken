"""Code copies every backend answer into the notebook, as it came (decision 7).

The Talker only ever says what a note says, so the note must be the backend's own
answer, not the Thinker's retelling: in Agent A's practice run 7 the brain said 4535
for 4533 and "ho gaya" for a change the backend never saved (15 Nov 2026).
"""

from __future__ import annotations

import json

from agents.agent_c.notebook import ANSWER, FACT, NEED_INFO, NEED_YES, STILL_GOOD
from agents.agent_c.thinker.brain import ToolCall
from engine.service.replies import OK, ToolReply
from engine.service.tool_catalog import PREPARE_PREFIX

NEED = "need:"
STILL_GOOD_WORDS = "still good"


def note_for_reply(call: ToolCall, reply: ToolReply) -> tuple[str, str, dict]:
    """(kind, text, data) for one backend answer. A prepared change becomes a
    "need yes" note carrying its token for the Guard; everything else is a fact."""
    data = {"tool": call.name, "inputs": dict(call.inputs), "status": reply.status, "data": dict(reply.data)}
    if call.name.startswith(PREPARE_PREFIX) and reply.status == OK and reply.data.get("token"):
        data["action"] = call.name[len(PREPARE_PREFIX):]
        return NEED_YES, str(reply.data.get("summary") or reply.message), data
    text = f"{call.name}: {reply.status}. {reply.message}"
    if reply.data:
        text += " " + json.dumps(reply.data, ensure_ascii=False, separators=(",", ":"))
    return FACT, text, data


def note_for_stop(text: str) -> tuple[str, str] | None:
    """(kind, text) for the Thinker's own closing line, or None when it wrote nothing."""
    words = text.strip()
    if not words:
        return None
    if words.lower().rstrip(".!") == STILL_GOOD_WORDS:
        return STILL_GOOD, words
    if words.lower().startswith(NEED):
        return NEED_INFO, words[len(NEED):].strip()
    return ANSWER, words
