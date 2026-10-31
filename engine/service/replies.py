"""What every tool call answers.

Messages are short and written so the agent can repeat them truthfully.
Short replies also keep the AI's reading cost and delay down.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# Each status is also the `outcome` written in the action log.
OK = "ok"
REFUSED = "refused"
ALREADY_DONE = "already_done"
EXPIRED = "expired"
ERROR = "error"


@dataclass(frozen=True)
class ToolReply:
    status: str
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        """Errors (failures, timeouts) show up as tool errors in MCP.
        A refusal is not an error: the service worked and said no."""
        return self.status == ERROR

    def to_json(self) -> str:
        reply: dict[str, Any] = {"status": self.status, "message": self.message}
        if self.data:
            # Pack data sits under its own key, so a pack field (like an order's
            # own "status") can never overwrite the reply's status.
            reply["data"] = self.data
        return json.dumps(reply, ensure_ascii=False, separators=(",", ":"))  # no spaces: fewer tokens
