"""Which tools the AI sees, and how they look (decision 5).

plain mode: every action is one tool, and it acts at once.
safe mode:  safe actions stay as they are. Every risky action becomes
            `prepare_<name>`, plus ONE shared `confirm` tool for all of them.

Fewer, clearer tools mean fewer wrong picks, which is why confirm is shared.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from engine.contract.action import STRING, Action, Input
from engine.storage.call_settings import SAFE, CallSettings

CONFIRM = "confirm"
PREPARE_PREFIX = "prepare_"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


def confirm_inputs(ticket_check: bool) -> dict[str, Input]:
    """The ticket input only exists when the ticket check is on, so an agent
    without a Guard (ticket check off) is never asked for something it can't have."""
    inputs = {"token": Input(STRING, "The token that prepare gave back.")}
    if ticket_check:
        inputs["ticket"] = Input(STRING, "The yes-ticket from the Guard for this token.")
    return inputs


def build_catalog(actions: list[Action], settings: CallSettings) -> list[ToolSpec]:
    safe = settings.mode == SAFE
    tools = []
    for action in actions:
        if safe and action.risky:
            tools.append(
                ToolSpec(
                    PREPARE_PREFIX + action.name,
                    f"Step 1 of 2. {action.description} Changes nothing yet: returns a token"
                    " and a summary. Read the summary to the caller and get a clear yes.",
                    input_schema(action.inputs),
                )
            )
        else:
            tools.append(ToolSpec(action.name, action.description, input_schema(action.inputs)))
    if safe:
        description = "Step 2 of 2. Do a prepared action, only after the caller clearly said yes to its summary."
        if settings.ticket_check:
            description += " Needs the yes-ticket from the Guard."
        tools.append(ToolSpec(CONFIRM, description, input_schema(confirm_inputs(settings.ticket_check))))
    return tools


def input_schema(inputs: dict[str, Input]) -> dict[str, Any]:
    """Turn the action's inputs into the JSON schema that MCP shows the AI."""
    properties = {}
    for name, spec in inputs.items():
        prop: dict[str, Any] = {"type": spec.type, "description": spec.description}
        if spec.choices:
            prop["enum"] = list(spec.choices)
        properties[name] = prop
    return {
        "type": "object",
        "properties": properties,
        "required": [name for name, spec in inputs.items() if spec.required],
        "additionalProperties": False,
    }
