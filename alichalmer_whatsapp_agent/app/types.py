from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ReplyOption:
    id: str
    title: str
    description: str = ""


@dataclass
class AgentReply:
    text: str
    buttons: list[ReplyOption] = field(default_factory=list)
    list_options: list[ReplyOption] = field(default_factory=list)
    list_button_text: str = "Open menu"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationState:
    customer_id: str
    language: Optional[str] = None
    active_flow: Optional[str] = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class InboundMessage:
    customer_id: str
    text: str
    message_type: str = "text"
    customer_name: str = ""
