from __future__ import annotations

import json
from typing import Any
from urllib import request

from ..config import Settings
from ..types import AgentReply, InboundMessage, ReplyOption


class WhatsAppService:
    def __init__(self, settings: Settings):
        self.access_token = settings.whatsapp_access_token
        self.phone_number_id = settings.whatsapp_phone_number_id
        self.verify_token = settings.whatsapp_verify_token
        self.graph_version = settings.whatsapp_graph_version

    def verify_webhook(self, mode: str, token: str, challenge: str) -> tuple[bool, str]:
        if mode == "subscribe" and token == self.verify_token:
            return True, challenge
        return False, "verification failed"

    def parse_inbound_messages(self, payload: dict[str, Any]) -> list[InboundMessage]:
        results: list[InboundMessage] = []
        for entry in payload.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                contacts = value.get("contacts", [])
                profile_name = contacts[0].get("profile", {}).get("name", "") if contacts else ""
                for message in value.get("messages", []):
                    message_type = message.get("type", "text")
                    text = ""
                    if message_type == "text":
                        text = message.get("text", {}).get("body", "")
                    elif message_type == "interactive":
                        interactive = message.get("interactive", {})
                        if interactive.get("type") == "button_reply":
                            reply = interactive.get("button_reply", {})
                            text = reply.get("id") or reply.get("title", "")
                        elif interactive.get("type") == "list_reply":
                            reply = interactive.get("list_reply", {})
                            text = reply.get("id") or reply.get("title", "")
                    elif message_type == "image":
                        text = "[image]"
                    if not text:
                        continue
                    results.append(
                        InboundMessage(
                            customer_id=message.get("from", ""),
                            text=text,
                            message_type=message_type,
                            customer_name=profile_name,
                        )
                    )
        return results

    async def send_reply(self, to: str, reply: AgentReply) -> dict[str, Any]:
        payload = self._build_payload(to, reply)
        if not (self.access_token and self.phone_number_id):
            return {"preview": True, "payload": payload}

        url = f"https://graph.facebook.com/{self.graph_version}/{self.phone_number_id}/messages"
        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=15.0) as response:
            return json.loads(response.read().decode("utf-8"))

    def _build_payload(self, to: str, reply: AgentReply) -> dict[str, Any]:
        base = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
        }
        if reply.list_options:
            return {
                **base,
                "type": "interactive",
                "interactive": {
                    "type": "list",
                    "body": {"text": reply.text},
                    "action": {
                        "button": reply.list_button_text,
                        "sections": [
                            {
                                "title": "Ali Chalmer",
                                "rows": [self._row(option) for option in reply.list_options],
                            }
                        ],
                    },
                },
            }
        if reply.buttons:
            return {
                **base,
                "type": "interactive",
                "interactive": {
                    "type": "button",
                    "body": {"text": reply.text},
                    "action": {
                        "buttons": [
                            {
                                "type": "reply",
                                "reply": {"id": option.id, "title": option.title},
                            }
                            for option in reply.buttons[:3]
                        ]
                    },
                },
            }
        return {**base, "type": "text", "text": {"body": reply.text}}

    @staticmethod
    def _row(option: ReplyOption) -> dict[str, str]:
        row = {"id": option.id, "title": option.title}
        if option.description:
            row["description"] = option.description
        return row
