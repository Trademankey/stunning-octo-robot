from __future__ import annotations

import json
from typing import Any
from urllib import request

from ..config import Settings
from ..store import ConversationStore


class LeadCaptureService:
    def __init__(self, settings: Settings, store: ConversationStore):
        self.webhook_url = settings.lead_webhook_url
        self.store = store

    async def capture(self, kind: str, customer_id: str, language: str, payload: dict[str, Any]) -> int:
        lead_id = self.store.save_lead(kind, customer_id, language, payload)
        if self.webhook_url:
            body = json.dumps(
                {
                    "lead_id": lead_id,
                    "kind": kind,
                    "customer_id": customer_id,
                    "language": language,
                    "payload": payload,
                }
            ).encode("utf-8")
            req = request.Request(
                self.webhook_url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=15.0):
                pass
        return lead_id
