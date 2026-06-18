from __future__ import annotations

import base64
import json
from typing import Any, Optional
from urllib import error, request

from ..config import Settings


class WooCommerceService:
    def __init__(self, settings: Settings):
        self.base_url = settings.woo_base_url.rstrip("/")
        self.consumer_key = settings.woo_consumer_key
        self.consumer_secret = settings.woo_consumer_secret

    @property
    def enabled(self) -> bool:
        return bool(self.consumer_key and self.consumer_secret and self.base_url)

    async def get_order(self, order_id: str) -> Optional[dict[str, Any]]:
        if not self.enabled:
            return None

        url = f"{self.base_url}/wp-json/wc/v3/orders/{order_id}"
        token = base64.b64encode(
            f"{self.consumer_key}:{self.consumer_secret}".encode("utf-8")
        ).decode("ascii")
        req = request.Request(
            url,
            headers={
                "Accept": "application/json",
                "Authorization": f"Basic {token}",
            },
        )
        try:
            with request.urlopen(req, timeout=15.0) as response:
                data = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            if exc.code == 404:
                return {"found": False}
            raise

        tracking = ""
        for meta in data.get("meta_data", []):
            key = str(meta.get("key", "")).lower()
            if key in {
                "_tracking_number",
                "tracking_number",
                "shipment_tracking_number",
                "_aftership_tracking_number",
            }:
                tracking = str(meta.get("value", "")).strip()
                if tracking:
                    break

        return {
            "found": True,
            "id": str(data.get("id", order_id)),
            "status": str(data.get("status", "")).lower(),
            "billing_email": data.get("billing", {}).get("email", ""),
            "billing_phone": data.get("billing", {}).get("phone", ""),
            "tracking": tracking,
        }
