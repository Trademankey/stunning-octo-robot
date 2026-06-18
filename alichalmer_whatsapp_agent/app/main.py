from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query, Request

from .agent import AliChalmerAgent
from .config import get_settings
from .knowledge_base import KnowledgeBase
from .services.lead_capture import LeadCaptureService
from .services.whatsapp import WhatsAppService
from .services.woocommerce import WooCommerceService
from .store import ConversationStore
from .types import InboundMessage

settings = get_settings()
store = ConversationStore(settings.database_url)
knowledge_base = KnowledgeBase(settings)
woo_service = WooCommerceService(settings)
lead_capture = LeadCaptureService(settings, store)
whatsapp_service = WhatsAppService(settings)
agent = AliChalmerAgent(settings, store, knowledge_base, woo_service, lead_capture)

app = FastAPI(title="AliChalmer WhatsApp Agent", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(alias="hub.mode"),
    hub_token: str = Query(alias="hub.verify_token"),
    hub_challenge: str = Query(alias="hub.challenge"),
) -> str:
    verified, response = whatsapp_service.verify_webhook(hub_mode, hub_token, hub_challenge)
    if not verified:
        raise HTTPException(status_code=403, detail=response)
    return response


@app.post("/webhook")
async def inbound_webhook(request: Request) -> dict:
    payload = await request.json()
    inbound_messages = whatsapp_service.parse_inbound_messages(payload)

    processed: list[dict] = []
    for inbound in inbound_messages:
        replies = await agent.handle_message(inbound)
        delivery_results = []
        for reply in replies:
            delivery_results.append(await whatsapp_service.send_reply(inbound.customer_id, reply))
        processed.append(
            {
                "customer_id": inbound.customer_id,
                "message_text": inbound.text,
                "reply_count": len(replies),
                "delivery_results": delivery_results,
            }
        )
    return {"processed": processed}


@app.post("/preview")
async def preview_message(payload: dict) -> dict:
    inbound = InboundMessage(
        customer_id=str(payload.get("customer_id", "preview-user")),
        text=str(payload.get("message", "")),
        message_type="text",
    )
    replies = await agent.handle_message(inbound)
    return {
        "replies": [
            {
                "text": reply.text,
                "buttons": [button.title for button in reply.buttons],
                "list_options": [option.title for option in reply.list_options],
            }
            for reply in replies
        ]
    }
