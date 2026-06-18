from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

from sqlalchemy import text

TESTS_DIR = Path(__file__).resolve().parent
SERVICE_ROOT = TESTS_DIR.parent
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.agent import AliChalmerAgent
from app.config import Settings
from app.knowledge_base import KnowledgeBase
from app.services.lead_capture import LeadCaptureService
from app.services.woocommerce import WooCommerceService
from app.store import ConversationStore
from app.types import InboundMessage


class FakeWooCommerceService(WooCommerceService):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self._order = None

    async def get_order(self, order_id: str):
        return self._order


class AgentFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        service_root = Path(self.temp_dir.name)
        knowledge_src = SERVICE_ROOT / "knowledge"
        knowledge_dst = service_root / "knowledge"
        knowledge_dst.mkdir(parents=True, exist_ok=True)
        for file_path in knowledge_src.glob("*.md"):
            (knowledge_dst / file_path.name).write_text(file_path.read_text(encoding="utf-8"), encoding="utf-8")

        self.settings = Settings(
            service_root=str(service_root),
            database_path=str(service_root / "data" / "agent_state.db"),
            database_url=f"sqlite:///{service_root / 'data' / 'agent_state.db'}",
            whatsapp_verify_token="token",
            whatsapp_access_token="",
            whatsapp_phone_number_id="",
            whatsapp_graph_version="v23.0",
            woo_base_url="https://alichalmer.com",
            woo_consumer_key="",
            woo_consumer_secret="",
            coffee_product_url="https://alichalmer.com/coffee",
            wholesale_page_url="https://alichalmer.com/wholesale",
            lead_webhook_url="",
        )
        self.store = ConversationStore(self.settings.database_url)
        self.kb = KnowledgeBase(self.settings)
        self.woo = FakeWooCommerceService(self.settings)
        self.leads = LeadCaptureService(self.settings, self.store)
        self.agent = AliChalmerAgent(self.settings, self.store, self.kb, self.woo, self.leads)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_prompts_for_language_when_uncertain(self) -> None:
        replies = asyncio.run(
            self.agent.handle_message(InboundMessage(customer_id="u1", text="yo"))
        )
        self.assertIn("coffee shops", replies[0].text)
        self.assertIn("Reply with English or Español", replies[0].text)

    def test_coffee_question_returns_product_link(self) -> None:
        asyncio.run(self.agent.handle_message(InboundMessage(customer_id="u2", text="English")))
        replies = asyncio.run(
            self.agent.handle_message(InboundMessage(customer_id="u2", text="What coffee do you sell?"))
        )
        self.assertIn("Xirandél Dark", replies[0].text)
        self.assertIn("Peligroso 19", replies[0].text)
        self.assertIn("https://alichalmer.com/coffee", replies[0].text)

    def test_sample_flow_captures_coffee_shop_lead(self) -> None:
        asyncio.run(self.agent.handle_message(InboundMessage(customer_id="u3", text="Hola")))
        asyncio.run(self.agent.handle_message(InboundMessage(customer_id="u3", text="Español")))
        asyncio.run(self.agent.handle_message(InboundMessage(customer_id="u3", text="Quiero muestras para mi cafeteria")))
        asyncio.run(self.agent.handle_message(InboundMessage(customer_id="u3", text="Ana")))
        asyncio.run(self.agent.handle_message(InboundMessage(customer_id="u3", text="Cafeteria Niebla")))
        asyncio.run(self.agent.handle_message(InboundMessage(customer_id="u3", text="Xalapa, Mexico")))
        asyncio.run(self.agent.handle_message(InboundMessage(customer_id="u3", text="Cafeteria con 1 sucursal")))
        asyncio.run(self.agent.handle_message(InboundMessage(customer_id="u3", text="80 bolsas al mes")))
        asyncio.run(self.agent.handle_message(InboundMessage(customer_id="u3", text="Muestra de los tres")))
        asyncio.run(self.agent.handle_message(InboundMessage(customer_id="u3", text="ana@example.com")))
        replies = asyncio.run(self.agent.handle_message(InboundMessage(customer_id="u3", text="+52 555 111 2222")))
        self.assertIn("información de tu cafetería", replies[0].text)

        with self.store._engine.connect() as conn:
            row = conn.execute(text("select kind, payload_json from leads")).mappings().first()
        self.assertEqual("coffee_shop_wholesale", row["kind"])
        self.assertIn("monthly_volume", row["payload_json"])

    def test_shipping_reply_is_for_usa_and_mexico_shops(self) -> None:
        asyncio.run(self.agent.handle_message(InboundMessage(customer_id="u4", text="English")))
        replies = asyncio.run(
            self.agent.handle_message(InboundMessage(customer_id="u4", text="Can you ship to Texas USA?"))
        )
        self.assertIn("coffee shops in the United States", replies[0].text)
        self.assertEqual(["Wholesale", "Samples", "Sales"], [button.title for button in replies[0].buttons])

    def test_unknown_question_clarifies_then_escalates(self) -> None:
        asyncio.run(self.agent.handle_message(InboundMessage(customer_id="u5", text="English")))
        first = asyncio.run(self.agent.handle_message(InboundMessage(customer_id="u5", text="Tell me about the books")))
        second = asyncio.run(self.agent.handle_message(InboundMessage(customer_id="u5", text="I need order status")))
        self.assertIn("only helps coffee shops", first[0].text)
        self.assertIn("sales team", second[0].text)


if __name__ == "__main__":
    unittest.main()
