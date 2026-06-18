from __future__ import annotations

from typing import Optional

from .config import Settings
from .intents import IntentMatch, detect_intent
from .knowledge_base import KnowledgeBase
from .language import detect_language, normalize_language_choice, normalize_text
from .services.lead_capture import LeadCaptureService
from .services.woocommerce import WooCommerceService
from .store import ConversationStore
from .types import AgentReply, ConversationState, InboundMessage, ReplyOption


class AliChalmerAgent:
    def __init__(
        self,
        settings: Settings,
        store: ConversationStore,
        knowledge_base: KnowledgeBase,
        woo_service: WooCommerceService,
        lead_capture: LeadCaptureService,
    ):
        self.settings = settings
        self.store = store
        self.knowledge_base = knowledge_base
        self.woo_service = woo_service
        self.lead_capture = lead_capture

    async def handle_message(self, message: InboundMessage) -> list[AgentReply]:
        state = self.store.get_conversation(message.customer_id)
        language_choice = normalize_language_choice(message.text)
        if language_choice:
            state.language = language_choice
            state.active_flow = None
            state.context = {}
            self.store.save_conversation(state)
            return [self._language_confirmed_reply(language_choice), self._main_menu_reply(language_choice)]

        if not state.language:
            detected = detect_language(message.text)
            if detected:
                state.language = detected
                if state.active_flow == "awaiting_language":
                    state.active_flow = None
            else:
                state.active_flow = "awaiting_language"
                self.store.save_conversation(state)
                return [self._language_prompt()]

        state.language = state.language or "en"

        if state.active_flow:
            replies = await self._handle_active_flow(state, message)
            self.store.save_conversation(state)
            return replies

        intent = detect_intent(message.text)
        replies = await self._handle_intent(state, message, intent)
        self.store.save_conversation(state)
        return replies

    async def _handle_active_flow(
        self,
        state: ConversationState,
        message: InboundMessage,
    ) -> list[AgentReply]:
        if state.active_flow == "awaiting_language":
            return [self._language_prompt()]
        if state.active_flow == "awaiting_shipping_location":
            state.active_flow = None
            return [self._shipping_reply(state.language or "en", message.text)]
        if state.active_flow == "awaiting_wholesale_name":
            return [self._capture_wholesale_name(state, message.text)]
        if state.active_flow == "awaiting_wholesale_business":
            return [self._capture_wholesale_business(state, message.text)]
        if state.active_flow == "awaiting_wholesale_location":
            return [self._capture_wholesale_location(state, message.text)]
        if state.active_flow == "awaiting_wholesale_shop_size":
            return [self._capture_wholesale_shop_size(state, message.text)]
        if state.active_flow == "awaiting_wholesale_quantity":
            return [self._capture_wholesale_quantity(state, message.text)]
        if state.active_flow == "awaiting_wholesale_roast":
            return [self._capture_wholesale_roast(state, message.text)]
        if state.active_flow == "awaiting_wholesale_email":
            return [self._capture_wholesale_email(state, message.text)]
        if state.active_flow == "awaiting_wholesale_phone":
            return await self._capture_wholesale_phone(state, message.text)
        state.active_flow = None
        return [self._clarify_reply(state.language or "en")]

    async def _handle_intent(
        self,
        state: ConversationState,
        message: InboundMessage,
        intent: IntentMatch,
    ) -> list[AgentReply]:
        language = state.language or "en"
        cleaned = normalize_text(message.text)

        if intent.intent == "menu" or cleaned in {"hi", "hello", "hola", "start"}:
            state.context["unknown_count"] = 0
            return [self._welcome_reply(language), self._main_menu_reply(language)]

        if intent.intent == "coffee":
            return [self._coffee_reply(language)]

        if intent.intent == "shipping":
            if any(token in cleaned for token in {"mexico", "méxico", "usa", "united states", "estados unidos"}):
                return [self._shipping_reply(language, cleaned)]
            state.active_flow = "awaiting_shipping_location"
            return [self._ask_shipping_location(language)]

        if intent.intent in {"wholesale", "sample", "pricing"}:
            state.active_flow = "awaiting_wholesale_name"
            state.context = {"intent": intent.intent}
            return [self._ask_wholesale_name(language, intent.intent)]

        if intent.intent == "human":
            state.active_flow = None
            state.context["person_requests"] = state.context.get("person_requests", 0) + 1
            await self.lead_capture.capture(
                "sales_handoff",
                message.customer_id,
                language,
                {"last_message": message.text},
            )
            return [AgentReply(text=self.knowledge_base.escalation_reply(language))]

        return [self._handle_unknown(state, message.text)]

    def _language_prompt(self) -> AgentReply:
        return AgentReply(
            text=(
                "Welcome to Ali Chalmer Coffee wholesale. I help coffee shops in the U.S. and Mexico carry our coffee.\n"
                "Bienvenido a mayoreo de Ali Chalmer Coffee. Ayudo a cafeterías de México y Estados Unidos a vender nuestro café.\n"
                "Reply with English or Español."
            ),
            buttons=[
                ReplyOption(id="english", title="English"),
                ReplyOption(id="español", title="Español"),
            ],
        )

    def _language_confirmed_reply(self, language: str) -> AgentReply:
        if language == "es":
            return AgentReply(text="Perfecto. Continuamos en español.")
        return AgentReply(text="Perfect. We will continue in English.")

    def _welcome_reply(self, language: str) -> AgentReply:
        if language == "es":
            return AgentReply(
                text=(
                    "Bienvenido a Ali Chalmer Coffee. Este bot ayuda a cafeterías de México y Estados Unidos "
                    "a vender nuestro café por mayoreo."
                )
            )
        return AgentReply(
            text=(
                "Welcome to Ali Chalmer Coffee. This bot helps coffee shops in the U.S. and Mexico "
                "carry our coffee wholesale."
            )
        )

    def _main_menu_reply(self, language: str) -> AgentReply:
        if language == "es":
            options = [
                ReplyOption(id="coffee", title="Tuestes", description="Ver opciones para tu barra"),
                ReplyOption(id="wholesale", title="Mayoreo", description="Abrir cuenta para cafetería"),
                ReplyOption(id="sample", title="Muestras", description="Pedir opciones de prueba"),
                ReplyOption(id="pricing", title="Precios", description="Cotización para negocio"),
                ReplyOption(id="shipping", title="Envío", description="México o Estados Unidos"),
                ReplyOption(id="human", title="Ventas", description="Hablar con ventas"),
            ]
            return AgentReply(
                text="Elige cómo quieres vender Ali Chalmer Coffee en tu cafetería.",
                list_options=options,
                list_button_text="Ver menú",
            )
        options = [
            ReplyOption(id="coffee", title="Roasts", description="Options for your bar"),
            ReplyOption(id="wholesale", title="Wholesale", description="Open a shop account"),
            ReplyOption(id="sample", title="Samples", description="Request tasting options"),
            ReplyOption(id="pricing", title="Pricing", description="Business quote"),
            ReplyOption(id="shipping", title="Shipping", description="U.S. or Mexico"),
            ReplyOption(id="human", title="Sales", description="Talk to sales"),
        ]
        return AgentReply(
            text="Choose how you want to carry Ali Chalmer Coffee in your shop.",
            list_options=options,
            list_button_text="Open menu",
        )

    def _coffee_reply(self, language: str) -> AgentReply:
        if language == "es":
            return AgentReply(
                text=(
                    f"{self.knowledge_base.coffee_reply('es')}\n\n"
                    f"Ver café: {self.settings.coffee_product_url}"
                ),
                buttons=[
                    ReplyOption(id="wholesale", title="Mayoreo"),
                    ReplyOption(id="sample", title="Muestras"),
                    ReplyOption(id="shipping", title="Envío"),
                ],
            )
        return AgentReply(
            text=(
                f"{self.knowledge_base.coffee_reply('en')}\n\n"
                f"View coffee: {self.settings.coffee_product_url}"
            ),
            buttons=[
                ReplyOption(id="wholesale", title="Wholesale"),
                ReplyOption(id="sample", title="Samples"),
                ReplyOption(id="shipping", title="Shipping"),
            ],
        )

    def _ask_shipping_location(self, language: str) -> AgentReply:
        if language == "es":
            return AgentReply(text="¿En qué ciudad, estado y país está tu cafetería?")
        return AgentReply(text="What city, state, and country is your coffee shop in?")

    def _shipping_reply(self, language: str, destination: str) -> AgentReply:
        return AgentReply(
            text=self.knowledge_base.shipping_reply(language, destination),
            buttons=[
                ReplyOption(id="wholesale", title="Mayoreo" if language == "es" else "Wholesale"),
                ReplyOption(id="sample", title="Muestras" if language == "es" else "Samples"),
                ReplyOption(id="human", title="Ventas" if language == "es" else "Sales"),
            ],
        )

    def _ask_wholesale_name(self, language: str, intent: str = "wholesale") -> AgentReply:
        pitch = self.knowledge_base.wholesale_pitch(language)
        if language == "es":
            if intent == "sample":
                return AgentReply(text=f"{pitch}\n\nPara muestras, primero compárteme tu nombre.")
            if intent == "pricing":
                return AgentReply(text=f"{pitch}\n\nPara cotizar, primero compárteme tu nombre.")
            return AgentReply(text=f"{pitch}\n\nPrimero compárteme tu nombre.")
        if intent == "sample":
            return AgentReply(text=f"{pitch}\n\nFor samples, first share your name.")
        if intent == "pricing":
            return AgentReply(text=f"{pitch}\n\nFor pricing, first share your name.")
        return AgentReply(text=f"{pitch}\n\nFirst, share your name.")

    def _capture_wholesale_name(self, state: ConversationState, text: str) -> AgentReply:
        state.context["name"] = text.strip()
        state.active_flow = "awaiting_wholesale_business"
        if state.language == "es":
            return AgentReply(text="Gracias. ¿Cómo se llama tu cafetería o negocio?")
        return AgentReply(text="Thanks. What is the name of your coffee shop or business?")

    def _capture_wholesale_business(self, state: ConversationState, text: str) -> AgentReply:
        state.context["business_name"] = text.strip()
        state.active_flow = "awaiting_wholesale_location"
        if state.language == "es":
            return AgentReply(text="¿En qué ciudad, estado y país está tu negocio?")
        return AgentReply(text="What city, state, and country is your business in?")

    def _capture_wholesale_location(self, state: ConversationState, text: str) -> AgentReply:
        state.context["location"] = text.strip()
        state.active_flow = "awaiting_wholesale_shop_size"
        if state.language == "es":
            return AgentReply(text="¿Es una cafetería, restaurante, tienda u otro negocio? ¿Cuántas sucursales tienen?")
        return AgentReply(text="Is it a coffee shop, restaurant, store, or another business? How many locations do you have?")

    def _capture_wholesale_shop_size(self, state: ConversationState, text: str) -> AgentReply:
        state.context["business_type_and_locations"] = text.strip()
        state.active_flow = "awaiting_wholesale_quantity"
        if state.language == "es":
            return AgentReply(text="¿Cuánto café compras o usas al mes aproximadamente? Puedes responder en bolsas, kilos o libras.")
        return AgentReply(text="About how much coffee do you buy or use per month? You can answer in bags, kilos, or pounds.")

    def _capture_wholesale_quantity(self, state: ConversationState, text: str) -> AgentReply:
        state.context["monthly_volume"] = text.strip()
        state.active_flow = "awaiting_wholesale_roast"
        if state.language == "es":
            return AgentReply(
                text=(
                    "¿Qué te interesa más: Ali Chalmer Roast medio-oscuro, Xirandél Dark, "
                    "Peligroso 19 extra oscuro, o una muestra de los tres?"
                )
            )
        return AgentReply(
            text=(
                "Which option interests you most: Ali Chalmer Roast medium-dark, Xirandél Dark, "
                "Peligroso 19 extra dark, or a sample of all three?"
            )
        )

    def _capture_wholesale_roast(self, state: ConversationState, text: str) -> AgentReply:
        state.context["roast_interest"] = text.strip()
        state.active_flow = "awaiting_wholesale_email"
        if state.language == "es":
            return AgentReply(text="Compárteme tu correo electrónico, por favor.")
        return AgentReply(text="Please share your email address.")

    def _capture_wholesale_email(self, state: ConversationState, text: str) -> AgentReply:
        state.context["email"] = text.strip()
        state.active_flow = "awaiting_wholesale_phone"
        if state.language == "es":
            return AgentReply(text="Y por último, tu teléfono o WhatsApp de contacto.")
        return AgentReply(text="And finally, your phone or WhatsApp contact number.")

    async def _capture_wholesale_phone(self, state: ConversationState, text: str) -> list[AgentReply]:
        state.context["phone"] = text.strip()
        state.active_flow = None
        language = state.language or "en"
        await self.lead_capture.capture("coffee_shop_wholesale", state.customer_id, language, state.context)
        if language == "es":
            return [
                AgentReply(
                    text=(
                        "Gracias. Ya tenemos la información de tu cafetería. "
                        "Ventas revisará volumen, muestras, precio y envío para México o Estados Unidos. "
                        f"También puedes ver la página comercial: {self.settings.wholesale_page_url}"
                    )
                )
            ]
        return [
            AgentReply(
                text=(
                    "Thanks. We have your coffee shop details. "
                    "Sales will review volume, samples, pricing, and U.S./Mexico shipping. "
                    f"You can also view the business page here: {self.settings.wholesale_page_url}"
                )
            )
        ]

    def _clarify_reply(self, language: str) -> AgentReply:
        if language == "es":
            return AgentReply(
                text=(
                    "Este bot solo ayuda a cafeterías de México y Estados Unidos a vender Ali Chalmer Coffee. "
                    "¿Quieres ver tuestes, pedir muestras, revisar envío o abrir mayoreo?"
                )
            )
        return AgentReply(
            text=(
                "This bot only helps coffee shops in the U.S. and Mexico carry Ali Chalmer Coffee. "
                "Do you want roasts, samples, shipping, or wholesale?"
            )
        )

    def _handle_unknown(self, state: ConversationState, text: str) -> AgentReply:
        language = state.language or "en"
        unknown_count = int(state.context.get("unknown_count", 0)) + 1
        state.context["unknown_count"] = unknown_count
        if unknown_count >= 2:
            state.active_flow = None
            return AgentReply(text=self.knowledge_base.escalation_reply(language))
        return self._clarify_reply(language)
