from __future__ import annotations

from pathlib import Path

from .config import Settings
from .language import normalize_text


class KnowledgeBase:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.knowledge_dir = Path(settings.service_root) / "knowledge"

    def document(self, name: str) -> str:
        return (self.knowledge_dir / name).read_text(encoding="utf-8")

    def coffee_reply(self, language: str) -> str:
        if language == "es":
            return (
                "Ali Chalmer Coffee ofrece tres tuestes pensados para cafeterías pequeñas: "
                "Ali Chalmer Roast, medio-oscuro, intenso, terroso y ahumado; "
                "Xirandél Dark, oscuro, profundo, con notas de chocolate oscuro y humo de montaña; "
                "y Peligroso 19 Blend, extra oscuro, complejo e intenso. "
                "La marca funciona bien para barras que quieren un café con historia, empaque memorable y sabor con carácter."
            )
        return (
            "Ali Chalmer Coffee offers three roasts for small coffee shops: "
            "Ali Chalmer Roast, a bold medium-dark roast with earthy and smoky character; "
            "Xirandél Dark, a deep dark roast with dark chocolate and mountain-smoke notes; "
            "and Peligroso 19 Blend, an intense extra-dark roast that is complex and dangerously smooth. "
            "It is a strong fit for shops that want memorable bags, story-driven branding, and coffee with character."
        )

    def wholesale_pitch(self, language: str) -> str:
        if language == "es":
            return (
                "Puedo ayudarte a abrir una cuenta de mayoreo para tu cafetería en México o Estados Unidos. "
                "Te pediré datos rápidos del negocio, volumen aproximado y contacto para que ventas prepare opciones de muestra, precio y envío."
            )
        return (
            "I can help open a wholesale conversation for your coffee shop in the U.S. or Mexico. "
            "I will collect quick business details, estimated volume, and contact info so sales can prepare sample, pricing, and shipping options."
        )

    def shipping_reply(self, language: str, destination: str) -> str:
        cleaned = normalize_text(destination)
        if any(term in cleaned for term in {"mexico", "méxico"}):
            if language == "es":
                return "Atendemos cafeterías dentro de México. Comparte ciudad, estado y volumen aproximado para preparar opciones de mayoreo y envío."
            return "We support coffee shops in Mexico. Share city, state, and estimated volume so sales can prepare wholesale and shipping options."
        if any(term in cleaned for term in {"usa", "united states", "eeuu", "estados unidos"}):
            if language == "es":
                return "Atendemos cafeterías en Estados Unidos. Comparte ciudad, estado y volumen aproximado para revisar mayoreo y envío."
            return "We support coffee shops in the United States. Share city, state, and estimated volume so sales can review wholesale and shipping."
        if language == "es":
            return "Este bot está enfocado en cafeterías de México y Estados Unidos. Compárteme ciudad, estado y país para revisar si podemos atenderte."
        return "This bot is focused on coffee shops in the U.S. and Mexico. Share city, state, and country so sales can confirm whether we can serve you."

    def escalation_reply(self, language: str) -> str:
        if language == "es":
            return "Voy a pasar esto al equipo de ventas de Ali Chalmer Coffee para que puedan darte seguimiento."
        return "I'll pass this to the Ali Chalmer Coffee sales team so they can follow up."
