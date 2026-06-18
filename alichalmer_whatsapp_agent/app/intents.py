from __future__ import annotations

from dataclasses import dataclass

from .language import normalize_text


@dataclass
class IntentMatch:
    intent: str
    confidence: float


ESCALATION_TERMS = {
    "refund",
    "fraud",
    "chargeback",
    "scam",
    "lawsuit",
    "missing order",
    "missing package",
    "lawyer",
    "reembolso",
    "fraude",
    "demanda",
    "contracargo",
    "estafa",
    "pedido perdido",
    "paquete perdido",
    "exclusive distributor",
    "distribuidor exclusivo",
    "contract",
    "contrato",
    "invoice",
    "factura",
}

KEYWORDS = {
    "coffee": {
        "coffee",
        "cafe",
        "café",
        "roast",
        "beans",
        "veracruz",
        "notas",
        "notes",
        "blend",
        "whole bean",
        "ground",
        "grano",
        "molido",
        "xirandel",
        "xirandél",
        "peligroso",
    },
    "shipping": {
        "shipping",
        "ship",
        "delivery",
        "envio",
        "envío",
        "entrega",
        "mexico",
        "méxico",
        "usa",
        "united states",
        "eeuu",
        "estados unidos",
    },
    "wholesale": {
        "wholesale",
        "bulk",
        "business",
        "reseller",
        "coffee shop",
        "coffee bar",
        "cafe",
        "café",
        "restaurant",
        "hotel",
        "shop",
        "store",
        "office coffee",
        "case",
        "cases",
        "bags",
        "volume",
        "mayoreo",
        "cafeteria",
        "cafetería",
        "negocio",
        "distribuidor",
        "varias bolsas",
        "por volumen",
    },
    "sample": {
        "sample",
        "samples",
        "try",
        "taste",
        "tasting",
        "starter pack",
        "paquete muestra",
        "muestra",
        "muestras",
        "probar",
        "cata",
    },
    "pricing": {
        "price",
        "pricing",
        "cost",
        "quote",
        "margin",
        "minimum",
        "moq",
        "precio",
        "cotizacion",
        "cotización",
        "costo",
        "margen",
        "minimo",
        "mínimo",
    },
    "human": {
        "person",
        "human",
        "agent",
        "representative",
        "someone",
        "persona",
        "humano",
        "asesor",
        "hablar con alguien",
    },
    "menu": {
        "menu",
        "menú",
        "help",
        "ayuda",
        "start",
        "inicio",
    },
}

NUMERIC_MENU_MAP = {
    "1": "coffee",
    "2": "wholesale",
    "3": "sample",
    "4": "shipping",
    "5": "human",
    "6": "human",
}


def detect_intent(text: str) -> IntentMatch:
    cleaned = normalize_text(text)

    if cleaned in NUMERIC_MENU_MAP:
        return IntentMatch(NUMERIC_MENU_MAP[cleaned], 0.95)

    for term in ESCALATION_TERMS:
        if term in cleaned:
            return IntentMatch("human", 0.99)

    best_intent = "unknown"
    best_score = 0.0
    for intent, keywords in KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in cleaned)
        if score > best_score:
            best_intent = intent
            best_score = float(score)

    if best_score == 0:
        return IntentMatch("unknown", 0.0)

    confidence = min(0.95, 0.3 + (best_score * 0.2))
    return IntentMatch(best_intent, confidence)
