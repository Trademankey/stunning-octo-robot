from __future__ import annotations

import re
from collections import Counter
from typing import Optional

SPANISH_WORDS = {
    "hola",
    "cafe",
    "café",
    "envio",
    "envío",
    "mayoreo",
    "muestras",
    "precio",
    "cotizacion",
    "cotización",
    "cafeteria",
    "cafetería",
    "negocio",
    "quiero",
    "gracias",
    "necesito",
    "hablar",
    "persona",
    "ayuda",
}

ENGLISH_WORDS = {
    "hello",
    "hi",
    "coffee",
    "shipping",
    "wholesale",
    "sample",
    "samples",
    "pricing",
    "price",
    "cafe",
    "shop",
    "business",
    "thanks",
    "need",
    "person",
    "sales",
    "help",
}

LANGUAGE_CHOICES = {
    "english": "en",
    "inglish": "en",
    "ingles": "en",
    "ingles": "en",
    "inglés": "en",
    "espanol": "es",
    "español": "es",
    "spanish": "es",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def normalize_language_choice(text: str) -> Optional[str]:
    cleaned = normalize_text(text)
    return LANGUAGE_CHOICES.get(cleaned)


def detect_language(text: str) -> Optional[str]:
    cleaned = normalize_text(text)
    explicit = normalize_language_choice(cleaned)
    if explicit:
        return explicit

    tokens = re.findall(r"[a-zA-Záéíóúñü]+", cleaned)
    if not tokens:
        return None

    counter = Counter()
    for token in tokens:
        if token in SPANISH_WORDS:
            counter["es"] += 1
        if token in ENGLISH_WORDS:
            counter["en"] += 1

    if counter["es"] > counter["en"]:
        return "es"
    if counter["en"] > counter["es"]:
        return "en"
    return None
