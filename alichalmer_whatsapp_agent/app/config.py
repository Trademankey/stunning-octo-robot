from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _resolve_path(path_value: str, service_root: Path) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((service_root / path).resolve())


@dataclass
class Settings:
    service_root: str
    database_path: str
    database_url: str
    whatsapp_verify_token: str
    whatsapp_access_token: str
    whatsapp_phone_number_id: str
    whatsapp_graph_version: str
    woo_base_url: str
    woo_consumer_key: str
    woo_consumer_secret: str
    coffee_product_url: str
    wholesale_page_url: str
    lead_webhook_url: str


def get_settings() -> Settings:
    service_root = Path(__file__).resolve().parents[1]
    database_path = _resolve_path(
        os.getenv("DATABASE_PATH", "data/agent_state.db"),
        service_root,
    )
    database_url = os.getenv("DATABASE_URL", f"sqlite:///{database_path}")
    return Settings(
        service_root=str(service_root),
        database_path=database_path,
        database_url=database_url,
        whatsapp_verify_token=os.getenv("WHATSAPP_VERIFY_TOKEN", ""),
        whatsapp_access_token=os.getenv("WHATSAPP_ACCESS_TOKEN", ""),
        whatsapp_phone_number_id=os.getenv("WHATSAPP_PHONE_NUMBER_ID", ""),
        whatsapp_graph_version=os.getenv("WHATSAPP_GRAPH_VERSION", "v23.0"),
        woo_base_url=os.getenv("WOOCOMMERCE_BASE_URL", "https://alichalmer.com"),
        woo_consumer_key=os.getenv("WOOCOMMERCE_CONSUMER_KEY", ""),
        woo_consumer_secret=os.getenv("WOOCOMMERCE_CONSUMER_SECRET", ""),
        coffee_product_url=os.getenv(
            "COFFEE_PRODUCT_URL",
            "https://alichalmer.com/product/ali-chalmer-roast/",
        ),
        wholesale_page_url=os.getenv(
            "WHOLESALE_PAGE_URL",
            "https://alichalmer.com/wholesale/",
        ),
        lead_webhook_url=os.getenv("LEAD_WEBHOOK_URL", ""),
    )
