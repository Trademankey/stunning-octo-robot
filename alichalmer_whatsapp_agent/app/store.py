from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy import Text
from sqlalchemy import create_engine
from sqlalchemy import insert
from sqlalchemy import select
from sqlalchemy import update
from sqlalchemy.engine import make_url

from .types import ConversationState


class ConversationStore:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self._lock = threading.Lock()
        url = make_url(database_url)
        if url.get_backend_name().startswith("sqlite") and url.database:
            db_path = Path(url.database)
            if not db_path.is_absolute():
                db_path = Path.cwd() / db_path
            db_path.parent.mkdir(parents=True, exist_ok=True)
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self._engine = create_engine(
            database_url,
            future=True,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        self._metadata = MetaData()
        self._conversations = Table(
            "conversations",
            self._metadata,
            Column("customer_id", String(255), primary_key=True),
            Column("language", String(16), nullable=True),
            Column("active_flow", String(255), nullable=True),
            Column("context_json", Text, nullable=False, default="{}"),
            Column("updated_at", DateTime, nullable=False),
        )
        self._leads = Table(
            "leads",
            self._metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("kind", String(64), nullable=False),
            Column("customer_id", String(255), nullable=False),
            Column("language", String(16), nullable=True),
            Column("payload_json", Text, nullable=False),
            Column("created_at", DateTime, nullable=False),
        )
        self._setup()

    def _setup(self) -> None:
        with self._lock:
            self._metadata.create_all(self._engine)

    def get_conversation(self, customer_id: str) -> ConversationState:
        with self._lock:
            with self._engine.connect() as conn:
                row = conn.execute(
                    select(
                        self._conversations.c.customer_id,
                        self._conversations.c.language,
                        self._conversations.c.active_flow,
                        self._conversations.c.context_json,
                    ).where(self._conversations.c.customer_id == customer_id)
                ).mappings().first()

        if not row:
            return ConversationState(customer_id=customer_id)

        return ConversationState(
            customer_id=row["customer_id"],
            language=row["language"],
            active_flow=row["active_flow"],
            context=json.loads(row["context_json"] or "{}"),
        )

    def save_conversation(self, state: ConversationState) -> None:
        payload = {
            "customer_id": state.customer_id,
            "language": state.language,
            "active_flow": state.active_flow,
            "context_json": json.dumps(state.context, ensure_ascii=False),
            "updated_at": datetime.utcnow(),
        }
        with self._lock:
            with self._engine.begin() as conn:
                updated = conn.execute(
                    update(self._conversations)
                    .where(self._conversations.c.customer_id == state.customer_id)
                    .values(**payload)
                )
                if updated.rowcount == 0:
                    conn.execute(insert(self._conversations).values(**payload))

    def save_lead(self, kind: str, customer_id: str, language: str, payload: dict) -> int:
        values = {
            "kind": kind,
            "customer_id": customer_id,
            "language": language,
            "payload_json": json.dumps(payload, ensure_ascii=False),
            "created_at": datetime.utcnow(),
        }
        with self._lock:
            with self._engine.begin() as conn:
                result = conn.execute(insert(self._leads).values(**values))
                inserted_id = result.inserted_primary_key[0]
        return int(inserted_id)
