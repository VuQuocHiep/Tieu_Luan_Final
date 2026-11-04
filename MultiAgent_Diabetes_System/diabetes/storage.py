import json
import os
from dotenv import load_dotenv
from sqlalchemy.engine import URL
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlalchemy import (create_engine, MetaData, Table, Column, Integer, Text,
                        String, ForeignKey, DateTime, func, select, insert, update, inspect, text)
from .config import ROOT

def connection_url():
    """Separate MySQL settings take precedence; passwords remain literal strings."""
    if os.getenv("MYSQL_USER") is not None:
        return URL.create("mysql+pymysql", username=os.environ["MYSQL_USER"],
            password=os.getenv("MYSQL_PASSWORD", ""), host=os.getenv("MYSQL_HOST", "localhost"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            database=os.getenv("MYSQL_DATABASE", "mutilagent"))
    return os.getenv("DATABASE_URL", "mysql+pymysql://diabetes:diabetes_local@localhost:3306/mutilagent")

class Database:
    """Default MySQL storage; an explicit SQLite URL is supported for unit tests."""
    def __init__(self, url=None):
        load_dotenv(ROOT / ".env")
        self.engine = create_engine(url if url is not None else connection_url(), pool_pre_ping=True)
        meta = MetaData()
        self.runs = Table("runs", meta,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("created_at", DateTime, server_default=func.now()),
            Column("patient", Text, nullable=False), Column("result", Text, nullable=False),
            Column("style", String(32), nullable=False))
        self.feedback_table = Table("feedback", meta,
            Column("run_id", Integer, ForeignKey("runs.id"), primary_key=True),
            Column("reward", Integer, nullable=False))
        self.samples = Table("training_samples", meta,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("dataset", String(64), nullable=False), Column("payload", Text, nullable=False))
        self.documents = Table("kb_documents", meta,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("digest", String(64), unique=True, nullable=False),
            Column("title", String(255), nullable=False), Column("source", Text, nullable=False),
            Column("content", Text().with_variant(MEDIUMTEXT(), "mysql"), nullable=False), Column("created_at", DateTime, server_default=func.now()))
        self.chunks = Table("kb_chunks", meta,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("document_id", Integer, ForeignKey("kb_documents.id"), nullable=False),
            Column("position", Integer, nullable=False), Column("content", Text, nullable=False))
        self.conversations = Table("conversations", meta,
            Column("id", String(36), primary_key=True), Column("title", String(255), nullable=False),
            Column("created_at", DateTime, server_default=func.now()))
        self.turns = Table("chat_turns", meta,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("conversation_id", String(36), ForeignKey("conversations.id"), nullable=False),
            Column("question", Text, nullable=False), Column("response", Text, nullable=False),
            Column("state", String(100), nullable=False), Column("action", String(32), nullable=False),
            Column("next_state", String(100)), Column("reward", Integer),
            Column("created_at", DateTime, server_default=func.now()))
        self.qvalues = Table("rl_qvalues", meta,
            Column("state", String(100), primary_key=True), Column("action", String(32), primary_key=True),
            Column("value", Text, nullable=False), Column("visits", Integer, nullable=False))
        self.catalog = Table("dataset_catalog", meta,
            Column("name", String(64), primary_key=True), Column("metadata", Text, nullable=False))
        self.model_runs = Table("model_runs", meta,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("metrics", Text, nullable=False), Column("created_at", DateTime, server_default=func.now()))
        meta.create_all(self.engine)
        if self.engine.dialect.name == "mysql":
            content_type = next(c["type"] for c in inspect(self.engine).get_columns("kb_documents") if c["name"] == "content")
            if type(content_type).__name__ == "TEXT":
                with self.engine.begin() as con:
                    con.execute(text("ALTER TABLE kb_documents MODIFY COLUMN content MEDIUMTEXT NOT NULL"))

    def add_document(self, title, source, content, digest, chunks):
        with self.engine.begin() as con:
            existing = con.execute(select(self.documents.c.id).where(self.documents.c.digest == digest)).scalar()
            if existing:
                return existing
            doc_id = con.execute(insert(self.documents).values(title=title[:255], source=source,
                content=content, digest=digest)).inserted_primary_key[0]
            con.execute(insert(self.chunks), [{"document_id": doc_id, "position": i, "content": part}
                for i, part in enumerate(chunks)])
            return doc_id

    def knowledge_chunks(self):
        d, c = self.documents, self.chunks
        with self.engine.connect() as con:
            return [dict(row) for row in con.execute(select(c.c.id, d.c.title,
                d.c.source.label("url"), c.c.content.label("text")).select_from(c.join(d))).mappings()]

    def document_list(self):
        with self.engine.connect() as con:
            return [dict(r) for r in con.execute(select(self.documents.c.id, self.documents.c.title,
                self.documents.c.source, self.documents.c.created_at)).mappings()]

    def create_conversation(self, title):
        import uuid
        cid = str(uuid.uuid4())
        with self.engine.begin() as con:
            con.execute(insert(self.conversations).values(id=cid, title=title[:255]))
        return cid

    def conversation_list(self):
        with self.engine.connect() as con:
            return [dict(row) for row in con.execute(select(self.conversations)
                .order_by(self.conversations.c.created_at.desc()).limit(30)).mappings()]

    def chat_history(self, conversation_id):
        with self.engine.connect() as con:
            return [dict(row) for row in con.execute(select(self.turns).where(
                self.turns.c.conversation_id == conversation_id).order_by(self.turns.c.id)).mappings()]

    def save_turn(self, conversation_id, question, response, state, action):
        with self.engine.begin() as con:
            con.execute(select(self.conversations.c.id).where(self.conversations.c.id == conversation_id).with_for_update())
            previous = con.execute(select(self.turns.c.id).where(self.turns.c.conversation_id == conversation_id)
                .order_by(self.turns.c.id.desc()).limit(1)).scalar()
            if previous:
                con.execute(update(self.turns).where(self.turns.c.id == previous).values(next_state=state))
            return con.execute(insert(self.turns).values(conversation_id=conversation_id, question=question,
                response=json.dumps(response, ensure_ascii=False), state=state, action=action)).inserted_primary_key[0]

    def qtable(self):
        with self.engine.connect() as con:
            return [dict(row) for row in con.execute(select(self.qvalues)).mappings()]

    def reward_turn(self, turn_id, reward, alpha=.2, gamma=.8):
        """One reward per turn; lock session and Q rows before updating Q(s,a)."""
        if reward not in (-1, 1):
            raise ValueError("Phản hồi phải là -1 hoặc 1.")
        t, q = self.turns, self.qvalues
        with self.engine.begin() as con:
            con.execute(select(self.conversations.c.id).order_by(self.conversations.c.id).with_for_update()).all()
            row = con.execute(select(t).where(t.c.id == turn_id).with_for_update()).mappings().first()
            if not row:
                raise ValueError("Không tìm thấy câu trả lời.")
            if row["reward"] is not None:
                return False
            # All updates acquire the same tiny table lock ordering to serialize new state/action rows.
            current = con.execute(select(q).where(q.c.state == row["state"],
                q.c.action == row["action"]).with_for_update()).mappings().first()
            next_values = con.execute(select(q.c.value).where(q.c.state == row["next_state"])).scalars().all() if row["next_state"] else []
            old = float(current["value"]) if current else 0.
            candidates = [float(v) for v in next_values]
            if len(candidates) < 3:
                candidates.append(0.)
            new = old + alpha * (reward + gamma * max(candidates) - old)
            values = {"value": str(new), "visits": current["visits"] + 1 if current else 1}
            if current:
                con.execute(update(q).where(q.c.state == row["state"], q.c.action == row["action"]).values(**values))
            else:
                con.execute(insert(q).values(state=row["state"], action=row["action"], **values))
            con.execute(update(t).where(t.c.id == turn_id).values(reward=reward))
            return True

    def save(self, patient, result, style):
        with self.engine.begin() as con:
            return con.execute(insert(self.runs).values(patient=json.dumps(patient),
                result=json.dumps(result, ensure_ascii=False), style=style)).inserted_primary_key[0]

    def feedback(self, run_id, reward):
        if reward not in (0, 1):
            raise ValueError("Reward phải là 0 hoặc 1.")
        with self.engine.begin() as con:
            exists = con.execute(select(self.runs.c.id).where(self.runs.c.id == run_id).with_for_update()).first()
            if not exists:
                raise ValueError("Không tìm thấy lần phân tích.")
            current = con.execute(select(self.feedback_table).where(self.feedback_table.c.run_id == run_id)).first()
            if current:
                con.execute(update(self.feedback_table).where(self.feedback_table.c.run_id == run_id).values(reward=reward))
            else:
                con.execute(insert(self.feedback_table).values(run_id=run_id, reward=reward))

    def stats(self):
        r, f = self.runs, self.feedback_table
        with self.engine.connect() as con:
            return [tuple(row) for row in con.execute(select(r.c.style, func.count(), func.sum(f.c.reward))
                .select_from(r.join(f, r.c.id == f.c.run_id)).group_by(r.c.style))]

    def recent(self):
        with self.engine.connect() as con:
            return [tuple(row) for row in con.execute(select(self.runs.c.id, self.runs.c.created_at,
                self.runs.c.result).order_by(self.runs.c.id.desc()).limit(20))]

    def import_samples(self, frame, dataset, metadata=None):
        with self.engine.begin() as con:
            con.execute(self.samples.delete().where(self.samples.c.dataset == dataset))
            rows = [{"dataset": dataset, "payload": json.dumps(row)} for row in frame.to_dict("records")]
            if rows:
                con.execute(insert(self.samples), rows)
            con.execute(self.catalog.delete().where(self.catalog.c.name == dataset))
            con.execute(insert(self.catalog).values(name=dataset, metadata=json.dumps(metadata or {}, ensure_ascii=False)))

    def dataset_list(self):
        with self.engine.connect() as con:
            return [dict(row) for row in con.execute(select(self.samples.c.dataset, func.count().label("samples"))
                .group_by(self.samples.c.dataset)).mappings()]

    def dataset_metadata(self, name):
        with self.engine.connect() as con:
            value = con.execute(select(self.catalog.c.metadata).where(self.catalog.c.name == name)).scalar()
            return json.loads(value) if value else {}

    def save_model_metrics(self, metrics):
        with self.engine.begin() as con:
            con.execute(insert(self.model_runs).values(metrics=json.dumps(metrics, ensure_ascii=False)))

    def load_samples(self, dataset):
        import pandas as pd
        with self.engine.connect() as con:
            rows = con.execute(select(self.samples.c.payload).where(self.samples.c.dataset == dataset)).scalars().all()
        if not rows:
            raise ValueError("Dataset chưa có trong MySQL.")
        return pd.DataFrame([json.loads(row) for row in rows])
