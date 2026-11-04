"""Check the configured MySQL and create application tables; no sample patient writes."""
import sys
from sqlalchemy import text, inspect
from sqlalchemy.exc import SQLAlchemyError
from diabetes.storage import Database

try:
    db = Database()
    with db.engine.connect() as connection:
        print("Connected database:", connection.execute(text("SELECT DATABASE()")).scalar())
        print("Tables:", ", ".join(inspect(db.engine).get_table_names()))
    db.engine.dispose()
except SQLAlchemyError as exc:
    original = getattr(exc, "orig", None)
    code = original.args[0] if original is not None and original.args else "unknown"
    print(f"MySQL unavailable: {type(exc).__name__}, code={code}. Check MYSQL_* settings or DATABASE_URL in .env.")
    sys.exit(1)
