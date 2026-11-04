"""SQLite sample warehouse with read-only analytical tools and approved drafts."""
import math
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "warehouse.db"

SAMPLE_PRODUCTS = [
    ("SP001", "Bàn phím cơ", "Phụ kiện", 8, 12, 650000),
    ("SP002", "Chuột không dây", "Phụ kiện", 40, 15, 250000),
    ("SP003", "Tai nghe USB", "Âm thanh", 6, 10, 420000),
    ("SP004", "Webcam HD", "Camera", 25, 8, 790000),
    ("SP005", "Cáp USB-C", "Phụ kiện", 12, 20, 90000),
    ("SP006", "Ổ cứng SSD 512GB", "Lưu trữ", 18, 10, 1100000),
]


def connect(db_path: str | Path = DB_PATH):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def initialize(db_path: str | Path = DB_PATH):
    with connect(db_path) as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS products(
          sku TEXT PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT NULL,
          quantity INTEGER NOT NULL CHECK(quantity >= 0), min_stock INTEGER NOT NULL, price INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS movements(
          id INTEGER PRIMARY KEY AUTOINCREMENT, sku TEXT NOT NULL,
          direction TEXT NOT NULL CHECK(direction IN ('IN','OUT')),
          quantity INTEGER NOT NULL CHECK(quantity > 0), moved_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS sensor_readings(
          id INTEGER PRIMARY KEY AUTOINCREMENT, temperature REAL, humidity REAL,
          gas_adc INTEGER, recorded_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS restock_drafts(
          id INTEGER PRIMARY KEY AUTOINCREMENT, sku TEXT NOT NULL,
          quantity INTEGER NOT NULL CHECK(quantity > 0), status TEXT NOT NULL,
          created_at TEXT NOT NULL);
        """)
        if con.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 0:
            con.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?)", SAMPLE_PRODUCTS)
            now = datetime.now()
            sales = {"SP001": 21, "SP002": 16, "SP003": 12, "SP004": 7, "SP005": 35, "SP006": 8}
            for sku, n in sales.items():
                for j in range(5):
                    date = (now - timedelta(days=5 + j * 5)).isoformat(timespec="seconds")
                    count = n // 5 + (1 if j < n % 5 else 0)
                    if count:
                        con.execute("INSERT INTO movements(sku,direction,quantity,moved_at) VALUES(?,?,?,?)",
                                    (sku, "OUT", count, date))
        if con.execute("SELECT COUNT(*) FROM sensor_readings").fetchone()[0] == 0:
            con.execute("INSERT INTO sensor_readings(temperature,humidity,gas_adc,recorded_at) VALUES(?,?,?,?)",
                        (29.7, 61.5, 420, datetime.now().isoformat(timespec="seconds")))


def products(db_path: str | Path = DB_PATH):
    with connect(db_path) as con:
        return [dict(r) for r in con.execute("SELECT * FROM products ORDER BY sku")]


def low_stock(db_path: str | Path = DB_PATH):
    with connect(db_path) as con:
        return [dict(r) for r in con.execute(
            "SELECT * FROM products WHERE quantity <= min_stock ORDER BY quantity * 1.0 / MAX(min_stock,1) ASC")]


def find_product(keyword: str, db_path: str | Path = DB_PATH):
    key = f"%{keyword.strip()}%"
    with connect(db_path) as con:
        return [dict(r) for r in con.execute(
            "SELECT * FROM products WHERE sku LIKE ? OR name LIKE ? LIMIT 15", (key, key))]


def sensor_status(db_path: str | Path = DB_PATH):
    with connect(db_path) as con:
        row = con.execute("SELECT * FROM sensor_readings ORDER BY recorded_at DESC, id DESC LIMIT 1").fetchone()
    if not row:
        return {"error": "Chưa có dữ liệu cảm biến."}
    return {**dict(row), "note": "Dữ liệu minh họa; gas_adc là chỉ số ADC thô, không phải ppm. Thời gian là lần ghi mẫu, không phải đo trực tiếp."}


def sales_summary(days: int = 30, db_path: str | Path = DB_PATH):
    days = max(1, min(int(days), 365))
    since = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    with connect(db_path) as con:
        rows = con.execute("""SELECT p.sku,p.name,COALESCE(SUM(m.quantity),0) AS units,
                                   COALESCE(SUM(m.quantity*p.price),0) AS revenue
                            FROM products p LEFT JOIN movements m
                              ON m.sku=p.sku AND m.direction='OUT' AND m.moved_at>=?
                            GROUP BY p.sku ORDER BY units DESC""", (since,)).fetchall()
    return {"days": days, "products": [dict(r) for r in rows],
            "note": "Doanh thu ước tính từ đơn giá mẫu x số lượng xuất; không phải dữ liệu kế toán."}


def reorder_suggestions(lead_days: int = 7, db_path: str | Path = DB_PATH):
    lead_days = max(1, min(int(lead_days), 90))
    sales = {p["sku"]: p["units"] for p in sales_summary(30, db_path)["products"]}
    result = []
    for p in products(db_path):
        daily = sales.get(p["sku"], 0) / 30.0
        target = max(p["min_stock"], math.ceil(daily * (lead_days + 7)))
        suggest = max(0, target - p["quantity"])
        if suggest:
            result.append({"sku": p["sku"], "name": p["name"],
                           "current": p["quantity"], "target": target,
                           "avg_out_per_day": round(daily, 2), "suggested_units": suggest})
    result.sort(key=lambda p: p["suggested_units"], reverse=True)
    return {"lead_days": lead_days, "suggestions": result,
            "method": "Quy tắc lượng xuất trung bình 30 ngày và tồn tối thiểu; KHÔNG phải mô hình Machine Learning."}


def add_draft(sku: str, qty: int, db_path: str | Path = DB_PATH):
    qty = int(qty)
    if qty <= 0 or qty > 100_000:
        raise ValueError("Số lượng phải từ 1 đến 100000.")
    with connect(db_path) as con:
        exists = con.execute("SELECT 1 FROM products WHERE sku = ?", (sku,)).fetchone()
        if not exists:
            raise ValueError("Mã sản phẩm không tồn tại.")
        cursor = con.execute("INSERT INTO restock_drafts(sku,quantity,status,created_at) VALUES(?,?,?,?)",
                             (sku, qty, "PENDING_REVIEW", datetime.now().isoformat(timespec="seconds")))
        return cursor.lastrowid


def get_drafts(db_path: str | Path = DB_PATH):
    with connect(db_path) as con:
        return [dict(r) for r in con.execute("SELECT * FROM restock_drafts ORDER BY id DESC LIMIT 30")]
