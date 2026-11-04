"""Document ingestion with provenance, overlap chunks, deduplication and approved public URLs."""
import hashlib
import io
import json
import re
from urllib.parse import urlparse
import requests
from .config import KB

ALLOWED_HOSTS = {"www.niddk.nih.gov", "www.cdc.gov", "diabetes.org", "www.who.int"}

def chunk_text(text, size=1200, overlap=180):
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) < 40:
        raise ValueError("Tài liệu không đủ văn bản; PDF ảnh cần OCR trước khi nạp.")
    return [text[start:start + size] for start in range(0, len(text), size - overlap)
            if len(text[start:start + size].strip()) >= 40]

def ingest_text(db, text, title, source):
    if not title.strip() or not source.strip():
        raise ValueError("Cần tên tài liệu và nguồn.")
    if len(text) > 60_000:
        raise ValueError("Tối đa 60.000 ký tự mỗi tài liệu; hãy chia sách thành các chương.")
    chunks = chunk_text(text)
    digest = hashlib.sha256((source + "\n" + text).encode()).hexdigest()
    return db.add_document(title, source, text, digest, chunks)

def seed_knowledge(db):
    for path in sorted(KB.glob("*.json")):
        for doc in json.loads(path.read_text(encoding="utf-8")):
            ingest_text(db, doc["text"], doc["title"], doc["url"])

def read_upload(name, content):
    if len(content) > 10_000_000:
        raise ValueError("Tệp tối đa 10 MB.")
    if name.lower().endswith(".pdf"):
        from pypdf import PdfReader
        pdf = PdfReader(io.BytesIO(content))
        if len(pdf.pages) > 100:
            raise ValueError("PDF tối đa 100 trang; hãy chia theo chương.")
        return "\n".join(page.extract_text() or "" for page in pdf.pages)
    return content.decode("utf-8-sig")

def fetch_web(url):
    from bs4 import BeautifulSoup
    for _ in range(4):
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS or parsed.port not in (None, 443) or parsed.username:
            raise ValueError("Chỉ nạp HTTPS từ NIDDK, CDC, ADA hoặc WHO.")
        with requests.get(url, timeout=(5, 30), allow_redirects=False, stream=True) as response:
            if response.is_redirect:
                from urllib.parse import urljoin
                url = urljoin(url, response.headers["Location"])
                continue
            response.raise_for_status()
            if "text/html" not in response.headers.get("Content-Type", ""):
                raise ValueError("URL phải là trang HTML; với PDF hãy tải tệp lên.")
            data = bytearray()
            for block in response.iter_content(16384):
                data.extend(block)
                if len(data) > 3_000_000:
                    raise ValueError("Trang web quá lớn.")
            soup = BeautifulSoup(bytes(data), "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()
            main = soup.find("main") or soup.find("article") or soup.body or soup
            return main.get_text("\n", strip=True)[:60_000], url
    raise ValueError("Quá nhiều lần chuyển hướng.")
