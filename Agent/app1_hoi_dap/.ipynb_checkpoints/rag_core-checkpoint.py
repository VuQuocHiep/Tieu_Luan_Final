"""Local document extraction and BM25 retrieval. No paid embedding API required."""
import io
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from pypdf import PdfReader
from common.text_utils import tokenize

STOPWORDS = {"la", "cua", "co", "nhung", "nhung", "nhu", "the", "nao", "gi", "cho", "toi",
             "hay", "can", "duoc", "vao", "ve", "va", "mot", "cac", "trong", "khi", "o", "de", "tai", "den"}


@dataclass(frozen=True)
class Chunk:
    source: str
    page: int
    text: str
    chunk_id: int

    def to_dict(self):
        return {"source": self.source, "page": self.page, "chunk_id": self.chunk_id, "text": self.text}


def split_text(text: str, size: int = 950, overlap: int = 160) -> list[str]:
    text = " ".join(text.split())
    if size <= overlap or overlap < 0:
        raise ValueError("Kích thước chunk phải lớn hơn overlap.")
    if not text:
        return []
    result = []
    step = size - overlap
    for start in range(0, len(text), step):
        part = text[start:start + size]
        if part:
            result.append(part)
        if start + size >= len(text):
            break
    return result


def extract_file(filename: str, data: bytes) -> list[tuple[int, str]]:
    if filename.lower().endswith(".txt"):
        return [(1, data.decode("utf-8-sig", errors="replace"))]
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return [(page_no, page.extract_text() or "")
                for page_no, page in enumerate(reader.pages, start=1)]
    raise ValueError("Chỉ hỗ trợ tệp PDF hoặc TXT.")


def build_chunks(documents: Iterable[tuple[str, bytes]]) -> list[Chunk]:
    chunks = []
    for filename, data in documents:
        for page, text in extract_file(filename, data):
            for part in split_text(text):
                chunks.append(Chunk(filename, page, part, len(chunks) + 1))
    return chunks


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.tokens = [[t for t in tokenize(c.text) if t not in STOPWORDS] for c in chunks]
        self.counts = [Counter(ts) for ts in self.tokens]
        self.lengths = [len(ts) for ts in self.tokens]
        self.avg_len = sum(self.lengths) / len(self.lengths) if self.lengths else 1
        self.df = Counter()
        for ts in self.tokens:
            self.df.update(set(ts))

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        qtokens = [t for t in tokenize(query) if t not in STOPWORDS]
        if not qtokens:
            return []
        scored = []
        n = len(self.chunks)
        for i, counts in enumerate(self.counts):
            score = 0.0
            for term in set(qtokens):
                freq = counts.get(term, 0)
                if not freq:
                    continue
                idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                denom = freq + 1.5 * (0.25 + 0.75 * self.lengths[i] / max(self.avg_len, 1))
                score += idf * freq * 2.5 / denom
            if score > 0:
                scored.append((score, self.chunks[i]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{**chunk.to_dict(), "score": round(score, 3)} for score, chunk in scored[:max(1, min(int(top_k), 8))]]


def sample_file() -> tuple[str, bytes]:
    file = Path(__file__).parent / "data" / "huong_dan_kho.txt"
    return file.name, file.read_bytes()
