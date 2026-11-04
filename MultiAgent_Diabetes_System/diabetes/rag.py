import json
from .config import KB

class KnowledgeAgent:
    def retrieve(self, query, top_k=3):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        documents = []
        for path in sorted(KB.glob("*.json")):
            documents.extend(json.loads(path.read_text(encoding="utf-8")))
        if not documents:
            return []
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        matrix = vectorizer.fit_transform([d["title"] + " " + d["text"] for d in documents])
        scores = cosine_similarity(vectorizer.transform([query]), matrix)[0]
        return [dict(documents[i], similarity=round(float(scores[i]), 3))
                for i in scores.argsort()[::-1][:top_k] if scores[i] > .03]

    def generate(self, query, rule, style):
        docs = self.retrieve(query + " " + rule["label"], top_k=1 if style == "concise" else 3)
        if not docs:
            return {"answer": "KB hiện không có nội dung phù hợp để trả lời. Hãy hỏi nhân viên y tế.", "sources": []}
        # Grounded template synthesis: no remote LLM or unsupported medical advice.
        title = "Thông tin tham khảo" if style == "concise" else "Giải thích dựa trên cơ sở tri thức"
        body = "\n\n".join(f"[{i}] {d['text']}" for i, d in enumerate(docs, 1))
        return {"answer": f"{title}\n\n{body}", "sources": docs,
                "method": "TF-IDF retrieval + tổng hợp theo mẫu có trích dẫn (không dùng LLM)"}
