import math
from .learning import predict
from .rag import KnowledgeAgent
from .rules import screen
from .storage import Database

class PresentationPolicy:
    """UCB1 bandit: learns explanation style only, never clinical thresholds/actions."""
    actions = ("concise", "detailed")

    def choose(self, stats):
        counts = {style: (int(n), float(reward)) for style, n, reward in stats}
        for style in self.actions:
            if style not in counts:
                return style
        total = sum(n for n, _ in counts.values())
        return max(self.actions, key=lambda s: counts[s][1] / counts[s][0] +
                   math.sqrt(2 * math.log(total) / counts[s][0]))

class RuleAgent:
    def run(self, patient):
        return screen(patient)

class PredictionAgent:
    def run(self, patient):
        return predict(patient)

class Coordinator:
    def __init__(self, db=None):
        self.db = db or Database()
        self.rule = RuleAgent()
        self.prediction = PredictionAgent()
        self.knowledge = KnowledgeAgent()
        self.policy = PresentationPolicy()

    def run(self, patient, question="Giải thích kết quả xét nghiệm và bước tiếp theo", persist=False):
        patient.validate()
        rule = self.rule.run(patient)
        prediction = self.prediction.run(patient)
        style = self.policy.choose(self.db.stats())
        rag = self.knowledge.generate(question, rule, style)
        result = {"rule": rule, "prediction": prediction, "rag": rag, "style": style,
                  "patient": patient.to_dict(),
                  "trace": ["RuleAgent: kiểm tra phạm vi và ngưỡng", "PredictionAgent: suy luận mạng nơ-ron nếu sẵn sàng",
                            "KnowledgeAgent: truy xuất KB và tổng hợp có nguồn", "PresentationPolicy: chọn cách trình bày"]}
        if persist:
            result["run_id"] = self.db.save(patient.to_dict(), result, style)
        return result
