from __future__ import annotations

from typing import Iterable

from agent import LiveAgent
from evaluator import OfflineEvaluator
from experience_pool import ExperiencePool
from models import LiveContext


class PromptOptimizer:
    """针对多种提示风格的小型PromptAgent式搜索。"""

    def __init__(self, context: LiveContext, pool: ExperiencePool, evaluator: OfflineEvaluator) -> None:
        self.context = context
        self.pool = pool
        self.evaluator = evaluator

    def optimize(self, validation_messages: Iterable[str]) -> dict[str, object]:
        styles = ["baseline", "concise", "conversion", "safety"]
        results: list[dict[str, object]] = []
        for style in styles:
            agent = LiveAgent(self.context, self.pool, prompt_style=style)
            scores = []
            examples = []
            for message in validation_messages:
                result = agent.respond(message)
                score = self.evaluator.score_response(message, result)
                scores.append(score["total"])
                examples.append({"message": message, "response": result["response"], "score": score})
            results.append(
                {
                    "style": style,
                    "average_score": round(sum(scores) / len(scores), 3),
                    "examples": examples,
                }
            )

        results.sort(key=lambda item: item["average_score"], reverse=True)
        return {"best_style": results[0]["style"], "ranked_results": results}
