from __future__ import annotations

from models import LiveContext
from text_utils import tokenize


class OfflineEvaluator:
    def __init__(self, context: LiveContext) -> None:
        self.context = context

    def score_response(self, user_message: str, agent_result: dict[str, object]) -> dict[str, float]:
        response = str(agent_result["response"])
        product = self.context.product
        relevance = self._keyword_score(user_message, response)
        persona = 1.0 if self.context.streamer_name in response or "口吻" in response else 0.6
        product_grounding = sum(point in response for point in product.selling_points) / len(
            product.selling_points
        )
        safety = 0.2 if any(word in response for word in ["绝对", "全网最低", "保证治好"]) else 1.0
        conciseness = 1.0 if len(response) <= 180 else 0.65
        engagement = 1.0 if "？" in response or "?" in response else 0.6
        business_guidance = 1.0 if any(word in response for word in ["券", "商品卡", "关注", "下单页"]) else 0.7

        total = (
            relevance * 0.2
            + persona * 0.1
            + product_grounding * 0.2
            + safety * 0.2
            + conciseness * 0.1
            + engagement * 0.1
            + business_guidance * 0.1
        )
        return {
            "relevance": round(relevance, 3),
            "persona": round(persona, 3),
            "product_grounding": round(product_grounding, 3),
            "safety": round(safety, 3),
            "conciseness": round(conciseness, 3),
            "engagement": round(engagement, 3),
            "business_guidance": round(business_guidance, 3),
            "total": round(total, 3),
        }

    @staticmethod
    def _keyword_score(user_message: str, response: str) -> float:
        user_tokens = set(tokenize(user_message))
        response_tokens = set(tokenize(response))
        if not user_tokens:
            return 0.5
        overlap = len(user_tokens & response_tokens) / max(1, len(user_tokens))
        return min(1.0, 0.45 + overlap)
