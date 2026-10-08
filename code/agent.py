from __future__ import annotations

import re

from experience_pool import ExperiencePool
from models import Insight, LiveContext


class LiveAgent:
    def __init__(self, context: LiveContext, pool: ExperiencePool, prompt_style: str) -> None:
        self.context = context
        self.pool = pool
        self.prompt_style = prompt_style

    def detect_scene(self, user_message: str) -> str:
        text = user_message.lower()
        if any(word in text for word in ["多少钱", "价格", "优惠", "便宜", "券"]):
            return "询价"
        if any(word in text for word in ["值不值", "纠结", "考虑", "有点贵", "再看看"]):
            return "犹豫"
        if any(word in text for word in ["效果", "功效", "能不能", "适合", "材质"]):
            return "商品功效"
        if any(word in text for word in ["活动", "规则", "怎么领", "什么时候", "满减"]):
            return "活动规则"
        if any(word in text for word in ["退", "换", "售后", "坏了", "物流"]):
            return "售后"
        return "通用互动"

    def respond(self, user_message: str, user_profile: str = "普通用户") -> dict[str, object]:
        scene = self.detect_scene(user_message)
        query = f"{scene} {user_profile} {user_message}"
        insights = self.pool.retrieve_insights(query)
        cases = self.pool.retrieve_success_cases(query)
        response = self._compose_response(scene, user_message, insights)
        action = self._choose_action(scene)
        return {
            "scene": scene,
            "action": action,
            "response": response,
            "used_insights": [insight.insight_id for insight in insights],
            "similar_cases": [case.session_id for case in cases],
        }

    def _compose_response(self, scene: str, user_message: str, insights: list[Insight]) -> str:
        product = self.context.product
        persona_prefix = f"{self.context.streamer_name}口吻："
        rule_hint = insights[0].rule if insights else "先直接回答问题，再自然引导继续互动。"

        if scene == "询价":
            body = (
                f"这款{product.name}现在{product.price}，{product.promotion}。"
                f"如果你主要看性价比，可以先关注{product.selling_points[0]}。"
                "你是自己用还是送人？我按场景帮你挑。"
            )
        elif scene == "犹豫":
            body = (
                f"纠结很正常。{product.name}比较适合看重{product.selling_points[0]}的人，"
                f"另外{product.selling_points[1]}也能减少后续踩坑。"
                "你最担心价格、效果，还是售后？"
            )
        elif scene == "商品功效":
            body = (
                f"按目前商品资料看，{product.name}的重点是{product.selling_points[0]}、"
                f"{product.selling_points[1]}。具体效果会因使用场景不同有差异，"
                "不确定的地方建议以商品详情页为准。"
            )
        elif scene == "活动规则":
            body = (
                f"活动可以分三步看：先领券，再看{product.promotion}，"
                "最后以下单页展示价格为准。你现在拍的话，我建议先把券领上。"
            )
        elif scene == "售后":
            body = (
                "先别急，如果是物流或质量问题，可以从订单页发起售后。"
                "具体时效和处理方式以平台客服为准，我可以先帮你梳理要准备的信息。"
            )
        else:
            body = (
                f"{product.name}今天主推的是{product.selling_points[0]}和"
                f"{product.selling_points[1]}。你可以把需求打在评论区，我按你的情况说。"
            )

        if self.prompt_style == "concise":
            body = self._make_concise(body)
        elif self.prompt_style == "conversion":
            body = f"{body} 觉得合适可以先点商品卡看看规格，别急着下单。"
        elif self.prompt_style == "safety":
            body = f"{body} 我不会替你虚构库存、最低价或绝对效果。"

        return f"{persona_prefix}{body}（策略依据：{rule_hint}）"

    @staticmethod
    def _make_concise(text: str) -> str:
        sentences = re.split(r"[。！？]", text)
        short = "。".join(sentence for sentence in sentences[:2] if sentence)
        return short + "。"

    @staticmethod
    def _choose_action(scene: str) -> str:
        return {
            "询价": "explain_price_and_coupon",
            "犹豫": "resolve_objection",
            "商品功效": "answer_with_product_knowledge",
            "活动规则": "explain_campaign_steps",
            "售后": "guide_after_sales",
        }.get(scene, "keep_interaction")
