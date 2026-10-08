from __future__ import annotations

import json

from models import LiveContext, Product, Trajectory


def build_demo_context() -> LiveContext:
    return LiveContext(
        streamer_name="小鹿",
        persona="亲切、节奏快、不强推，像懂产品的朋友。",
        room_goal="提升停留、评论互动和商品点击。",
        product=Product(
            name="轻盈保温杯",
            price="到手 79 元",
            selling_points=["316 不锈钢内胆", "一键弹盖防漏", "轻量便携", "6 小时保温"],
            promotion="今晚直播间有 10 元券，拍下以下单页为准",
            compliance_notes=["不得承诺全网最低价", "不得虚构库存", "不得夸大保温时长"],
        ),
    )


def build_seed_experiences() -> list[Trajectory]:
    return [
        Trajectory(
            session_id="session_001",
            scene="询价",
            user_profile="价格敏感型用户",
            user_utterance="这个杯子多少钱，有没有券？",
            agent_response="到手 79 元，直播间先领 10 元券，拍下以下单页为准。你要通勤用的话，轻量和防漏会比较实用。",
            outcome={"stay_seconds": 95, "dialogue_turns": 4, "click": 1, "follow": 0, "risk_free": 1},
            human_feedback="价格说明清楚，并补充了使用场景。",
        ),
        Trajectory(
            session_id="session_002",
            scene="询价",
            user_profile="价格敏感型用户",
            user_utterance="这个多少钱？",
            agent_response="质量很好，大家都很喜欢，赶紧拍。",
            outcome={"stay_seconds": 18, "dialogue_turns": 1, "click": 0, "follow": 0, "risk_free": 1},
            human_feedback="没有直接回答价格，强推感明显。",
        ),
        Trajectory(
            session_id="session_003",
            scene="犹豫",
            user_profile="观望用户",
            user_utterance="有点贵，我再看看。",
            agent_response="可以理解，主要看你是不是需要防漏和轻便。这个杯子适合通勤放包里，担心漏水的话会省心一些。",
            outcome={"stay_seconds": 80, "dialogue_turns": 3, "click": 1, "follow": 0, "risk_free": 1},
            human_feedback="先承认顾虑，再解释适用场景。",
        ),
        Trajectory(
            session_id="session_004",
            scene="商品功效",
            user_profile="谨慎用户",
            user_utterance="真的能保温一整天吗？",
            agent_response="商品资料标注 6 小时保温，实际还会受水温和环境影响，建议按详情页说明判断。",
            outcome={"stay_seconds": 70, "dialogue_turns": 3, "click": 1, "follow": 0, "risk_free": 1},
            human_feedback="没有夸大功效，合规。",
        ),
        Trajectory(
            session_id="session_005",
            scene="商品功效",
            user_profile="谨慎用户",
            user_utterance="真的能保温一整天吗？",
            agent_response="绝对可以保温一整天，放心买。",
            outcome={"stay_seconds": 12, "dialogue_turns": 1, "click": 0, "follow": 0, "risk_free": 0},
            human_feedback="夸大承诺，有合规风险。",
        ),
        Trajectory(
            session_id="session_006",
            scene="活动规则",
            user_profile="活动敏感用户",
            user_utterance="券怎么领？",
            agent_response="先点商品卡领 10 元券，再看下单页价格。活动今晚有效，别急，先确认规格。",
            outcome={"stay_seconds": 88, "dialogue_turns": 4, "click": 1, "follow": 1, "risk_free": 1},
            human_feedback="步骤清楚，节奏合适。",
        ),
        Trajectory(
            session_id="session_007",
            scene="售后",
            user_profile="售后用户",
            user_utterance="收到漏水能退吗？",
            agent_response="如果收到有问题，可以从订单页申请售后，具体处理以平台规则和客服为准。",
            outcome={"stay_seconds": 60, "dialogue_turns": 2, "click": 0, "follow": 1, "risk_free": 1},
            human_feedback="边界清晰，未越权承诺。",
        ),
    ]


def mock_llm_client(prompt: str) -> str:
    """Demo LLM client used for local runs.

    Replace this function with a real model call when you have an API key or an
    internal LLM service. The important contract is: input prompt, output JSON.
    """
    del prompt
    return json.dumps(
        [
            {
                "scene": "询价",
                "rule": "先直接回答价格和优惠，再结合用户用途补充一个最相关卖点，最后用一个轻问题促成继续互动。",
                "evidence": ["session_001", "session_002"],
                "score": 4.4,
                "risk": "不得虚构最低价、库存或未确认优惠。",
            },
            {
                "scene": "犹豫",
                "rule": "先接住用户的犹豫情绪，再把卖点绑定到使用场景，避免催促式成交。",
                "evidence": ["session_003"],
                "score": 4.2,
                "risk": "不得用制造焦虑的方式强推。",
            },
            {
                "scene": "商品功效",
                "rule": "严格依据商品资料回答效果边界；遇到绝对化问题时说明影响因素，并提示以下单页和详情页为准。",
                "evidence": ["session_004", "session_005"],
                "score": 4.6,
                "risk": "不得夸大功效或使用绝对承诺。",
            },
            {
                "scene": "活动规则",
                "rule": "把活动拆成领券、看规格、看下单页价格三个步骤，降低用户理解成本。",
                "evidence": ["session_006"],
                "score": 4.1,
                "risk": "不得承诺平台未展示的优惠。",
            },
            {
                "scene": "售后",
                "rule": "先安抚用户，再给出订单页售后路径；涉及退换细则时提示以平台客服和订单页为准。",
                "evidence": ["session_007"],
                "score": 4.3,
                "risk": "不得越权承诺退款、换货或赔付。",
            },
        ],
        ensure_ascii=False,
    )
