from __future__ import annotations

import json

from agent import LiveAgent
from demo_data import build_demo_context, build_seed_experiences, mock_llm_client
from evaluator import OfflineEvaluator
from experience_pool import ExperiencePool
from optimizer import PromptOptimizer


def print_json(title: str, payload: object) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    context = build_demo_context()
    pool = ExperiencePool()
    for trajectory in build_seed_experiences():
        pool.add_trajectory(trajectory)

    insight_mode = "llm"  # 可改为 "rule" 使用规则版；"llm" 会调用 llm_client。
    insights = pool.extract_insights(mode=insight_mode, llm_client=mock_llm_client)
    print_json("提取出的经验洞察", [insight.__dict__ for insight in insights])

    evaluator = OfflineEvaluator(context)
    optimizer = PromptOptimizer(context, pool, evaluator)
    validation_messages = [
        "这个杯子多少钱，有优惠吗？",
        "有点贵，我想再看看",
        "能保温一整天吗？",
        "今晚活动券怎么领？",
        "收到漏水可以退吗？",
    ]
    optimization_result = optimizer.optimize(validation_messages)
    print_json(
        "提示词策略离线评估排名",
        [
            {"style": item["style"], "average_score": item["average_score"]}
            for item in optimization_result["ranked_results"]
        ],
    )

    best_style = str(optimization_result["best_style"])
    agent = LiveAgent(context, pool, prompt_style=best_style)
    demo_message = "这个杯子多少钱？我主要通勤用，怕漏水。"
    agent_result = agent.respond(demo_message, user_profile="价格敏感型通勤用户")
    score = evaluator.score_response(demo_message, agent_result)
    print_json("新用户输入", {"message": demo_message, "best_style": best_style})
    print_json("Agent 输出", agent_result)
    print_json("自动评估", score)


if __name__ == "__main__":
    main()
