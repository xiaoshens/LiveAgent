from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import json
import re

from models import Insight, Trajectory
from text_utils import cosine_similarity


@dataclass
class ExperiencePool:
    trajectories: list[Trajectory] = field(default_factory=list)
    insights: list[Insight] = field(default_factory=list)

    def add_trajectory(self, trajectory: Trajectory) -> None:
        self.trajectories.append(trajectory)

    def extract_insights(
        self,
        mode: str = "rule",
        llm_client: Callable[[str], str] | None = None,
    ) -> list[Insight]:
        """
        从轨迹提取可复用洞察
        Args:
            mode: "rule" uses the built-in rule-based extractor. Any other value
                uses the LLM branch.
            llm_client: A function that accepts a prompt string and returns a JSON
                string. The JSON must be a list of insight objects.
        """
        grouped: dict[str, list[Trajectory]] = {}
        for trajectory in self.trajectories:
            grouped.setdefault(trajectory.scene, []).append(trajectory)

        if mode != "rule":
            if llm_client is None:
                raise ValueError(
                    "mode='llm' requires llm_client. Example: "
                    "pool.extract_insights(mode='llm', llm_client=your_llm_func)"
                )
            prompt = self._build_insight_prompt(grouped)
            raw_response = llm_client(prompt)
            insights = self._parse_llm_insights(raw_response)
            self.insights = insights
            return insights

        insights: list[Insight] = []
        for scene, items in grouped.items():
            successes = [item for item in items if item.label == "success"]
            failures = [item for item in items if item.label == "failed"]
            if not successes:
                continue

            best = max(successes, key=lambda item: item.score)
            evidence = [best.session_id] + [item.session_id for item in failures[:2]]
            rule = self._summarize_rule(scene, best, failures)
            score = round(min(5.0, 3.5 + len(successes) * 0.4 - len(failures) * 0.1), 2)
            insights.append(
                Insight(
                    insight_id=f"insight_{len(insights) + 1:03d}",
                    scene=scene,
                    rule=rule,
                    evidence=evidence,
                    score=score,
                )
            )

        self.insights = insights
        return insights

    def _build_insight_prompt(self, grouped: dict[str, list[Trajectory]]) -> str:
        examples: list[dict[str, object]] = []
        for scene, items in grouped.items():
            examples.append(
                {
                    "scene": scene,
                    "trajectories": [
                        {
                            "session_id": item.session_id,
                            "label": item.label,
                            "score": round(item.score, 3),
                            "user_profile": item.user_profile,
                            "user_utterance": item.user_utterance,
                            "agent_response": item.agent_response,
                            "outcome": item.outcome,
                            "human_feedback": item.human_feedback,
                        }
                        for item in items
                    ],
                }
            )

        return (
            "你是一个视频直播 AI 分身策略分析专家。请根据直播互动轨迹，"
            "对每个场景总结一条可复用的策略洞察。\n"
            "要求：\n"
            "1. 对比 success 和 failed 轨迹，指出成功做法和失败规避点。\n"
            "2. rule 必须是可直接放入 Agent 提示词的中文策略。\n"
            "3. 不得虚构价格、库存、疗效、平台规则或无法确认的信息。\n"
            "4. 只返回 JSON，不要返回 Markdown。\n"
            "JSON 格式：\n"
            "[\n"
            "  {\n"
            '    "scene": "询价",\n'
            '    "rule": "先说明到手价和优惠结构，再...",\n'
            '    "evidence": ["session_001", "session_002"],\n'
            '    "score": 4.2,\n'
            '    "risk": "不得虚构价格、库存、疗效、平台规则或无法确认的信息。"\n'
            "  }\n"
            "]\n"
            "直播轨迹如下：\n"
            f"{json.dumps(examples, ensure_ascii=False, indent=2)}"
        )

    @staticmethod
    def _parse_llm_insights(raw_response: str) -> list[Insight]:
        content = raw_response.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError("LLM insight response must be a JSON list.")

        insights: list[Insight] = []
        for index, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                raise ValueError("Each LLM insight must be a JSON object.")

            scene = str(item.get("scene", "")).strip()
            rule = str(item.get("rule", "")).strip()
            if not scene or not rule:
                raise ValueError("Each LLM insight must include non-empty scene and rule.")

            evidence_raw = item.get("evidence", [])
            if isinstance(evidence_raw, list):
                evidence = [str(value) for value in evidence_raw]
            else:
                evidence = [str(evidence_raw)]

            try:
                score = float(item.get("score", 3.5))
            except (TypeError, ValueError):
                score = 3.5

            insights.append(
                Insight(
                    insight_id=str(item.get("insight_id") or f"insight_{index:03d}"),
                    scene=scene,
                    rule=rule,
                    evidence=evidence,
                    score=max(0.0, min(5.0, score)),
                    risk=str(
                        item.get(
                            "risk",
                            "不得虚构价格、库存、疗效、平台规则或无法确认的信息。",
                        )
                    ),
                )
            )
        return insights

    def retrieve_insights(self, query: str, top_k: int = 3) -> list[Insight]:
        scored = [
            (cosine_similarity(query, insight.searchable_text()), insight)
            for insight in self.insights
        ]
        scored.sort(key=lambda item: (item[0], item[1].score), reverse=True)
        return [insight for score, insight in scored[:top_k] if score > 0]

    def retrieve_success_cases(self, query: str, top_k: int = 2) -> list[Trajectory]:
        candidates = [item for item in self.trajectories if item.label == "success"]
        scored = [(cosine_similarity(query, item.searchable_text()), item) for item in candidates]
        scored.sort(key=lambda item: (item[0], item[1].score), reverse=True)
        return [item for score, item in scored[:top_k] if score > 0]

    @staticmethod
    def _summarize_rule(scene: str, best: Trajectory, failures: list[Trajectory]) -> str:
        if scene == "询价":
            return "先说明到手价和优惠结构，再补一句适用场景，最后用轻量问题引导用户继续互动。"
        if scene == "犹豫":
            return "先承认用户顾虑，再用具体使用场景和核心卖点降低决策成本，避免强推。"
        if scene == "商品功效":
            return "围绕商品资料回答，不夸大效果；对不确定信息要提醒以官方说明为准。"
        if scene == "活动规则":
            return "按步骤解释活动门槛、时间和领取方式，避免一次性堆太多信息。"
        if scene == "售后":
            return "先安抚情绪，再给出可执行路径；涉及平台政策时提示以订单页或客服为准。"
        failed_hint = failures[0].human_feedback if failures else "避免答非所问。"
        return f"复用高分回复的结构，重点避免：{failed_hint}"
