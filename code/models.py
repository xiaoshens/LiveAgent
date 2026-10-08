from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Product:
    name: str
    price: str
    selling_points: list[str]
    promotion: str
    compliance_notes: list[str]


@dataclass
class LiveContext:
    streamer_name: str
    persona: str
    room_goal: str
    product: Product


@dataclass
class Trajectory:
    session_id: str
    scene: str
    user_profile: str
    user_utterance: str
    agent_response: str
    outcome: dict[str, float]
    human_feedback: str = ""

    @property
    def score(self) -> float:
        return (
            self.outcome.get("stay_seconds", 0) / 60 * 0.25
            + self.outcome.get("dialogue_turns", 0) * 0.2
            + self.outcome.get("click", 0) * 1.5
            + self.outcome.get("follow", 0) * 1.0
            + self.outcome.get("risk_free", 1) * 0.8
        )

    @property
    def label(self) -> str:
        if self.outcome.get("risk_free", 1) < 1:
            return "failed"
        if self.score >= 2.4:
            return "success"
        if self.score >= 1.7:
            return "neutral"
        return "failed"

    def searchable_text(self) -> str:
        return " ".join(
            [
                self.scene,
                self.user_profile,
                self.user_utterance,
                self.agent_response,
                self.human_feedback,
            ]
        )


@dataclass
class Insight:
    insight_id: str
    scene: str
    rule: str
    evidence: list[str]
    score: float
    risk: str = "不得虚构价格、库存、疗效、平台规则或无法确认的信息。"

    def searchable_text(self) -> str:
        return f"{self.scene} {self.rule} {self.risk}"
