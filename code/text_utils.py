from __future__ import annotations

from math import sqrt
import re


TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in TOKEN_RE.findall(text.lower()):
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", raw_token)
        if chinese_chars:
            tokens.extend(chinese_chars)
            tokens.extend(
                "".join(chinese_chars[index : index + 2])
                for index in range(len(chinese_chars) - 1)
            )
        else:
            tokens.append(raw_token)
    return tokens


def cosine_similarity(left: str, right: str) -> float:
    left_tokens = tokenize(left)
    right_tokens = tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0

    left_counts: dict[str, int] = {}
    right_counts: dict[str, int] = {}
    for token in left_tokens:
        left_counts[token] = left_counts.get(token, 0) + 1
    for token in right_tokens:
        right_counts[token] = right_counts.get(token, 0) + 1

    shared = set(left_counts) & set(right_counts)
    dot = sum(left_counts[token] * right_counts[token] for token in shared)
    left_norm = sqrt(sum(value * value for value in left_counts.values()))
    right_norm = sqrt(sum(value * value for value in right_counts.values()))
    return dot / (left_norm * right_norm)
