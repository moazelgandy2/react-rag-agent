import ast
import re
from dataclasses import dataclass
from enum import Enum

from .tools import calculator


class Route(str, Enum):
    CALCULATOR = "calculator"
    DIRECT = "direct"
    AGENT = "agent"


@dataclass
class OrchestrationDecision:
    route: Route
    reason: str


_MATH_CHARS = re.compile(r"^[0-9\s\+\-\*\/%\(\)\.,]+$")
_MATH_KEYWORDS = (
    "calculate",
    "what is",
    "how much",
    "percentage",
    "%",
    "plus",
    "minus",
    "times",
    "divide",
    "sum",
)


def decide_route(message: str) -> OrchestrationDecision:
    normalized = message.strip().lower()
    if _looks_like_math(normalized):
        return OrchestrationDecision(route=Route.CALCULATOR, reason="math intent")

    if _is_direct_conversation(normalized):
        return OrchestrationDecision(route=Route.DIRECT, reason="conversation intent")

    return OrchestrationDecision(route=Route.AGENT, reason="knowledge/reasoning intent")


def run_direct_reply(message: str) -> str:
    text = message.strip()
    if not text:
        return ""

    lower = text.lower()
    if any(token in lower for token in ("hi", "hello", "hey")):
        return "Hey! I am here and ready. What do you want to explore?"
    if "thank" in lower:
        return "Anytime. Want to continue with the next question?"
    if any(token in lower for token in ("how are you", "what's up", "hows it going")):
        return "I am doing great and ready to help."

    return "Got it. I can help with that—share the specific question and I will handle it."


def run_calculator_route(message: str) -> str:
    expression = _extract_expression(message)
    if not expression:
        return "I could not parse a valid math expression from that."
    return calculator.invoke({"expression": expression})


def _looks_like_math(message: str) -> bool:
    if not message:
        return False

    if _MATH_CHARS.match(message):
        return True

    has_keyword = any(keyword in message for keyword in _MATH_KEYWORDS)
    has_digit = any(char.isdigit() for char in message)
    return has_keyword and has_digit


def _is_direct_conversation(message: str) -> bool:
    if not message:
        return False

    direct_phrases = (
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "how are you",
        "good morning",
        "good evening",
    )
    return any(message == phrase or message.startswith(f"{phrase} ") for phrase in direct_phrases)


def _extract_expression(message: str) -> str | None:
    cleaned = message.strip().lower()
    replacements = {
        "what is": "",
        "calculate": "",
        "how much is": "",
        "plus": "+",
        "minus": "-",
        "times": "*",
        "multiplied by": "*",
        "x": "*",
        "divide by": "/",
        "divided by": "/",
        "percent of": "* 0.01 *",
    }
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)

    cleaned = cleaned.replace("?", " ").replace(",", " ").strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    candidates = [cleaned]
    digits_tokens = re.findall(r"[0-9\.]+|[\+\-\*\/%\(\)]", cleaned)
    if digits_tokens:
        candidates.append(" ".join(digits_tokens))

    for candidate in candidates:
        if not candidate:
            continue
        if _is_safe_expression(candidate):
            return candidate
    return None


def _is_safe_expression(expression: str) -> bool:
    if not _MATH_CHARS.match(expression):
        return False

    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError:
        return False

    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Load,
    )

    for node in ast.walk(parsed):
        if not isinstance(node, allowed_nodes):
            return False
    return True
