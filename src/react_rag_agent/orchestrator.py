import ast
import json
import re
from dataclasses import dataclass
from enum import Enum

from langchain_ollama import ChatOllama

from .cache import orchestrator_cache
from .config import settings
from .tools import calculator


class Route(str, Enum):
    CALCULATOR = "calculator"
    DIRECT = "direct"
    AGENT = "agent"


@dataclass
class OrchestrationDecision:
    route: Route
    reason: str
    source: str


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

_ORCHESTRATOR_PROMPT = """You are an orchestration router for a local AI system.
Choose exactly one route for each user message:

- calculator: arithmetic or numeric computation only
- direct: simple social conversation, acknowledgments, greetings, thanks
- agent: all knowledge requests, document-grounded questions, reasoning tasks, and anything uncertain

Always prefer agent if uncertain.

Return strict JSON only:
{"route":"agent|calculator|direct","reason":"short reason"}
"""


def decide_route(message: str) -> OrchestrationDecision:
    heuristic = _decide_route_heuristic(message)
    if not settings.orchestrator_enabled:
        return heuristic

    llm_decision = _decide_route_llm(message)
    if llm_decision is None:
        return heuristic
    return llm_decision


def run_direct_reply(message: str) -> str:
    text = message.strip()
    if not text:
        return ""

    lower = text.lower()
    if any(token in lower for token in ("hi", "hello", "hey")):
        return "Hey! Great to see you. What do you want to work on today?"
    if "thank" in lower:
        return "You are welcome. Want me to continue with the next step?"
    if any(token in lower for token in ("how are you", "what's up", "hows it going")):
        return "I am doing well and fully focused. What should we tackle next?"

    return "Got it. I am with you. Share the exact task and I will handle it."


def run_calculator_route(message: str) -> str:
    expression = _extract_expression(message)
    if not expression:
        return "I could not parse a valid math expression from that."
    return calculator.invoke({"expression": expression})


def _decide_route_llm(message: str) -> OrchestrationDecision | None:
    cache_key = message.strip().lower()
    cached = orchestrator_cache.get(cache_key)
    if cached is not None:
        try:
            route = Route(cached["route"])
            reason = cached.get("reason", "cached")
            source = cached.get("source", "cache")
            return OrchestrationDecision(route=route, reason=reason, source=source)
        except Exception:
            pass

    try:
        llm = ChatOllama(
            model=settings.orchestrator_model,
            base_url=settings.ollama_base_url,
            temperature=settings.orchestrator_temperature,
            num_ctx=settings.orchestrator_num_ctx,
            num_predict=settings.orchestrator_num_predict,
            keep_alive=settings.orchestrator_keep_alive,
        )
        response = llm.invoke(
            [
                ("system", _ORCHESTRATOR_PROMPT),
                ("user", f"Message: {message}"),
            ]
        )
        content = str(response.content).strip()
        payload = _parse_router_json(content)
        if payload is None:
            return None

        route_raw = str(payload.get("route", "")).strip().lower()
        reason = str(payload.get("reason", "llm route")).strip() or "llm route"
        if route_raw not in {Route.CALCULATOR.value, Route.DIRECT.value, Route.AGENT.value}:
            return None

        decision = OrchestrationDecision(route=Route(route_raw), reason=reason, source="llm")
        orchestrator_cache.set(
            cache_key,
            {
                "route": decision.route.value,
                "reason": decision.reason,
                "source": decision.source,
            },
        )
        return decision
    except Exception:
        return None


def _parse_router_json(content: str) -> dict | None:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _decide_route_heuristic(message: str) -> OrchestrationDecision:
    normalized = message.strip().lower()
    if _looks_like_math(normalized):
        return OrchestrationDecision(
            route=Route.CALCULATOR, reason="math intent", source="heuristic"
        )

    if _is_direct_conversation(normalized):
        return OrchestrationDecision(
            route=Route.DIRECT,
            reason="conversation intent",
            source="heuristic",
        )

    return OrchestrationDecision(
        route=Route.AGENT,
        reason="knowledge/reasoning intent",
        source="heuristic",
    )


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
