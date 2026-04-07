from collections.abc import Generator
from typing import Any

from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from .config import settings
from .prompts import SYSTEM_PROMPT
from .tools import all_tools


_PROMPT_BAKED_IN = True


def _build_input(user_message: str, include_system_prompt: bool = False) -> dict[str, Any]:
    if include_system_prompt:
        return {"messages": [("system", SYSTEM_PROMPT), ("user", user_message)]}
    return {"messages": [("user", user_message)]}


def build_agent():
    global _PROMPT_BAKED_IN

    llm = ChatOllama(
        model=settings.reasoning_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        num_ctx=settings.reasoning_num_ctx,
        num_predict=settings.reasoning_num_predict,
        keep_alive=settings.reasoning_keep_alive,
    )

    try:
        agent = create_react_agent(
            model=llm,
            tools=all_tools,
            prompt=SYSTEM_PROMPT,
        )
        _PROMPT_BAKED_IN = True
    except TypeError:
        agent = create_react_agent(
            model=llm,
            tools=all_tools,
        )
        _PROMPT_BAKED_IN = False

    return agent


def invoke_agent(agent, user_message: str) -> dict[str, Any]:
    result = agent.invoke(
        _build_input(user_message, include_system_prompt=not _PROMPT_BAKED_IN),
        config={"recursion_limit": settings.max_iterations * 2},
    )
    return result


def stream_agent(agent, user_message: str) -> Generator[dict[str, Any], None, None]:
    stream = agent.stream(
        _build_input(user_message, include_system_prompt=not _PROMPT_BAKED_IN),
        config={"recursion_limit": settings.max_iterations * 2},
        stream_mode="values",
    )
    for step in stream:
        yield step


def invoke_agent_with_messages(agent, messages: list[tuple[str, str]]) -> dict[str, Any]:
    payload: dict[str, Any] = {"messages": messages}
    if not _PROMPT_BAKED_IN:
        payload = {"messages": [("system", SYSTEM_PROMPT), *messages]}
    result = agent.invoke(
        payload,
        config={"recursion_limit": settings.max_iterations * 2},
    )
    return result


def stream_agent_with_messages(
    agent,
    messages: list[tuple[str, str]],
) -> Generator[dict[str, Any], None, None]:
    payload: dict[str, Any] = {"messages": messages}
    if not _PROMPT_BAKED_IN:
        payload = {"messages": [("system", SYSTEM_PROMPT), *messages]}

    stream = agent.stream(
        payload,
        config={"recursion_limit": settings.max_iterations * 2},
        stream_mode="values",
    )
    for step in stream:
        yield step
