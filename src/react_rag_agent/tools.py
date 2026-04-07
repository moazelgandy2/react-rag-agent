import os

from langchain_core.tools import tool

from .retrieval import retrieve


@tool
def retrieve_documents(query: str) -> str:
    """Search the knowledge base for relevant information.
    Use this tool when you need to find facts, data, or context from the ingested documents.
    Input should be a specific search query — rephrase the user's question as a targeted search.
    """

    results = retrieve(query)
    if not results:
        return "No relevant documents found for this query."

    blocks: list[str] = []
    for index, result in enumerate(results, start=1):
        source = os.path.basename(str(result.get("source", "unknown")))
        blocks.append(
            f"[{index}] (Source: {source}, Page: {result.get('page', 'N/A')}, "
            f"Relevance: {result.get('relevance_score', 0.0)})\n"
            f"{result.get('content', '')}"
        )

    return "\n\n".join(blocks)


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.
    Use this when the user asks a question that requires arithmetic,
    such as adding numbers, percentages, or unit conversions.
    Input should be a valid Python math expression, e.g. "2 + 2" or "100 * 0.15".
    """

    allowed = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
    except Exception as exc:  # noqa: BLE001
        return f"Error evaluating expression: {exc}"
    return str(result)


all_tools = [retrieve_documents, calculator]
