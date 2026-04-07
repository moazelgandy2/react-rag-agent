from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from langchain_core.messages import AIMessage, ToolMessage

from .agent import build_agent, stream_agent
from .config import settings

console = Console()


def main() -> None:
    console.print(
        Panel.fit(
            "Type 'quit' to exit\nType 'ingest' to re-run document ingestion",
            title="🤖 ReAct RAG Agent",
            subtitle=f"Model: {settings.reasoning_model} | Chroma: {settings.chroma_persist_dir}",
        )
    )

    try:
        agent = build_agent()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]❌ Failed to initialize agent: {exc}[/]")
        return

    console.print("Agent ready. Ask me anything about your documents.")

    while True:
        user_input = console.input("[bold green]You:[/] ").strip()

        if user_input.lower() in {"quit", "exit"}:
            break

        if user_input.lower() == "ingest":
            try:
                from .ingest import run_ingestion

                run_ingestion()
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]❌ Ingestion failed: {exc}[/]")
            continue

        if not user_input:
            continue

        try:
            for step in stream_agent(agent, user_input):
                messages = step.get("messages", [])
                if not messages:
                    continue

                last_message = messages[-1]
                if isinstance(last_message, AIMessage) and last_message.tool_calls:
                    for tool_call in last_message.tool_calls:
                        tool_name = tool_call.get("name", "unknown")
                        args = tool_call.get("args", {})
                        console.print(f"[bold yellow]🔧 Calling tool: {tool_name}({args})[/]")
                elif isinstance(last_message, ToolMessage):
                    content = str(last_message.content)
                    preview = f"{content[:200]}..." if len(content) > 200 else content
                    console.print(f"[dim]📋 Tool result: {preview}[/]")
                elif isinstance(last_message, AIMessage) and not last_message.tool_calls:
                    response_text = str(last_message.content)
                    console.print("[bold cyan]🤖 Agent:[/]")
                    console.print(Markdown(response_text))

        except ConnectionError:
            console.print(
                f"[red]❌ Cannot connect to Ollama. Is it running on {settings.ollama_base_url}?[/]"
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]❌ Error: {exc}[/]")


if __name__ == "__main__":
    main()
