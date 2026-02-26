"""
Igris-Enhanced.py — Enhanced terminal-based AI agent.

This is the upgraded main entry point that integrates all fixes and features:
  • Issue #1 — OpenClaw skills (web search, file ops, math, code exec, etc.)
  • Issue #2 — Model capacity increase (llama-3.1-70b, tuned prompts, streaming)
  • Issue #3 — LangGraph workflows + Pydantic AI config + document reading
  • System Control — Shutdown, reboot, sleep, lock with confirmations
  • Memory Fix — Atomic writes, backup/recovery, real-time saves

Original files (Igris.py, Igris-Beta.py) are preserved — this file replaces
them as the primary entry point.
"""

import os
import sys

# Rich must be imported early for pretty tracebacks
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import print as rprint

console = Console()

# ── Bootstrap ──────────────────────────────────────────────────────────────
# Make sure we're running from the script's directory so relative paths work
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import settings
from memory import load_memory, save_memory
from skills import ALL_SKILLS, list_skills
from agent_graph import create_agent_graph, run_agent_turn, SYSTEM_PROMPT
from document_loader import load_documents_from_directory, build_vector_store, load_vector_store

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# ── Validate API key ──────────────────────────────────────────────────────
def _check_api_key() -> None:
    if not settings.groq_api_key or settings.groq_api_key == "your_groq_api_key_here":
        console.print(
            Panel(
                "[bold red]GROQ_API_KEY is not set.[/bold red]\n\n"
                "1. Copy [cyan].env.example[/cyan] to [cyan].env[/cyan]\n"
                "2. Paste your Groq API key\n"
                "3. Re-run this script",
                title="⚠  Configuration Required",
                border_style="red",
            )
        )
        sys.exit(1)


# ── Startup banner ────────────────────────────────────────────────────────
def _print_banner() -> None:
    banner = Text()
    banner.append("╔══════════════════════════════════════════╗\n", style="bold cyan")
    banner.append("║          ", style="bold cyan")
    banner.append("⚔  IGRIS — AI Agent  ⚔", style="bold white")
    banner.append("          ║\n", style="bold cyan")
    banner.append("║  ", style="bold cyan")
    banner.append("Enhanced Terminal-Based AI Assistant", style="dim white")
    banner.append("   ║\n", style="bold cyan")
    banner.append("╚══════════════════════════════════════════╝", style="bold cyan")
    console.print(banner)
    console.print()


def _print_help() -> None:
    table = Table(title="Commands", show_header=True, header_style="bold cyan")
    table.add_column("Command", style="green", min_width=18)
    table.add_column("Description", style="white")
    table.add_row("quit / exit", "Save memory and exit")
    table.add_row("skills", "List all available skills")
    table.add_row("ingest", "Index documents from the ./documents folder")
    table.add_row("clear", "Clear conversation history (keeps memory file)")
    table.add_row("help", "Show this help table")
    console.print(table)
    console.print()


# ── Document ingestion command ────────────────────────────────────────────
def _handle_ingest() -> None:
    console.print("\n[cyan]Scanning documents directory...[/cyan]")
    docs = load_documents_from_directory()
    if docs:
        console.print("[cyan]Building vector store...[/cyan]")
        build_vector_store(docs)
        console.print("[green]Documents indexed successfully![/green]\n")
    else:
        console.print(
            "[yellow]No documents found.  Place PDF/DOCX/TXT/CSV files in the "
            "'documents' folder and try again.[/yellow]\n"
        )


# ── Main loop ─────────────────────────────────────────────────────────────
def main() -> None:
    _check_api_key()
    _print_banner()

    # Load persistent memory
    memory = load_memory()
    history_messages = memory.chat_memory.messages.copy()

    # Build the LangGraph agent
    console.print("[dim]Initialising agent graph...[/dim]")
    try:
        graph = create_agent_graph()
    except Exception as exc:
        console.print(f"[red]Failed to initialise agent: {exc}[/red]")
        sys.exit(1)

    console.print(
        f"[green]Igris is ready.[/green]  Model: [cyan]{settings.model_name}[/cyan]  "
        f"| Max tokens: [cyan]{settings.model_max_tokens}[/cyan]  "
        f"| Skills: [cyan]{len(ALL_SKILLS)}[/cyan]"
    )
    console.print("[dim]Type 'help' for commands, 'quit' to exit.[/dim]\n")

    _print_help()

    while True:
        try:
            user_input = input("You ❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            user_input = "quit"

        if not user_input:
            continue

        # ── Built-in commands ──────────────────────────────────────────
        lower = user_input.lower()

        if lower in ("quit", "exit"):
            console.print("[dim]Saving memory...[/dim]")
            # Reconstruct memory object for saving
            memory.chat_memory.messages = history_messages
            save_memory(memory)
            console.print(
                "[bold cyan]Fare thee well, Your Majesty. I await thy next summons. ⚔[/bold cyan]"
            )
            break

        if lower == "help":
            _print_help()
            continue

        if lower == "skills":
            console.print(list_skills())
            continue

        if lower == "ingest":
            _handle_ingest()
            continue

        if lower == "clear":
            history_messages.clear()
            console.print("[yellow]Conversation history cleared.[/yellow]\n")
            continue

        # ── Run through the LangGraph agent ────────────────────────────
        try:
            console.print("[dim]Thinking...[/dim]", end="")
            response, history_messages = run_agent_turn(
                graph, history_messages, user_input,
            )
            # Clear the "Thinking..." line
            console.print("\r", end="")

            console.print(
                Panel(
                    response,
                    title="[bold white]Igris[/bold white]",
                    border_style="cyan",
                    padding=(0, 1),
                )
            )

            # ── Real-time memory save (fixes corruption issue) ─────────
            memory.chat_memory.messages = history_messages
            save_memory(memory)

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
            continue
        except Exception as exc:
            console.print(f"\n[red]Error: {exc}[/red]")
            console.print("[dim]Try rephrasing or type 'help' for commands.[/dim]\n")


if __name__ == "__main__":
    main()
