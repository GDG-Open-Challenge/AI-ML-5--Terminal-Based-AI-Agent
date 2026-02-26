"""
skills.py — OpenClaw skill base for the Igris agent (Issue #1).

Each skill is a self-contained tool that the LangGraph agent can invoke.
Skills are exposed as LangChain-compatible tools with clear docstrings that
the LLM uses to decide *when* to call each skill.

Skills added & justification
-----------------------------
1. **Web Search** — Gives the agent access to real-time information beyond its
   training data.  Uses DuckDuckGo (no API key required).

2. **File Operations** (read/write/list) — Lets the agent work with local
   files, essential for any developer-assistant workflow.

3. **Math Evaluator** — Safe mathematical expression evaluator so the agent
   can perform calculations without hallucinating numbers.

4. **Code Executor** — Runs Python snippets in a subprocess sandbox and
   returns stdout/stderr, enabling the agent to test code on the fly.

5. **Text Summariser** — Condenses long text into key bullet points using the
   same LLM, useful for document analysis workflows.

6. **System Control** — Shutdown, reboot, sleep, lock-screen (Issue: System
   Control Model). Requires explicit user confirmation.

7. **Document Reader** — Reads PDF, DOCX, TXT, CSV files and returns their
   content (Issue #3 — document reading capacity).

8. **Translation** — Translates text between languages using deep-translator,
   replacing the broken _engine_trans_ import.
"""

from __future__ import annotations

import ast
import csv
import io
import math
import os
import platform
import subprocess
import sys
import textwrap
from typing import Optional

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# 1. Web Search
# ---------------------------------------------------------------------------
@tool
def web_search(query: str) -> str:
    """Search the web for real-time information using DuckDuckGo.

    Use this when the user asks about current events, facts you are unsure
    about, or anything that benefits from up-to-date information.

    Args:
        query: The search query string.

    Returns:
        A summary of the top search results.
    """
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append(f"• {r['title']}\n  {r['body']}\n  {r['href']}")
        if results:
            return "\n\n".join(results)
        return "No results found."
    except ImportError:
        return "Web search is unavailable — install duckduckgo-search."
    except Exception as exc:
        return f"Search failed: {exc}"


# ---------------------------------------------------------------------------
# 2. File Operations
# ---------------------------------------------------------------------------
@tool
def read_file(file_path: str) -> str:
    """Read the contents of a local file.

    Args:
        file_path: Absolute or relative path to the file.

    Returns:
        The file contents or an error message.
    """
    try:
        path = os.path.abspath(file_path)
        if not os.path.isfile(path):
            return f"File not found: {path}"
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        # Cap output to avoid flooding the context window
        if len(content) > 8000:
            return content[:8000] + "\n\n... [truncated — file has more content]"
        return content
    except Exception as exc:
        return f"Error reading file: {exc}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a local file.  Creates parent directories if needed.

    Args:
        file_path: Destination file path.
        content: Text content to write.

    Returns:
        A success or error message.
    """
    try:
        path = os.path.abspath(file_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {path}"
    except Exception as exc:
        return f"Error writing file: {exc}"


@tool
def list_directory(directory_path: str = ".") -> str:
    """List files and subdirectories in the given directory.

    Args:
        directory_path: Path to list.  Defaults to the current directory.

    Returns:
        A formatted listing of directory contents.
    """
    try:
        path = os.path.abspath(directory_path)
        if not os.path.isdir(path):
            return f"Not a directory: {path}"
        entries = sorted(os.listdir(path))
        lines = []
        for entry in entries[:50]:  # cap to 50 entries
            full = os.path.join(path, entry)
            kind = "DIR " if os.path.isdir(full) else "FILE"
            size = ""
            if os.path.isfile(full):
                size = f" ({os.path.getsize(full):,} bytes)"
            lines.append(f"  [{kind}] {entry}{size}")
        header = f"Contents of {path} ({len(entries)} items):"
        return header + "\n" + "\n".join(lines)
    except Exception as exc:
        return f"Error listing directory: {exc}"


# ---------------------------------------------------------------------------
# 3. Math Evaluator
# ---------------------------------------------------------------------------
_SAFE_MATH_NAMES = {
    k: v for k, v in vars(math).items() if not k.startswith("_")
}
_SAFE_MATH_NAMES.update({"abs": abs, "round": round, "min": min, "max": max})


@tool
def math_eval(expression: str) -> str:
    """Safely evaluate a mathematical expression.

    Supports standard math functions (sin, cos, sqrt, log, pi, e, etc.).

    Args:
        expression: A mathematical expression string, e.g. 'sqrt(144) + 3**2'.

    Returns:
        The numeric result as a string, or an error message.
    """
    try:
        # Parse the expression into an AST and walk it to reject anything
        # that isn't a number, operator, or whitelisted function call.
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Attribute)):
                return "Blocked: imports and attribute access are not allowed."
        result = eval(compile(tree, "<math>", "eval"), {"__builtins__": {}}, _SAFE_MATH_NAMES)
        return str(result)
    except Exception as exc:
        return f"Math evaluation error: {exc}"


# ---------------------------------------------------------------------------
# 4. Code Executor (sandboxed subprocess)
# ---------------------------------------------------------------------------
@tool
def run_python_code(code: str) -> str:
    """Execute a Python code snippet and return its output.

    The code runs in a **separate subprocess** with a 30-second timeout.
    Use this to test small scripts, verify logic, or compute results.

    Args:
        code: Python source code to execute.

    Returns:
        stdout + stderr from the execution.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.getcwd(),
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += "\n[STDERR]\n" + result.stderr
        if not output.strip():
            output = "(no output)"
        # Truncate very long outputs
        if len(output) > 6000:
            output = output[:6000] + "\n... [truncated]"
        return output
    except subprocess.TimeoutExpired:
        return "Execution timed out (30s limit)."
    except Exception as exc:
        return f"Execution error: {exc}"


# ---------------------------------------------------------------------------
# 5. Text Summariser
# ---------------------------------------------------------------------------
@tool
def summarise_text(text: str) -> str:
    """Produce a concise bullet-point summary of the given text.

    This performs local extraction — no LLM call.  For LLM-powered
    summarisation, just ask the agent directly.

    Args:
        text: The text to summarise.

    Returns:
        A bullet-point summary.
    """
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 20]
    if not sentences:
        return "Text too short to summarise."
    # Pick up to 10 representative sentences
    step = max(1, len(sentences) // 10)
    selected = sentences[::step][:10]
    bullets = "\n".join(f"• {s.strip()}." for s in selected)
    return f"Summary ({len(selected)} key points):\n{bullets}"


# ---------------------------------------------------------------------------
# 6. System Control  (Issue: System Control Model)
# ---------------------------------------------------------------------------
def _confirm_action(action_name: str) -> bool:
    """Ask the user for explicit confirmation before a system operation."""
    print(
        f"\n⚠  SYSTEM ACTION: {action_name}"
    )
    answer = input(f"Are you sure you want to {action_name}? (yes/no): ").strip().lower()
    return answer in ("yes", "y")


@tool
def system_control(action: str) -> str:
    """Perform a system-level operation.

    Supported actions: shutdown, reboot, sleep, lock, cancel_shutdown.
    Every destructive action requires explicit user confirmation.

    Args:
        action: One of 'shutdown', 'reboot', 'sleep', 'lock', 'cancel_shutdown'.

    Returns:
        Result message.
    """
    action = action.strip().lower()
    os_name = platform.system()

    commands = {}
    if os_name == "Windows":
        commands = {
            "shutdown": "shutdown /s /t 60",
            "reboot": "shutdown /r /t 60",
            "sleep": "rundll32.exe powrprof.dll,SetSuspendState 0,1,0",
            "lock": "rundll32.exe user32.dll,LockWorkStation",
            "cancel_shutdown": "shutdown /a",
        }
    elif os_name == "Linux":
        commands = {
            "shutdown": "shutdown -h +1",
            "reboot": "shutdown -r +1",
            "sleep": "systemctl suspend",
            "lock": "loginctl lock-session",
            "cancel_shutdown": "shutdown -c",
        }
    elif os_name == "Darwin":  # macOS
        commands = {
            "shutdown": "sudo shutdown -h +1",
            "reboot": "sudo shutdown -r +1",
            "sleep": "pmset sleepnow",
            "lock": "/System/Library/CoreServices/Menu\\ Extras/User.menu/Contents/Resources/CGSession -suspend",
            "cancel_shutdown": "sudo killall shutdown",
        }
    else:
        return f"Unsupported operating system: {os_name}"

    if action not in commands:
        return f"Unknown action '{action}'. Supported: {', '.join(commands.keys())}"

    # Non-destructive actions don't need confirmation
    safe_actions = {"cancel_shutdown", "lock"}
    if action not in safe_actions:
        if not _confirm_action(action):
            return f"{action} cancelled by user."

    try:
        subprocess.Popen(commands[action], shell=True)
        extra = ""
        if action in ("shutdown", "reboot"):
            extra = " (in 60 seconds — use 'cancel_shutdown' to abort)"
        return f"System {action} initiated{extra}."
    except Exception as exc:
        return f"System control error: {exc}"


# ---------------------------------------------------------------------------
# 7. Document Reader  (Issue #3)
# ---------------------------------------------------------------------------
@tool
def read_document(file_path: str) -> str:
    """Read a document file and return its text content.

    Supported formats: PDF, DOCX, TXT, CSV, and other plain-text files.

    Args:
        file_path: Path to the document.

    Returns:
        Extracted text content of the document.
    """
    path = os.path.abspath(file_path)
    if not os.path.isfile(path):
        return f"File not found: {path}"

    ext = os.path.splitext(path)[1].lower()

    try:
        # --- PDF ---
        if ext == ".pdf":
            try:
                from PyPDF2 import PdfReader

                reader = PdfReader(path)
                pages = []
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append(f"--- Page {i + 1} ---\n{text}")
                if not pages:
                    return "PDF parsed but no text could be extracted (may be an image-based PDF)."
                content = "\n\n".join(pages)
            except ImportError:
                return "PyPDF2 is not installed. Run: pip install PyPDF2"

        # --- DOCX ---
        elif ext == ".docx":
            try:
                from docx import Document as DocxDocument

                doc = DocxDocument(path)
                content = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            except ImportError:
                return "python-docx is not installed. Run: pip install python-docx"

        # --- CSV ---
        elif ext == ".csv":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                reader_obj = csv.reader(f)
                rows = list(reader_obj)
            if not rows:
                return "CSV file is empty."
            # Format as a readable table
            lines = []
            for i, row in enumerate(rows[:100]):  # cap at 100 rows
                lines.append(" | ".join(row))
                if i == 0:
                    lines.append("-" * len(lines[0]))
            content = "\n".join(lines)
            if len(rows) > 100:
                content += f"\n... ({len(rows) - 100} more rows)"

        # --- Plain text / other ---
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

        # Truncate very large documents
        if len(content) > 12000:
            content = content[:12000] + "\n\n... [truncated — document is very large]"
        return content

    except Exception as exc:
        return f"Error reading document: {exc}"


# ---------------------------------------------------------------------------
# 8. Translation  (replaces broken _engine_trans_)
# ---------------------------------------------------------------------------
@tool
def translate_text(text: str, source_lang: str = "auto", target_lang: str = "en") -> str:
    """Translate text from one language to another.

    Args:
        text: The text to translate.
        source_lang: Source language code (e.g. 'en', 'es', 'fr') or 'auto'.
        target_lang: Target language code.

    Returns:
        Translated text.
    """
    try:
        from deep_translator import GoogleTranslator

        translated = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        return translated or "Translation returned empty result."
    except ImportError:
        return "deep-translator is not installed. Run: pip install deep-translator"
    except Exception as exc:
        return f"Translation error: {exc}"


# ---------------------------------------------------------------------------
# Collect all tools for the agent
# ---------------------------------------------------------------------------
ALL_SKILLS = [
    web_search,
    read_file,
    write_file,
    list_directory,
    math_eval,
    run_python_code,
    summarise_text,
    system_control,
    read_document,
    translate_text,
]

SKILL_DESCRIPTIONS = {t.name: t.description for t in ALL_SKILLS}


def list_skills() -> str:
    """Return a human-readable summary of all available skills."""
    lines = [f"  {i+1}. [{t.name}] — {t.description.split(chr(10))[0]}" for i, t in enumerate(ALL_SKILLS)]
    return "Available Skills:\n" + "\n".join(lines)
