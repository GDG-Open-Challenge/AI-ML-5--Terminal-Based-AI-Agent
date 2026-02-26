# ⚔ Igris — Terminal-Based AI Agent

A powerful terminal-based AI agent built with **LangChain**, **LangGraph**, and **Groq** (Llama 3.1 70B). Igris can search the web, read documents, execute code, control your system, and hold natural conversations — all from your terminal.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Web Search** | Real-time information via DuckDuckGo |
| 📄 **Document Reading** | PDF, DOCX, TXT, CSV ingestion |
| 📁 **File Operations** | Read, write, and list files |
| 🧮 **Math Evaluator** | Safe mathematical expression evaluation |
| 🐍 **Code Execution** | Run Python snippets in a sandboxed subprocess |
| 📝 **Summarisation** | Condense long texts into key points |
| 💻 **System Control** | Shutdown, reboot, sleep, lock (with confirmation) |
| 🌐 **Translation** | Translate between 100+ languages |
| 🧠 **Persistent Memory** | Corruption-resistant memory with atomic writes |
| 🔄 **LangGraph Workflows** | State-graph agent with tool routing |

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
cp .env.example .env
# Edit .env and paste your Groq API key
```

Get your free API key at [console.groq.com](https://console.groq.com).

### 3. Run the enhanced agent

```bash
python Igris-Enhanced.py
```

## 📂 Project Structure

```
├── Igris-Enhanced.py      # Main entry point (enhanced)
├── Igris-Beta.py          # Original beta (patched)
├── Igris.py               # Original with RAG (patched)
├── config.py              # Pydantic-validated configuration
├── memory.py              # Atomic memory persistence
├── skills.py              # OpenClaw skill base (10 tools)
├── agent_graph.py         # LangGraph agent workflow
├── document_loader.py     # Document ingestion pipeline
├── requirements.txt       # All dependencies
├── .env.example           # Environment variable template
└── documents/             # Drop files here for ingestion
```

## 🛠 Commands

| Command | Action |
|---------|--------|
| `help` | Show available commands |
| `skills` | List all agent skills |
| `ingest` | Index documents from `./documents` |
| `clear` | Clear conversation history |
| `quit` | Save memory and exit |

## 🔧 Configuration

All settings are in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Your Groq API key (required) |
| `MODEL_NAME` | `llama-3.1-70b-versatile` | Groq model to use |
| `MODEL_TEMPERATURE` | `0.7` | Response creativity (0–2) |
| `MODEL_MAX_TOKENS` | `2048` | Max tokens per response |

## 📋 Issues Addressed

### Issue #1 — OpenClaw Skill Base
Added 10 skills with LangChain `@tool` decorators: web search, file read/write/list, math evaluator, code executor, summariser, system control, document reader, translation. Each skill has a clear docstring the LLM uses for tool selection.

### Issue #2 — Model Capacity Increase
- **Model upgrade**: `llama3-8b-8192` → `llama-3.1-70b-versatile` (8.75× more parameters)
- **Token limit**: 500 → 2048
- **Prompt tuning**: System prompt rewritten with explicit tool-usage directives
- **Streaming**: Enabled for better perceived throughput

### Issue #3 — LangGraph + Pydantic AI + Document Reading
- **LangGraph**: Full state-graph agent with agent→tools→agent loop
- **Pydantic**: `pydantic-settings` for validated configuration
- **Documents**: PDF, DOCX, TXT, CSV loading with FAISS vector store

### System Control Model
Shutdown, reboot, sleep, lock screen operations with 60-second delay and cancellation support. All destructive operations require explicit `yes` confirmation.

### Memory Corruption Fix
- **Atomic writes**: Write to temp file → `os.replace` (never half-written)
- **Backup**: `.bak` copy before every save
- **Recovery**: Auto-fallback to backup if primary is corrupt
- **Real-time**: Memory saved after every exchange, not just on quit

## 📜 License

MIT License — see [LICENSE](LICENSE).
