"""
agent_graph.py — LangGraph-based agent workflow (Issue #3).

Replaces the simple ConversationChain with a proper state-graph that can:
  • Decide whether to use a tool or respond directly
  • Route to the correct tool
  • Collect tool results and generate a final answer
  • Maintain conversation memory across turns

Architecture
------------
             ┌────────────┐
      ┌─────→│   Agent    │─────┐
      │      │ (LLM node) │     │
      │      └────────────┘     │
      │           │             │
      │      tool_calls?     no tools
      │           │             │
      │           ▼             ▼
      │      ┌──────────┐  ┌────────┐
      │      │  Tools   │  │  END   │
      │      │  (exec)  │  └────────┘
      │      └──────────┘
      │           │
      └───────────┘
"""

from __future__ import annotations

import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from config import settings
from skills import ALL_SKILLS

# ----- Tuned system prompt  (Issue #2 — prompt tuning) ---------------------
SYSTEM_PROMPT = """\
You are Igris, a powerful AI assistant forged to serve your user with precision and loyalty.

Core directives:
1. THINK step-by-step before answering complex questions.
2. When the user asks you to perform an action (search, read files, calculate, \
   control the system, read documents, translate, write code), USE THE APPROPRIATE TOOL. \
   Do not guess — call the tool and report the real result.
3. For system operations (shutdown, reboot, sleep, lock), ALWAYS confirm with the user first.
4. Be concise but thorough. Provide context for your answers.
5. If a tool fails, explain what went wrong and suggest alternatives.
6. You can read documents in PDF, DOCX, TXT, and CSV formats. Offer this when users mention files.
7. For translation requests, use the translate_text tool.
8. Address the user as 'Your Majesty' as a mark of respect.

Available capabilities: web search, file operations, math, code execution, \
text summarisation, system control, document reading, translation.
"""


# ----- State definition ---------------------------------------------------
class AgentState(TypedDict):
    """State passed between graph nodes."""
    messages: Annotated[Sequence[BaseMessage], operator.add]


# ----- Build the LLM with tools bound ------------------------------------

def create_llm():
    """Create the Groq LLM with increased capacity settings (Issue #2)."""
    llm = ChatGroq(
        model=settings.model_name,
        api_key=settings.groq_api_key,
        temperature=settings.model_temperature,
        max_tokens=settings.model_max_tokens,
        streaming=True,  # Stream tokens for better perceived throughput
    )
    return llm


def create_agent_graph():
    """Build and compile the LangGraph agent workflow."""

    llm = create_llm()
    llm_with_tools = llm.bind_tools(ALL_SKILLS)

    # --- Node: agent (calls the LLM) -----
    def agent_node(state: AgentState) -> dict:
        """Invoke the LLM with the current message history."""
        messages = state["messages"]

        # Inject system prompt if not already present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # --- Node: tools (executes tool calls) -----
    tool_node = ToolNode(ALL_SKILLS)

    # --- Routing function -----
    def should_continue(state: AgentState) -> str:
        """Decide whether to invoke tools or end the turn."""
        last_message = state["messages"][-1]

        # If the LLM returned tool_calls, route to the tool node
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"

    # --- Assemble the graph -----
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )

    # After tools execute, loop back to agent so it can use the results
    graph.add_edge("tools", "agent")

    return graph.compile()


# ----- Convenience: run a single turn ------------------------------------

def run_agent_turn(graph, history: list[BaseMessage], user_input: str) -> tuple[str, list[BaseMessage]]:
    """Run one conversational turn through the agent graph.

    Args:
        graph: The compiled LangGraph.
        history: Previous messages (maintained by the caller).
        user_input: The user's new message.

    Returns:
        (response_text, updated_history)
    """
    user_msg = HumanMessage(content=user_input)
    input_messages = list(history) + [user_msg]

    result = graph.invoke({"messages": input_messages})

    # The final message in the result is the agent's response
    all_messages = result["messages"]
    final_msg = all_messages[-1]
    response_text = final_msg.content if hasattr(final_msg, "content") else str(final_msg)

    # Update history with the new user message and the agent's answer
    updated_history = list(history) + [user_msg, AIMessage(content=response_text)]

    return response_text, updated_history
