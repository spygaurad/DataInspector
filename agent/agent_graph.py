# agent/agent_graph.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .runtime_ctx import set_df_payload, set_df_summary, set_sota_bundled
from .tools import (
    tool_describe_step,
    tool_inspect_dataset,
    tool_list_steps,
    tool_list_versions,
    tool_propose_plan,
    tool_reset_to_version,
    tool_run_step,
    tool_sota_preprocessing,
)


def _to_text(content: Any, limit: int = 4000) -> str:
    """Coerce any message content to a string that Gemini will accept."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        s = json.dumps(content, default=str, ensure_ascii=False)
    except Exception:
        s = str(content)
    # lightly truncate huge tool dumps
    return (s[:limit] + " …") if len(s) > limit else s

def _sanitize_messages(msgs: list[Any]) -> list[Any]:
    """Keep only system/human/assistant messages and ensure content is str."""
    clean = []
    for m in msgs or []:
        # Drop raw ToolMessage or unknown roles (Gemini doesn't accept them)
        role = getattr(m, "type", None) or getattr(m, "role", None) or ""
        if isinstance(m, ToolMessage) or role == "tool":
            # Optionally compress tool outputs into a short assistant line instead:
            txt = _to_text(getattr(m, "content", None))
            if txt:
                clean.append(AIMessage(content=f"[Tool result] {txt}"))
            continue

        c = _to_text(getattr(m, "content", None))
        if isinstance(m, SystemMessage):
            clean.append(SystemMessage(content=c))
        elif isinstance(m, HumanMessage):
            clean.append(HumanMessage(content=c))
        elif isinstance(m, AIMessage):
            clean.append(AIMessage(content=c))
        else:
            # Unknown BaseMessage; best-effort map by role string
            r = str(role).lower()
            if r == "system":
                clean.append(SystemMessage(content=c))
            elif r in ("human", "user"):
                clean.append(HumanMessage(content=c))
            elif r in ("assistant", "ai", "aimessage"):
                clean.append(AIMessage(content=c))
            # else: ignore silently
    return clean

TOOLS = [tool_inspect_dataset, tool_sota_preprocessing, tool_list_steps, tool_describe_step, tool_propose_plan, tool_run_step, tool_list_versions, tool_reset_to_version]

SYSTEM_PRIMER = (
    "You are a data-quality assistant.\n"
    "\n"
    "Workflow:\n"
    "1) Call inspect_dataset() to summarize columns/dtypes and GUESS task/label.\n"
    "   • If you are NOT SURE about the task (or the label for supervised tasks), ASK the user to confirm and END THE TURN.\n"
    "   • Do NOT call sota_preprocessing until the user explicitly confirms the task (and label if supervised).\n"
    "   Acceptable confirmations include messages like: "
    "   'task=classification label=HARDSHIP_INDEX', 'Task: regression', or 'Unsupervised'.\n"
    "2) After the user confirms, call sota_preprocessing(task, modality, ...) and PRESENT a brief 'SOTA Evidence' section (3–6 bullets with titles and links from the tool).\n"
    "3) Call list_steps() and map SOTA insights to the available tools. Produce a plan (no execution yet); cite up to 2 SOTA sources per step.\n"
    "4) Ask: 'Which step should we execute first?' Do NOT call run_step until the user explicitly picks.\n"
    "5) After the user picks, call describe_step(name) and list ONLY real parameters from the tool. Ask for missing/optional params and confirm them.\n"
    "6) Execute with run_step(name, params_json). Version controls inside params_json when relevant:\n"
    "   • source: 'current' | 'prev' | 'base' | '@-1' | '@-2' | <int>\n"
    "   • dry_run: true|false (preview without mutating)\n"
    "   • new_version: true|false (create new snapshot vs replace current)\n"
    "   Avoid loops: if the same step+params just ran, ask to change parameters or source.\n"
    "7) Summarize results; optionally call list_versions() and offer reset_to_version(spec). If helpful, research again before proposing next steps.\n"
    "\n"
    "Rules:\n"
    "- Return exactly one tool call at a time.\n"
    "- Never call sota_preprocessing before explicit task confirmation.\n"
    "- Never call run_step without an explicit user choice.\n"
    "- When users ask about parameters, use describe_step (or list_steps) and answer ONLY from tool output.\n"
    "- Reject parameters that are not in the tool signature.\n"
)


class AgentState(TypedDict):
    messages: List[Any]
    df_payload: Optional[Dict[str, Any]]
    results: List[Dict[str, Any]]
    steps_taken: int
    max_steps: int
    confirmed_step: Optional[str]
    confirmed_params: Dict[str, Any]
    last_task: Optional[str]
    plan: Optional[Dict[str, Any]]

def make_agent_node(llm):
    """LLM emits tool calls; we sanitize history and ALWAYS append an AIMessage."""
    llm_with_tools = llm.bind_tools(TOOLS)

    def _node(state: AgentState) -> AgentState:
        d = (state.get("df_payload") or {}).get("data", {})
        rows = len(d.get("data", []) or [])
        cols = len(d.get("columns", []) or [])
        shape_note = SystemMessage(content=f"Current dataset shape: {rows} rows × {cols} columns.")

        history = _sanitize_messages(state.get("messages", []))
        inputs = [SystemMessage(content=SYSTEM_PRIMER), *history, shape_note]

        ai = llm_with_tools.invoke(inputs)
        # guard: ensure we append an AIMessage object
        if not isinstance(ai, AIMessage):
            ai = AIMessage(content=_to_text(getattr(ai, "content", ai)))

        state["messages"] = state["messages"] + [ai]
        # debug
        # print("DEBUG roles after agent:", [getattr(m, "type", None) or getattr(m, "role", None) for m in state["messages"]])
        return state

    return _node

def tools_exec_node():
    """
    Execute tools only here, after injecting df_payload into runtime context.
    Also updates state with tool outputs (summary/SOTA/plan/step_result).
    """
    tool_node = ToolNode(TOOLS)

    def _node(state: AgentState) -> AgentState:
        # Inject dataset into runtime context BEFORE any tool executes
        set_df_payload(state.get("df_payload"))

        # If no dataset at all, be friendly and stop
        if state.get("df_payload") is None:
            state["messages"].append(type(state["messages"][-1])(content="I don't have a dataset yet. Please upload one."))
            return state

        # Hard gate: block run_step unless user confirmed a step
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        for c in tool_calls:
            if c.get("name") == "run_step":
                intended = (c.get("args") or {}).get("name")
                if intended and intended != state.get("confirmed_step"):
                    state["messages"].append(type(last)(content="I have a plan ready. Which step should we run first?"))
                    return state

        # Actually execute the tool(s) requested by the last assistant message
        out = tool_node.invoke({"messages": state["messages"]})
        # Append ONLY new ToolMessages; do NOT overwrite the conversation
        new_msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
        if not new_msgs:
            # fallback: if provider returned the whole list, take the tail
            if len(out["messages"]) > len(state["messages"]):
                new_msgs = out["messages"][len(state["messages"]):]
            else:
                new_msgs = out["messages"]

        state["messages"] = state["messages"] + new_msgs

        # Parse the most recent tool payload (dict in .content)
        payload = new_msgs[-1].content if new_msgs else None
        if isinstance(payload, dict):
            typ = payload.get("type")
            if typ == "dataset_summary":
                set_df_summary(payload)
                state["last_task"] = payload.get("task_guess")
            elif typ == "sota":
                set_sota_bundled(payload.get("bundled_results") or [])
            elif typ == "plan":
                state["plan"] = payload
            elif typ == "step_result":
                state["df_payload"] = payload["df"]
                set_df_payload(state["df_payload"])
                state["results"].append({"name": payload["name"], "stats": payload["stats"]})
                state["steps_taken"] += 1
                state["confirmed_step"] = None
                state["confirmed_params"] = {}

        # print("DEBUG roles after tools:", [getattr(m, "type", None) or getattr(m, "role", None) for m in state["messages"]])
        return state

    return _node

def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if state.get("steps_taken", 0) >= state.get("max_steps", 8):
        return "end"
    # Continue if the last assistant message contains tool calls
    return "continue" if getattr(last, "tool_calls", None) else "end"

def build_app(llm):
    g = StateGraph(AgentState)
    g.add_node("agent", make_agent_node(llm))
    g.add_node("tools", tools_exec_node())

    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
    g.add_edge("tools", "agent")

    return g.compile()
