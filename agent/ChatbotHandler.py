import os
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.agent_graph import build_app
from pipeline.utils_cool import df_to_payload, parse_user_choice

from .runtime_ctx import get_df_summary
from dotenv import load_dotenv
load_dotenv()

class ChatbotHandler:
    def __init__(self):
        self.ctx: Dict[str, Any] = {
            "graph_app": None,      # LangGraph app
            "state": {              # mirrors your prior STATE
                "df_payload": None,
                "results": [],
                "steps_taken": 0,
                "confirmed_step": None,
                "confirmed_params": {},
                "last_task": None,
                "plan": None,
                "messages": [],
                "max_steps": 8,
            },
        }
        # keep chat UI history in the component; we only need to return a reply string to it
        # but Gradio Chatbot expects (history, ""), so we'll append our reply to history.
        self._boot_text: Optional[str] = None  # first reply after upload

        # LLM for the graph
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0,
            api_key=os.getenv("GOOGLE_API_KEY"),
        )

    def _format_summary(self, s: Dict[str, Any]) -> str:
        cols = s.get("columns") or []
        dtypes = s.get("dtypes") or {}
        shape = s.get("shape") or (None, None)
        label_guess = s.get("label_guess") or "None"
        task_guess = s.get("task_guess") or "Unknown"
        issues = s.get("issues") or []

        # keep it concise but helpful
        dt_pairs = [f"{k}: {v}" for k, v in list(dtypes.items())[:8]]
        if len(dtypes) > 8:
            dt_pairs.append("…")

        lines = [
            "### Dataset summary",
            f"- Shape: {shape[0]} rows × {shape[1]} columns",
            f"- Columns: {', '.join(map(str, cols[:10]))}{'…' if len(cols) > 10 else ''}",
            f"- Dtypes: {', '.join(dt_pairs)}",
            f"- Label guess: {label_guess}",
            f"- Task guess: {task_guess}",
        ]
        if issues:
            lines.append(f"- Potential issues: {('; '.join(issues[:3]))}{'…' if len(issues) > 3 else ''}")
        return "\n".join(lines)

    # ---------------- Boot on upload: run inspect + SOTA + plan ----------------
    def update_context(self, file_path: Optional[str], data_type: Optional[str], df: Optional["pd.DataFrame"]):
        if df is None:
            return ""

        # (Re)build graph + seed state (unchanged)
        self.ctx["graph_app"] = build_app(self.llm)
        df_payload = df_to_payload(df)
        st = self.ctx["state"]
        st.update({
            "df_payload": df_payload,
            "results": [],
            "steps_taken": 0,
            "confirmed_step": None,
            "confirmed_params": {},
            "last_task": None,
            "plan": None,
            "messages": [HumanMessage(content="A new dataset was uploaded. Start the workflow.")],
            "max_steps": 8,
        })

        final = self.ctx["graph_app"].invoke(st)
        for k in ["df_payload","results","steps_taken","confirmed_step","confirmed_params","last_task","plan","messages"]:
            st[k] = final.get(k, st.get(k))

        # Build the boot text from the stored summary (authoritative + consistent)
        s = get_df_summary() or {}
        summary_text = self._format_summary(s)

        # Decide whether to ask for confirmation
        task_guess = (s.get("task_guess") or "").lower()
        label_guess = s.get("label_guess")
        needs_task = task_guess not in {"classification", "regression", "unsupervised"}
        needs_label = (task_guess in {"classification", "regression"}) and (not label_guess)

        if needs_task or needs_label:
            ask = "\n\nPlease confirm the task" + (" and label column" if needs_label else "") + \
                  ". For example: `task=classification label=noisy_letter_grade`."
        else:
            ask = f"\n\nIf that looks right, say `confirm task={task_guess}" + \
                  (f" label={label_guess}`" if label_guess else "`") + \
                  " and I’ll fetch SOTA and propose a plan."

        self._boot_text = summary_text + ask
        return self._boot_text

    # ---------------- One chat turn → graph turn ----------------
    def respond(self, message: str, history: List):
        if history is None:
            history = []
        msg = (message or "").strip()
        if not msg:
            return history, ""

        # If we have a prepared boot reply (from upload) and the chat is empty,
        # show it before processing the user's first message.
        if self._boot_text and len(history) == 0:
            history.append(("[system]", self._boot_text))
            self._boot_text = None

        # Require a booted graph
        if self.ctx.get("graph_app") is None:
            history.append((msg, "Please upload a dataset first."))
            return history, ""

        st = self.ctx["state"]

        # Allow quick “run X a=b” parsing before we call the graph (same as your old handle_chat)
        step, params = parse_user_choice(msg)
        if step:
            st["confirmed_step"] = step
            st["confirmed_params"] = {**(st.get("confirmed_params") or {}), **params}

        # Build this turn’s input state
        messages = (st.get("messages") or []) + [HumanMessage(content=msg)]
        turn_state = {
            "messages": messages,
            "df_payload": st.get("df_payload"),
            "results": st.get("results", []),
            "steps_taken": st.get("steps_taken", 0),
            "max_steps": max(8, st.get("steps_taken", 0) + 4),
            "confirmed_step": st.get("confirmed_step"),
            "confirmed_params": st.get("confirmed_params", {}),
            "last_task": st.get("last_task"),
            "plan": st.get("plan"),
        }

        # Invoke graph for this turn
        final = self.ctx["graph_app"].invoke(turn_state)

        # Persist state back
        for k in ["df_payload","results","steps_taken","confirmed_step","confirmed_params","last_task","plan","messages"]:
            st[k] = final.get(k, turn_state.get(k, st.get(k)))

        # Extract assistant text
        reply = self._extract_ai_text(final.get("messages", [])) or "Done."
        history.append((msg, reply))
        return history, ""

    # ---------------- helper: extract last AI string ----------------
    def _extract_ai_text(self, messages: List[Any]) -> str:
        def coerce_text(content: Any) -> str:
            if content is None: return ""
            if isinstance(content, str): return content
            if isinstance(content, list):
                parts = []
                for c in content:
                    if isinstance(c, dict):
                        parts.append(str(c.get("text") or c.get("content") or c.get("data") or ""))
                    else:
                        parts.append(str(c))
                return " ".join(p for p in parts if p)
            return str(content)

        for m in reversed(messages or []):
            role = getattr(m, "type", None) or getattr(m, "role", None)
            if role in ("ai", "assistant", "aimessage"):
                return coerce_text(getattr(m, "content", None))
            if isinstance(m, dict):
                r = (m.get("role") or m.get("type") or "").lower()
                if r in ("assistant", "ai", "aimessage"):
                    return coerce_text(m.get("content"))
        return ""
