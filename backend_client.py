# backend_client.py
from __future__ import annotations
import requests
from typing import Optional, List, Tuple

class LangGraphClient:
    """
    Minimal client to talk to your LangGraph app running on http://localhost:8010.
    Adjust ENDPOINTS to match your server routes/payloads.
    """

    def __init__(self, base_url: str = "http://localhost:8010"):
        self.base_url = base_url.rstrip("/")
        # Example endpoints — change if your server differs
        self.endpoints = {
            "chat": f"{self.base_url}/chat"   # expects JSON {session_id, message, history?}
        }

    def send_message(
        self,
        message: str,
        session_id: Optional[str] = "default",
        history: Optional[List[Tuple[str, str]]] = None,
        timeout: int = 30
    ) -> str:
        """
        Sends a message to the backend and returns the assistant reply text.
        """
        payload = {
            "session_id": session_id,
            "message": message,
            "history": history or []  # [[user, assistant], ...] if your server uses it
        }
        try:
            r = requests.post(self.endpoints["chat"], json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            # Expecting {"reply": "..."}; adjust if your API returns a different shape
            reply = data.get("reply") or data.get("message") or data.get("text") or ""
            return str(reply)
        except requests.RequestException as e:
            return f"[Backend error] {e}"
