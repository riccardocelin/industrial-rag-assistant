"""Simple Tkinter GUI for querying the RAG API.

This GUI mirrors the API usage in test/test_api.py by sending a POST request
with {"question": ..., "force_no_context": ...} to /ask.
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox

import requests

API_URL = "http://localhost:8000/ask"


class RAGGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Industrial RAG Assistant")
        self.root.geometry("700x700")
        self.root.minsize(520, 520)

        self._build_ui()

    def _build_ui(self) -> None:
        container = tk.Frame(self.root, padx=16, pady=16)
        container.pack(fill="both", expand=True)

        title = tk.Label(
            container,
            text="Ask the RAG System",
            font=("Arial", 16, "bold"),
            anchor="w",
        )
        title.pack(fill="x", pady=(0, 12))

        tk.Label(container, text="Your question:", anchor="w").pack(fill="x")
        self.question_box = tk.Text(container, height=8, wrap="word")
        self.question_box.pack(fill="x", pady=(6, 14))

        self.ask_button = tk.Button(
            container,
            text="Generate Answer",
            command=self.on_ask_clicked,
            height=2,
        )
        self.ask_button.pack(fill="x", pady=(0, 14))

        tk.Label(container, text="System answer:", anchor="w").pack(fill="x")
        self.answer_box = tk.Text(container, height=18, wrap="word", state="disabled")
        self.answer_box.pack(fill="both", expand=True, pady=(6, 0))

    def on_ask_clicked(self) -> None:
        question = self.question_box.get("1.0", "end").strip()
        if not question:
            messagebox.showwarning("Missing question", "Please type a question first.")
            return

        self._set_loading_state(True)
        self._write_answer("Generating answer...")

        thread = threading.Thread(target=self._fetch_answer, args=(question,), daemon=True)
        thread.start()

    def _fetch_answer(self, question: str) -> None:
        try:
            response = requests.post(
                API_URL,
                json={"question": question, "force_no_context": False},
                timeout=120,
            )
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No answer field in response.")
                self.root.after(0, lambda: self._write_answer(answer))
            else:
                self.root.after(
                    0,
                    lambda: self._write_answer(
                        f"Error {response.status_code}: {response.text}"
                    ),
                )
        except requests.RequestException as exc:
            self.root.after(0, lambda: self._write_answer(f"Request failed: {exc}"))
        finally:
            self.root.after(0, lambda: self._set_loading_state(False))

    def _set_loading_state(self, is_loading: bool) -> None:
        state = "disabled" if is_loading else "normal"
        self.ask_button.config(state=state)

    def _write_answer(self, text: str) -> None:
        self.answer_box.config(state="normal")
        self.answer_box.delete("1.0", "end")
        self.answer_box.insert("1.0", text)
        self.answer_box.config(state="disabled")


def main() -> None:
    root = tk.Tk()
    RAGGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
