"""ChatML as a plain string: the frame of the riders chat register.

The chat register's pinned prompts are rendered with this function, so its output is part of every chat-register
item's definition and never changes. kotodama's own renderer is checked against it by a parity test in kotodama.
"""

from __future__ import annotations


def render_chatml(messages: list[dict[str, str]]) -> str:
    """Each message as ``<|im_start|>{role}\\n{content}<|im_end|>\\n``, then an open assistant turn."""
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages]
    return "".join(parts) + "<|im_start|>assistant\n"
