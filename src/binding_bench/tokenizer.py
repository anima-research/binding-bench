"""The tokenizer that sizes and checks the benchmark's items.

Item generators use it to keep prompts within length and to verify single-token candidates. Every candidate pool is
also checked against the other tokenizers it is scored with (``riders.items.GRID_TOKENIZERS``,
``binding check-fluid-tokens``). Changing this name changes which items the generators emit, so it is part of every
regenerable battery's recipe.
"""

from __future__ import annotations

TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-135M"
