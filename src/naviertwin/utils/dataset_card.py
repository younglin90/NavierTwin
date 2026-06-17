"""Dataset card / datasheet generator (Markdown).

Examples:
    >>> from naviertwin.utils.dataset_card import dataset_card_md
    >>> md = dataset_card_md(name='cavity', n_samples=100, source='LES')
    >>> '# Dataset' in md
    True
"""

from __future__ import annotations

from pathlib import Path


def dataset_card_md(
    *, name: str, n_samples: int, source: str = "",
    license: str = "Apache-2.0", description: str = "",
    schema: dict[str, str] | None = None,
) -> str:
    parts = [
        "# Dataset Card",
        f"- **Name**: {name}",
        f"- **Samples**: {n_samples}",
        f"- **Source**: {source}",
        f"- **License**: {license}",
    ]
    if description:
        parts += ["", "## Description", description]
    if schema:
        parts += ["", "## Schema"]
        items = list(schema.items())
        idx = 0
        while idx < len(items):
            k, v = items[idx]
            parts.append(f"- `{k}`: {v}")
            idx += 1
    return "\n".join(parts) + "\n"


def write_card(path: str | Path, **kwargs) -> None:
    Path(path).write_text(dataset_card_md(**kwargs))


__all__ = ["dataset_card_md", "write_card"]
