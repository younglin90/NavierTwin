"""Fluent .cas 파서 확장 — section index lookup + zone-name 추출.

기존 fluent_reader 보완: ASCII .cas 의 section header (id) 를 빠르게 enumerate.

Examples:
    >>> from naviertwin.core.cfd_reader import parse_section_ids
    >>> txt = '(0 "header") (10 (0 1 2 0 0))'
    >>> ids = parse_section_ids(txt)
    >>> 0 in ids and 10 in ids
    True
"""

from __future__ import annotations

import re
from pathlib import Path

_SECTION_RE = re.compile(r"\(\s*(\d+)\b")


def parse_section_ids(text: str) -> list[int]:
    """ASCII .cas section ID list."""
    return list(map(lambda m: int(m.group(1)), _SECTION_RE.finditer(text)))


def section_count(path: str | Path, section_id: int) -> int:
    text = Path(path).read_text(errors="ignore")
    return parse_section_ids(text).count(section_id)


def list_zone_names(path: str | Path) -> list[str]:
    """Section 45 (zone-spec) 의 zone name 추출 (best effort)."""
    text = Path(path).read_text(errors="ignore")
    # zone-spec: (45 (id zone_name zone_type) ...)
    zone_re = re.compile(r"\(\s*45\s+\(\s*\d+\s+([\w\-]+)")
    return zone_re.findall(text)


__all__ = ["list_zone_names", "parse_section_ids", "section_count"]
