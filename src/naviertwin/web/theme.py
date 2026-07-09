"""NavierTwin 웹 GUI 디자인 시스템 — 다크 엔지니어링 테마 단일 출처.

vuetify3 커스텀 다크 테마, 커스텀 CSS(폰트/애니메이션/스크롤바), matplotlib 다크
스타일, PyVista 뷰어 배경을 한 곳에서 정의한다. 팔레트는 기존 차트 색상
(`#2f81f7`/`#3fb950`/`#f0883e`)과 일치시켜 3D 뷰어·차트·UI 전반의 색을 통일한다.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

# ──────────────────────────────────────────────────────────────────────
# 팔레트 (GitHub dark 계열 — 차트 색과 통일)
# ──────────────────────────────────────────────────────────────────────
BACKGROUND = "#0d1117"
SURFACE = "#161b22"
PANEL = "#1c2128"
BORDER = "#30363d"
PRIMARY = "#2f81f7"
SECONDARY = "#3fb950"  # success/green
WARNING = "#f0883e"
ERROR = "#f85149"
INFO = "#22d3ee"
TEXT = "#e6edf3"
MUTED = "#8b949e"

# 3D 뷰어 배경 그라데이션 (아래→위)
VIEWER_BG_BOTTOM = "#0d1117"
VIEWER_BG_TOP = "#1b2735"


def vuetify_config() -> dict[str, Any]:
    """``VAppLayout(vuetify_config=...)`` 로 주입할 커스텀 다크 테마 설정."""
    return {
        "theme": {
            "defaultTheme": "navierDark",
            "themes": {
                "navierDark": {
                    "dark": True,
                    "colors": {
                        "background": BACKGROUND,
                        "surface": SURFACE,
                        "primary": PRIMARY,
                        "secondary": SECONDARY,
                        "success": SECONDARY,
                        "warning": WARNING,
                        "error": ERROR,
                        "info": INFO,
                        "on-background": TEXT,
                        "on-surface": TEXT,
                    },
                },
            },
        },
    }


# 커스텀 CSS — 폰트, 스크롤바, 카드 hover, 브랜드 타이틀, 패널 페이드, busy pulse.
CUSTOM_CSS = f"""
:root {{
  --nt-bg: {BACKGROUND};
  --nt-surface: {SURFACE};
  --nt-panel: {PANEL};
  --nt-border: {BORDER};
  --nt-primary: {PRIMARY};
  --nt-muted: {MUTED};
}}

.v-application, .v-application__wrap {{
  background: var(--nt-bg) !important;
}}

/* 다크 스크롤바 */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: var(--nt-bg); }}
::-webkit-scrollbar-thumb {{
  background: #2d333b; border-radius: 6px; border: 2px solid var(--nt-bg);
}}
::-webkit-scrollbar-thumb:hover {{ background: #3d444d; }}

/* 브랜드 타이틀 그라데이션 */
.nt-brand {{
  font-weight: 700; letter-spacing: 0.5px;
  background: linear-gradient(90deg, {PRIMARY} 0%, {INFO} 60%, {SECONDARY} 100%);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
}}

/* 드로어 패널 카드 hover — 살짝 떠오르는 느낌 */
.nt-drawer .v-card {{
  transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.12s ease;
  border: 1px solid var(--nt-border);
}}
.nt-drawer .v-card:hover {{
  border-color: {PRIMARY}66;
  box-shadow: 0 2px 12px #0008;
}}

/* 확장 패널 본문 등장 애니메이션 */
.v-expansion-panel-text__wrapper {{ animation: ntFade 0.28s ease; }}
@keyframes ntFade {{
  from {{ opacity: 0; transform: translateY(-6px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}

/* 작업 중 파이프라인 칩 pulse */
.nt-chip-active {{ animation: ntPulse 1.3s ease-in-out infinite; }}
@keyframes ntPulse {{
  0%, 100% {{ opacity: 1; }}
  50%      {{ opacity: 0.45; }}
}}

/* 숫자/지표 monospace */
.nt-mono {{
  font-family: "JetBrains Mono", "Fira Code", ui-monospace, SFMono-Regular, monospace;
  font-size: 0.82rem;
}}

/* 진행바 라운드 */
.v-progress-linear {{ border-radius: 3px; }}

/* 드로어/툴바 배경 톤 */
.nt-drawer, .v-navigation-drawer {{ background: var(--nt-surface) !important; }}
.v-toolbar.v-app-bar {{
  background: linear-gradient(180deg, #1c2431 0%, {SURFACE} 100%) !important;
  border-bottom: 1px solid var(--nt-border);
}}
"""


@contextlib.contextmanager
def mpl_dark() -> Iterator[None]:
    """matplotlib 다크 스타일 컨텍스트 (facecolor 를 surface 색으로 통일)."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    overrides = {
        "figure.facecolor": SURFACE,
        "axes.facecolor": PANEL,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BORDER,
        "grid.color": "#2d333b",
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlecolor": TEXT,
    }
    with plt.style.context("dark_background"), mpl.rc_context(overrides):
        yield


__all__ = [
    "BACKGROUND",
    "BORDER",
    "CUSTOM_CSS",
    "ERROR",
    "INFO",
    "MUTED",
    "PANEL",
    "PRIMARY",
    "SECONDARY",
    "SURFACE",
    "TEXT",
    "VIEWER_BG_BOTTOM",
    "VIEWER_BG_TOP",
    "WARNING",
    "mpl_dark",
    "vuetify_config",
]
