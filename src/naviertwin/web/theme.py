"""NavierTwin 웹 GUI 디자인 시스템 — "풍동 계측실(Wind-Tunnel Instrument)" 테마.

방향: 심연 잉크 네이비 위 **포스포 시안 단일 강렬 악센트**(오실로스코프
트레이스), 앰버 보조 계기색. 배경은 단색이 아니라 청사진 측정 격자 +
레이어드 글로우가 느리게 드리프트한다. 타이포는 Chakra Petch(각진 HUD
디스플레이체, 제목/버튼/브랜드)와 Spline Sans Mono(계측 수치) — woff2 를
base64 내장(:mod:`naviertwin.web.fonts`)해 오프라인 Electron 에서도 동일하다.
한글 본문은 시스템 폰트 폴백.

모든 색은 CSS 변수(``--nt-*``)와 이 모듈 상수의 단일 출처를 공유한다 —
vuetify 테마, 3D 뷰어 배경, matplotlib 차트가 같은 팔레트를 쓴다.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

from naviertwin.web.fonts import FONT_CSS

# ──────────────────────────────────────────────────────────────────────
# 팔레트 — 강철 주광(steel daylight) 슬레이트 네이비 + 포스포 시안(주) + 앰버
# ──────────────────────────────────────────────────────────────────────
BACKGROUND = "#1b2739"   # 슬레이트 네이비 (밝은 계기실)
SURFACE = "#243449"      # 계기함
PANEL = "#2c3e58"        # 패널
BORDER = "#43597b"       # 케이싱 라인
PRIMARY = "#12d6f2"      # 포스포 시안 (스코프 트레이스)
SECONDARY = "#4fe89b"    # 성공(유량 정상)
WARNING = "#ffbe5c"      # 앰버 계기
ERROR = "#ff6f7d"        # 경보
INFO = "#8fdcff"         # 보조 시안
TEXT = "#eef4fb"
MUTED = "#a4b7d0"

# 3D 뷰어 배경 그라데이션 (아래→위)
VIEWER_BG_BOTTOM = "#1b2739"
VIEWER_BG_TOP = "#324667"

DISPLAY_FONT = "'Chakra Petch', 'Malgun Gothic', sans-serif"
MONO_FONT = "'Spline Sans Mono', ui-monospace, monospace"


def vuetify_config() -> dict[str, Any]:
    """``VAppLayout(vuetify_config=...)`` 로 주입할 커스텀 다크 테마 설정."""
    return {
        "theme": {
            "defaultTheme": "windTunnel",
            "themes": {
                "windTunnel": {
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
                        "on-primary": "#04121a",
                        "on-success": "#04160c",
                    },
                },
            },
        },
    }


CUSTOM_CSS = FONT_CSS + f"""
:root {{
  --nt-bg: {BACKGROUND};
  --nt-surface: {SURFACE};
  --nt-panel: {PANEL};
  --nt-border: {BORDER};
  --nt-primary: {PRIMARY};
  --nt-amber: {WARNING};
  --nt-text: {TEXT};
  --nt-muted: {MUTED};
  --nt-display: {DISPLAY_FONT};
  --nt-mono: {MONO_FONT};
}}

/* ── 배경: 슬레이트 그라데이션 + 청사진 측정 격자(느린 드리프트) ───── */
.v-application, .v-application__wrap {{
  background:
    radial-gradient(140% 90% at 85% -10%, #33496c 0%, transparent 55%),
    radial-gradient(120% 80% at -10% 110%, #263752 0%, transparent 50%),
    var(--nt-bg) !important;
}}
.v-main {{
  position: relative;
}}
.v-main::before {{
  content: "";
  position: absolute; inset: 0;
  pointer-events: none;
  background-image:
    linear-gradient(rgba(18, 214, 242, 0.06) 1px, transparent 1px),
    linear-gradient(90deg, rgba(18, 214, 242, 0.06) 1px, transparent 1px),
    linear-gradient(rgba(18, 214, 242, 0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(18, 214, 242, 0.025) 1px, transparent 1px);
  background-size: 140px 140px, 140px 140px, 28px 28px, 28px 28px;
  animation: ntGridDrift 60s linear infinite;
}}
@keyframes ntGridDrift {{
  from {{ background-position: 0 0, 0 0, 0 0, 0 0; }}
  to   {{ background-position: 140px 140px, 140px 140px, 28px 28px, 28px 28px; }}
}}

/* ── 타이포 ────────────────────────────────────────────────────────── */
.v-application {{
  color: var(--nt-text);
}}
.v-toolbar-title, .v-expansion-panel-title, .v-btn, .v-card-title,
.v-chip, .v-tab, .v-label {{
  font-family: var(--nt-display) !important;
}}
.v-btn {{ letter-spacing: 0.06em; }}
.nt-mono, .v-slider-thumb__label, .v-progress-linear__content {{
  font-family: var(--nt-mono) !important;
  font-size: 0.8rem;
}}

/* ⓘ/⚠ 설명 아이콘 — 평소엔 물러나 있고 호버하면 살아난다.
   설명이 패널을 채우는 대신 여기 접혀 있으므로, "누를 수 있다"는 신호가
   있어야 사용자가 존재를 알아챈다. */
.nt-tip {{
  cursor: help;
  opacity: 0.55;
  transition: opacity 0.15s ease, transform 0.15s ease;
  flex: 0 0 auto;
}}
.nt-tip:hover {{
  opacity: 1;
  transform: scale(1.25);
}}

/* 브랜드 — 시안 발광 + 스코프 스윕 */
.nt-brand {{
  font-family: var(--nt-display) !important;
  font-weight: 700; letter-spacing: 0.14em;
  color: var(--nt-primary);
  text-shadow: 0 0 14px rgba(18, 214, 242, 0.5);
  position: relative;
}}
.nt-brand::after {{
  content: ""; position: absolute; left: 0; right: 0; bottom: -3px; height: 2px;
  background: linear-gradient(90deg, transparent, var(--nt-primary), transparent);
  animation: ntSweep 3.2s ease-in-out infinite;
  opacity: 0.85;
}}
@keyframes ntSweep {{
  0%, 100% {{ transform: translateX(-30%); opacity: 0.15; }}
  50% {{ transform: translateX(30%); opacity: 0.9; }}
}}

/* ── 크롬(툴바/드로어/푸터) ────────────────────────────────────────── */
.v-toolbar.v-app-bar {{
  background: linear-gradient(180deg, #2c405d 0%, var(--nt-surface) 100%) !important;
  border-bottom: 1px solid var(--nt-border);
  box-shadow: 0 1px 0 rgba(18, 214, 242, 0.18) !important;
}}
.nt-drawer, .v-navigation-drawer {{
  background: var(--nt-surface) !important;
  border-right: 1px solid var(--nt-border) !important;
}}
.v-footer {{
  background: var(--nt-surface) !important;
  border-top: 1px solid var(--nt-border);
  font-family: var(--nt-mono);
  font-size: 0.72rem !important;
}}

/* ── 드로어 로딩 캐스케이드 (첫 페인트 시 위→아래 순차 등장) ───────── */
.nt-drawer .v-expansion-panel, .nt-drawer .v-sheet {{
  animation: ntRise 0.5s cubic-bezier(0.2, 0.9, 0.3, 1) both;
}}
.nt-drawer .v-sheet {{ animation-delay: 0.05s; }}
.nt-drawer .v-expansion-panel:nth-child(1) {{ animation-delay: 0.10s; }}
.nt-drawer .v-expansion-panel:nth-child(2) {{ animation-delay: 0.16s; }}
.nt-drawer .v-expansion-panel:nth-child(3) {{ animation-delay: 0.22s; }}
.nt-drawer .v-expansion-panel:nth-child(4) {{ animation-delay: 0.28s; }}
.nt-drawer .v-expansion-panel:nth-child(5) {{ animation-delay: 0.34s; }}
.nt-drawer .v-expansion-panel:nth-child(6) {{ animation-delay: 0.40s; }}
.nt-drawer .v-expansion-panel:nth-child(7) {{ animation-delay: 0.46s; }}
.nt-drawer .v-expansion-panel:nth-child(8) {{ animation-delay: 0.52s; }}
@keyframes ntRise {{
  from {{ opacity: 0; transform: translateY(14px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}

/* ── 확장 패널: 계기 모듈 카드 ─────────────────────────────────────── */
.nt-drawer .v-expansion-panel {{
  background: var(--nt-panel) !important;
  border: 1px solid var(--nt-border);
  border-radius: 10px !important;
  margin-bottom: 6px;
  transition: border-color 0.25s ease, box-shadow 0.25s ease;
}}
.nt-drawer .v-expansion-panel--active {{
  border-color: rgba(18, 214, 242, 0.5);
  box-shadow: inset 3px 0 0 var(--nt-primary), 0 4px 18px rgba(0, 0, 0, 0.3);
}}
.v-expansion-panel-text__wrapper {{ animation: ntFade 0.3s ease; }}
@keyframes ntFade {{
  from {{ opacity: 0; transform: translateY(-6px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
.nt-drawer .v-card {{
  background: rgba(255, 255, 255, 0.04) !important;
  border: 1px solid var(--nt-border);
  transition: border-color 0.2s ease;
}}
.nt-drawer .v-card:hover {{ border-color: rgba(18, 214, 242, 0.4); }}

/* ── 버튼: 주 액션 발광 ────────────────────────────────────────────── */
.v-btn--variant-elevated.bg-primary, .v-btn.bg-primary {{
  box-shadow: 0 0 16px rgba(18, 214, 242, 0.32) !important;
}}
.v-btn.bg-primary:hover {{ box-shadow: 0 0 26px rgba(18, 214, 242, 0.55) !important; }}

/* ── 파이프라인 칩 ─────────────────────────────────────────────────── */
.v-chip {{ font-size: 0.66rem !important; letter-spacing: 0.04em; }}
.nt-chip-active {{ animation: ntPulse 1.3s ease-in-out infinite; }}
@keyframes ntPulse {{
  0%, 100% {{ opacity: 1; }}
  50% {{ opacity: 0.4; }}
}}

/* ── 진행바/스크롤바/다이얼로그 ────────────────────────────────────── */
.v-progress-linear {{ border-radius: 2px; }}
.v-progress-linear__determinate {{
  box-shadow: 0 0 10px rgba(18, 214, 242, 0.65);
}}
::-webkit-scrollbar {{ width: 9px; height: 9px; }}
::-webkit-scrollbar-track {{ background: var(--nt-bg); }}
::-webkit-scrollbar-thumb {{
  background: #3a4d6b; border-radius: 5px; border: 2px solid var(--nt-bg);
}}
::-webkit-scrollbar-thumb:hover {{ background: #4a6088; }}
.v-dialog .v-card {{
  background: var(--nt-panel) !important;
  border: 1px solid rgba(18, 214, 242, 0.3);
  box-shadow: 0 0 40px rgba(0, 0, 0, 0.45), 0 0 22px rgba(18, 214, 242, 0.15) !important;
}}
.v-snackbar__wrapper {{ font-family: var(--nt-display); }}
"""


@contextlib.contextmanager
def mpl_dark() -> Iterator[None]:
    """matplotlib 다크 스타일 컨텍스트 — 계기 팔레트로 통일."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    overrides = {
        "figure.facecolor": SURFACE,
        "axes.facecolor": PANEL,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BORDER,
        "grid.color": "#1a2740",
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlecolor": TEXT,
        "font.family": "monospace",
    }
    with plt.style.context("dark_background"), mpl.rc_context(overrides):
        yield


__all__ = [
    "BACKGROUND",
    "BORDER",
    "CUSTOM_CSS",
    "DISPLAY_FONT",
    "ERROR",
    "INFO",
    "MONO_FONT",
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
