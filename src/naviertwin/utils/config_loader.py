"""YAML/TOML/JSON 설정 로더 + Hydra-style dotted override.

사용:
    cfg = load_config("experiment.yaml", overrides=["lr=1e-4", "model.hidden=64"])

지원 포맷: .yaml, .yml, .toml, .json (자동 감지).

Examples:
    >>> from naviertwin.utils.config_loader import merge_overrides
    >>> cfg = {"lr": 1e-3, "model": {"hidden": 32}}
    >>> merged = merge_overrides(cfg, ["lr=1e-4", "model.hidden=128", "new_key=42"])
    >>> merged["lr"]
    0.0001
    >>> merged["model"]["hidden"]
    128
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _parse_scalar(s: str) -> Any:
    """문자열 → int/float/bool/str/None 자동 추론."""
    s = s.strip()
    low = s.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none", ""):
        return None
    try:
        if "." in s or "e" in low:
            return float(s)
        return int(s)
    except ValueError:
        pass
    # quoted string
    if len(s) >= 2 and s[0] in {'"', "'"} and s[-1] == s[0]:
        return s[1:-1]
    return s


def set_dotted(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    """cfg["a"]["b"]["c"] = value using 'a.b.c' notation."""
    keys = dotted.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def get_dotted(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def merge_overrides(
    cfg: dict[str, Any], overrides: list[str]
) -> dict[str, Any]:
    """'key=value' 또는 'nested.key=value' 리스트를 cfg 에 적용."""
    out: dict[str, Any] = json.loads(json.dumps(cfg))  # deep copy (JSON-safe)
    for item in overrides or []:
        if "=" not in item:
            logger.warning("override 무시 (no '='): %s", item)
            continue
        key, _, val = item.partition("=")
        set_dotted(out, key.strip(), _parse_scalar(val))
    return out


def load_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """YAML/TOML/JSON 파일 → dict, 그리고 overrides 적용.

    path=None 이면 빈 dict 로 시작.
    """
    if path is None:
        cfg: dict[str, Any] = {}
    else:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"설정 파일 없음: {p}")
        suffix = p.suffix.lower()
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as exc:
                raise RuntimeError(
                    "yaml 필요: pip install pyyaml"
                ) from exc
            with p.open("r", encoding="utf-8") as f:
                cfg = dict(yaml.safe_load(f) or {})
        elif suffix == ".toml":
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib  # type: ignore[import-not-found]
                except ImportError as exc:
                    raise RuntimeError("tomli 필요 (Python <3.11)") from exc
            with p.open("rb") as f:
                cfg = dict(tomllib.load(f))
        elif suffix == ".json":
            with p.open("r", encoding="utf-8") as f:
                cfg = dict(json.load(f))
        else:
            raise ValueError(f"지원하지 않는 포맷: {suffix}")

    if overrides:
        cfg = merge_overrides(cfg, overrides)
    logger.info("Config 로드: path=%s, override=%d", path, len(overrides or []))
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> Path:
    """YAML/TOML/JSON 저장."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("yaml 필요") from exc
        with p.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    elif suffix == ".toml":
        # 간이 TOML 작성 (최상위 scalar/section 만)
        with p.open("w", encoding="utf-8") as f:
            _write_toml(cfg, f)
    elif suffix == ".json":
        with p.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"지원하지 않는 포맷: {suffix}")
    return p


def _write_toml(cfg: dict[str, Any], f: Any, prefix: str = "") -> None:
    """단순 TOML 직접 출력 (중첩 dict 는 섹션)."""
    scalars = {k: v for k, v in cfg.items() if not isinstance(v, dict)}
    sections = {k: v for k, v in cfg.items() if isinstance(v, dict)}
    for k, v in scalars.items():
        if isinstance(v, str):
            f.write(f'{k} = "{v}"\n')
        elif isinstance(v, bool):
            f.write(f"{k} = {str(v).lower()}\n")
        elif v is None:
            pass
        else:
            f.write(f"{k} = {v}\n")
    for name, sub in sections.items():
        full = f"{prefix}{name}" if prefix else name
        f.write(f"\n[{full}]\n")
        _write_toml(sub, f, prefix=full + ".")


__all__ = [
    "load_config",
    "save_config",
    "merge_overrides",
    "set_dotted",
    "get_dotted",
]
