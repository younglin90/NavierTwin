"""CLI entrypoint — Click-like dispatcher (no Click dep).

Examples:
    >>> from naviertwin.utils.cli_entry import build_parser
    >>> p = build_parser()
    >>> ns = p.parse_args(["train", "--epochs", "5"])
    >>> ns.cmd
    'train'
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="naviertwin")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_train = sub.add_parser("train")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--config", type=str, default="config.toml")
    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--input", type=str, required=True)
    p_pred.add_argument("--output", type=str, default="out.npz")
    p_info = sub.add_parser("info")
    _ = p_info  # suppress unused
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"command: {args.cmd}")
    return 0


__all__ = ["build_parser", "main"]
