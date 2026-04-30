"""Sign NavierTwin release metadata for update-check handoff."""

from __future__ import annotations

import argparse
import json
import os
import sys
from base64 import b64decode, b64encode
from binascii import Error as Base64Error
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from naviertwin.utils.updater import (  # noqa: E402
    SIGNATURE_ALGORITHM,
    canonical_release_metadata_payload,
    load_release_metadata,
)


def _load_private_key(private_key_base64: str) -> Ed25519PrivateKey:
    """Load an Ed25519 private key from a base64-encoded 32-byte raw seed."""
    try:
        private_key_bytes = b64decode(private_key_base64.strip(), validate=True)
    except Base64Error as exc:
        raise ValueError("private key must be base64") from exc
    if len(private_key_bytes) != 32:
        raise ValueError("Ed25519 private key must be 32 raw bytes")
    return Ed25519PrivateKey.from_private_bytes(private_key_bytes)


def public_key_base64(private_key_base64: str) -> str:
    """Return the matching base64-encoded raw public key."""
    public_key = _load_private_key(private_key_base64).public_key()
    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return b64encode(public_bytes).decode("ascii")


def sign_release_metadata_payload(
    payload: dict[str, Any],
    *,
    key_id: str,
    private_key_base64: str,
) -> dict[str, Any]:
    """Return release metadata with an Ed25519 signature field attached."""
    if not key_id.strip():
        raise ValueError("key_id is required")
    signed = dict(payload)
    signed.pop("signature", None)
    signature = _load_private_key(private_key_base64).sign(
        canonical_release_metadata_payload(signed)
    )
    signed["signature"] = {
        "algorithm": SIGNATURE_ALGORITHM,
        "key_id": key_id.strip(),
        "value": b64encode(signature).decode("ascii"),
    }
    return signed


def _read_private_key(args: argparse.Namespace) -> str:
    if args.private_key_file:
        return Path(args.private_key_file).read_text(encoding="utf-8").strip()
    value = os.environ.get(args.private_key_env, "").strip()
    if value:
        return value
    raise ValueError(
        f"set {args.private_key_env} or pass --private-key-file with a base64 Ed25519 key"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sign NavierTwin release metadata JSON for update-check.",
    )
    parser.add_argument("--input", required=True, help="Unsigned release metadata JSON")
    parser.add_argument("--output", required=True, help="Signed metadata JSON output path")
    parser.add_argument("--key-id", required=True, help="Trusted updater key id")
    parser.add_argument(
        "--private-key-env",
        default="NAVIER_TWIN_RELEASE_PRIVATE_KEY_B64",
        help="Environment variable containing a base64 Ed25519 private key",
    )
    parser.add_argument(
        "--private-key-file",
        default=None,
        help="File containing a base64 Ed25519 private key",
    )
    parser.add_argument(
        "--print-public-key",
        action="store_true",
        default=False,
        help="Print the matching public key after signing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        private_key = _read_private_key(args)
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("release metadata must be a JSON object")
        signed = sign_release_metadata_payload(
            payload,
            key_id=args.key_id,
            private_key_base64=private_key,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(signed, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        public_key = public_key_base64(private_key)
        load_release_metadata(output, trusted_public_keys={args.key_id: public_key})
    except Exception as exc:  # noqa: BLE001
        print(f"sign-release-metadata error: {exc}", file=sys.stderr)
        return 2

    if args.print_public_key:
        print(public_key)
    else:
        print(str(output))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
