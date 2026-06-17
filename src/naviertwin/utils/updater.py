"""Offline update metadata contract used by release channels."""

from __future__ import annotations

import json
import platform
import re
import subprocess
from base64 import b64decode
from binascii import Error as Base64Error
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from naviertwin import __version__
from naviertwin.utils.hashing import hash_file

SUPPORTED_CHANNELS = {"stable", "beta", "nightly"}
WINDOWS_INSTALLER_NAME = "NavierTwinSetup.exe"
SIGNATURE_ALGORITHM = "ed25519"
DEFAULT_RELEASE_PUBLIC_KEYS = {
    "naviertwin-release-2026q2": "GUpIKQZAM20tuTWBR3WF7vfE43pZOeVuF1D3und2JRY=",
}


@dataclass(frozen=True)
class ReleaseMetadata:
    """Validated release metadata loaded from a GitHub Releases mirror."""

    version: str
    channel: str
    url: str
    sha256: str
    notes: str = ""
    signature_key_id: str = ""
    installer_signing: dict[str, object] | None = None


@dataclass(frozen=True)
class UpdateCheckResult:
    """Deterministic update-check result used by CLI and GUI callers."""

    current_version: str
    latest_version: str
    channel: str
    update_available: bool
    url: str
    sha256: str
    installer_signing: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class ArtifactVerificationResult:
    """SHA-256 verification result of a downloaded release artifact."""

    path: str
    expected_sha256: str
    actual_sha256: str
    size_bytes: int
    verified: bool
    authenticode: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _version_key(version: str) -> tuple[int, ...]:
    """Convert a numeric release version into a comparable tuple."""
    parts = version.strip().split(".")
    if not parts:
        raise ValueError(f"numeric dotted version required: {version!r}")
    values = []
    part_idx = 0
    while part_idx < len(parts):
        part = parts[part_idx]
        if not part.isdigit():
            raise ValueError(f"numeric dotted version required: {version!r}")
        values.append(int(part))
        part_idx += 1
    return tuple(values)


def is_newer_version(candidate: str, current: str = __version__) -> bool:
    """Return whether ``candidate`` is newer than ``current``."""
    left = _version_key(candidate)
    right = _version_key(current)
    width = max(len(left), len(right))
    return left + (0,) * (width - len(left)) > right + (0,) * (width - len(right))


def _validate_release_url(url: str, version: str) -> None:
    """Validate that release metadata points at the expected versioned artifact."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("release metadata url must be an https URL")

    expected_tag = f"/releases/download/v{version}/"
    if expected_tag not in parsed.path:
        raise ValueError(f"release metadata url tag must match version v{version}")

    filename = PurePosixPath(parsed.path).name
    if filename != WINDOWS_INSTALLER_NAME:
        raise ValueError(f"release metadata url must point to {WINDOWS_INSTALLER_NAME}")


def _normalize_sha256(value: str) -> str:
    """Validate and normalize a SHA-256 hex digest."""
    digest = value.strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("expected a 64-character sha256 hex digest")
    return digest


def _normalize_thumbprint(value: str) -> str:
    """Normalize a SHA-1/SHA-256 certificate thumbprint."""
    thumbprint = re.sub(r"[^0-9a-fA-F]", "", value).upper()
    if re.fullmatch(r"[0-9A-F]{40}|[0-9A-F]{64}", thumbprint) is None:
        raise ValueError("certificate_thumbprint must be 40 or 64 hex characters")
    return thumbprint


def _validate_installer_signing(value: object) -> dict[str, object]:
    """Validate optional release metadata with Windows Authenticode identity."""
    if not isinstance(value, dict):
        raise ValueError("installer_signing must be a JSON object")
    publisher = str(value.get("publisher", "")).strip()
    thumbprint = _normalize_thumbprint(str(value.get("certificate_thumbprint", "")))
    if not publisher:
        raise ValueError("installer_signing.publisher is required")
    return {
        "publisher": publisher,
        "certificate_thumbprint": thumbprint,
        "authenticode_required": bool(value.get("authenticode_required", True)),
    }


def canonical_release_metadata_payload(data: Mapping[str, Any]) -> bytes:
    """Return the canonical byte payload covered by release metadata signatures."""
    payload = {
        "channel": str(data.get("channel", "")).strip(),
        "notes": str(data.get("notes", "")).strip(),
        "sha256": str(data.get("sha256", "")).strip().lower(),
        "url": str(data.get("url", "")).strip(),
        "version": str(data.get("version", "")).strip(),
    }
    if "installer_signing" in data:
        payload["installer_signing"] = _validate_installer_signing(
            data.get("installer_signing")
        )
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _validate_release_signature(
    data: Mapping[str, Any],
    *,
    trusted_public_keys: Mapping[str, str] | None = None,
) -> str:
    """Verify the metadata authenticity signature and return its key id."""
    signature = data.get("signature")
    if not isinstance(signature, dict):
        raise ValueError("release metadata requires signature")

    algorithm = str(signature.get("algorithm", "")).strip().lower()
    key_id = str(signature.get("key_id", "")).strip()
    signature_value = str(signature.get("value", "")).strip()
    if algorithm != SIGNATURE_ALGORITHM:
        raise ValueError(f"unsupported release metadata signature algorithm: {algorithm!r}")
    if not key_id:
        raise ValueError("release metadata signature requires key_id")
    if not signature_value:
        raise ValueError("release metadata signature requires value")

    public_keys = trusted_public_keys or DEFAULT_RELEASE_PUBLIC_KEYS
    public_key_value = public_keys.get(key_id)
    if public_key_value is None:
        raise ValueError(f"untrusted release metadata signature key: {key_id!r}")

    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        public_key = Ed25519PublicKey.from_public_bytes(b64decode(public_key_value, validate=True))
        signature_bytes = b64decode(signature_value, validate=True)
        public_key.verify(signature_bytes, canonical_release_metadata_payload(data))
    except Base64Error as exc:
        raise ValueError("release metadata signature must be base64") from exc
    except InvalidSignature as exc:
        raise ValueError("release metadata signature verification failed") from exc

    return key_id


def load_release_metadata(
    path: Path,
    *,
    trusted_public_keys: Mapping[str, str] | None = None,
) -> ReleaseMetadata:
    """Load and validate local release metadata JSON.

    Expected schema:
        ``{"version": "4.2.59", "channel": "stable", "url": "...",
        "sha256": "...", "signature": {"algorithm": "ed25519", ...}}``
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("release metadata must be a JSON object")

    version = str(data.get("version", "")).strip()
    channel = str(data.get("channel", "")).strip()
    url = str(data.get("url", "")).strip()
    sha256 = str(data.get("sha256", "")).strip().lower()
    notes = str(data.get("notes", "")).strip()
    installer_signing = (
        _validate_installer_signing(data.get("installer_signing"))
        if "installer_signing" in data
        else None
    )

    if channel not in SUPPORTED_CHANNELS:
        raise ValueError(f"unsupported release channel: {channel!r}")
    if not version:
        raise ValueError("release metadata requires version")
    _version_key(version)
    if not url:
        raise ValueError("release metadata requires url")
    _validate_release_url(url, version)
    try:
        sha256 = _normalize_sha256(sha256)
    except ValueError as exc:
        raise ValueError("release metadata requires a 64-character sha256 hex digest") from exc
    signature_key_id = _validate_release_signature(
        data,
        trusted_public_keys=trusted_public_keys,
    )

    return ReleaseMetadata(
        version=version,
        channel=channel,
        url=url,
        sha256=sha256,
        notes=notes,
        signature_key_id=signature_key_id,
        installer_signing=installer_signing,
    )


def check_for_update(
    metadata_path: Path,
    *,
    current_version: str = __version__,
    channel: str = "stable",
    trusted_public_keys: Mapping[str, str] | None = None,
) -> UpdateCheckResult:
    """Evaluate a local release metadata file against the current version."""
    metadata = load_release_metadata(
        metadata_path,
        trusted_public_keys=trusted_public_keys,
    )
    if channel not in SUPPORTED_CHANNELS:
        raise ValueError(f"unsupported release channel: {channel!r}")
    if metadata.channel != channel:
        return UpdateCheckResult(
            current_version=current_version,
            latest_version=current_version,
            channel=channel,
            update_available=False,
            url="",
            sha256="",
            installer_signing=None,
        )

    return UpdateCheckResult(
        current_version=current_version,
        latest_version=metadata.version,
        channel=channel,
        update_available=is_newer_version(metadata.version, current_version),
        url=metadata.url,
        sha256=metadata.sha256,
        installer_signing=metadata.installer_signing,
    )


def _read_authenticode_signature(path: Path) -> dict[str, object]:
    """Read Windows Authenticode signature details via PowerShell."""
    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        (
            "$sig = Get-AuthenticodeSignature -LiteralPath $args[0]; "
            "$cert = $sig.SignerCertificate; "
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            "ConvertTo-Json @{"
            "Status=[string]$sig.Status;"
            "StatusMessage=[string]$sig.StatusMessage;"
            "Subject=$(if ($cert) { [string]$cert.Subject } else { '' });"
            "Thumbprint=$(if ($cert) { [string]$cert.Thumbprint } else { '' });"
            "Issuer=$(if ($cert) { [string]$cert.Issuer } else { '' })"
            "}"
        ),
        str(path),
    ]
    completed = subprocess.run(  # noqa: S603
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "powershell Authenticode check failed")
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise ValueError("unexpected Authenticode PowerShell output")
    return payload


def _verify_authenticode(
    path: Path,
    expected: Mapping[str, object],
) -> dict[str, object]:
    """Verify Windows Authenticode identity when the current OS supports it."""
    expected_publisher = str(expected.get("publisher", "")).strip()
    expected_thumbprint = _normalize_thumbprint(
        str(expected.get("certificate_thumbprint", ""))
    )
    report: dict[str, object] = {
        "checked": False,
        "status": "unavailable",
        "expected_publisher": expected_publisher,
        "expected_certificate_thumbprint": expected_thumbprint,
        "authenticode_required": bool(expected.get("authenticode_required", True)),
    }
    if platform.system().lower() != "windows":
        report["reason"] = "Authenticode verification requires Windows PowerShell"
        return report

    try:
        signature = _read_authenticode_signature(path)
    except (OSError, RuntimeError, ValueError) as exc:
        report.update({"checked": False, "status": "error", "reason": str(exc)})
        return report

    status = str(signature.get("Status", ""))
    subject = str(signature.get("Subject", ""))
    actual_thumbprint = re.sub(
        r"[^0-9a-fA-F]", "", str(signature.get("Thumbprint", ""))
    ).upper()
    publisher_ok = expected_publisher.lower() in subject.lower()
    thumbprint_ok = actual_thumbprint == expected_thumbprint
    valid = status.lower() == "valid"
    report.update(
        {
            "checked": True,
            "status": "verified" if valid and publisher_ok and thumbprint_ok else "failed",
            "signature_status": status,
            "publisher": subject,
            "certificate_thumbprint": actual_thumbprint,
            "issuer": str(signature.get("Issuer", "")),
            "publisher_ok": publisher_ok,
            "thumbprint_ok": thumbprint_ok,
        }
    )
    return report


def verify_release_artifact(
    path: Path,
    *,
    expected_sha256: str,
    installer_signing: Mapping[str, object] | None = None,
) -> ArtifactVerificationResult:
    """Verify a downloaded release artifact against signed metadata SHA-256."""
    candidate = Path(path)
    expected = _normalize_sha256(expected_sha256)
    size_bytes = candidate.stat().st_size
    actual = hash_file(candidate)
    authenticode = (
        _verify_authenticode(candidate, installer_signing)
        if installer_signing is not None
        else None
    )
    return ArtifactVerificationResult(
        path=str(candidate),
        expected_sha256=expected,
        actual_sha256=actual,
        size_bytes=size_bytes,
        verified=actual == expected,
        authenticode=authenticode,
    )


__all__ = [
    "ArtifactVerificationResult",
    "DEFAULT_RELEASE_PUBLIC_KEYS",
    "ReleaseMetadata",
    "SIGNATURE_ALGORITHM",
    "UpdateCheckResult",
    "check_for_update",
    "canonical_release_metadata_payload",
    "is_newer_version",
    "load_release_metadata",
    "verify_release_artifact",
]
