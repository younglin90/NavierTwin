"""Offline updater metadata contract tests."""

from __future__ import annotations

import base64
import hashlib
import json

import pytest

_TEST_KEY_ID = "test-release-key"
_TEST_PRIVATE_KEY_BYTES = bytes([17]) * 32


def _metadata(
    version: str = "4.2.59",
    channel: str = "stable",
    url: str = "https://github.com/naviertwin/naviertwin/releases/download/v4.2.59/NavierTwinSetup.exe",
    sha256: str = "a" * 64,
    installer_signing: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "version": version,
        "channel": channel,
        "url": url,
        "sha256": sha256,
        "notes": "smoke",
    }
    if installer_signing is not None:
        payload["installer_signing"] = installer_signing
    return payload


def _installer_signing() -> dict[str, object]:
    return {
        "publisher": "NavierTwin Contributors",
        "certificate_thumbprint": "a1" * 20,
        "authenticode_required": True,
    }


def _test_public_keys() -> dict[str, str]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private_key = Ed25519PrivateKey.from_private_bytes(_TEST_PRIVATE_KEY_BYTES)
    public_key = private_key.public_key()
    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return {_TEST_KEY_ID: base64.b64encode(public_bytes).decode("ascii")}


def _signed_metadata(**overrides: object) -> dict[str, object]:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from naviertwin.utils.updater import (
        SIGNATURE_ALGORITHM,
        canonical_release_metadata_payload,
    )

    payload = _metadata(**overrides)
    private_key = Ed25519PrivateKey.from_private_bytes(_TEST_PRIVATE_KEY_BYTES)
    signature = private_key.sign(canonical_release_metadata_payload(payload))
    payload["signature"] = {
        "algorithm": SIGNATURE_ALGORITHM,
        "key_id": _TEST_KEY_ID,
        "value": base64.b64encode(signature).decode("ascii"),
    }
    return payload


def test_update_metadata_detects_newer_version(tmp_path) -> None:
    from naviertwin.utils.updater import check_for_update

    path = tmp_path / "release.json"
    path.write_text(json.dumps(_signed_metadata()), encoding="utf-8")

    result = check_for_update(
        path,
        current_version="4.2.58",
        channel="stable",
        trusted_public_keys=_test_public_keys(),
    )

    assert result.update_available is True
    assert result.latest_version == "4.2.59"
    assert result.sha256 == "a" * 64


def test_verify_release_artifact_matches_expected_hash(tmp_path) -> None:
    from naviertwin.utils.updater import verify_release_artifact

    data = b"naviertwin installer bytes"
    artifact = tmp_path / "NavierTwinSetup.exe"
    artifact.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()

    result = verify_release_artifact(artifact, expected_sha256=expected)

    assert result.verified is True
    assert result.path == str(artifact)
    assert result.size_bytes == len(data)
    assert result.actual_sha256 == expected


def test_verify_release_artifact_reports_authenticode_unavailable_on_non_windows(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import naviertwin.utils.updater as updater
    from naviertwin.utils.updater import verify_release_artifact

    data = b"naviertwin installer bytes"
    artifact = tmp_path / "NavierTwinSetup.exe"
    artifact.write_bytes(data)
    expected = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(updater.platform, "system", lambda: "Linux")

    result = verify_release_artifact(
        artifact,
        expected_sha256=expected,
        installer_signing=_installer_signing(),
    )

    assert result.verified is True
    assert result.authenticode is not None
    assert result.authenticode["status"] == "unavailable"
    assert result.authenticode["checked"] is False
    assert result.authenticode["expected_certificate_thumbprint"] == ("A1" * 20)


def test_verify_release_artifact_reports_hash_mismatch(tmp_path) -> None:
    from naviertwin.utils.updater import verify_release_artifact

    artifact = tmp_path / "NavierTwinSetup.exe"
    artifact.write_bytes(b"tampered installer bytes")

    result = verify_release_artifact(artifact, expected_sha256="0" * 64)

    assert result.verified is False
    assert result.expected_sha256 == "0" * 64
    assert result.actual_sha256 != result.expected_sha256


def test_update_metadata_accepts_signed_installer_identity(tmp_path) -> None:
    from naviertwin.utils.updater import load_release_metadata

    path = tmp_path / "release.json"
    path.write_text(
        json.dumps(_signed_metadata(installer_signing=_installer_signing())),
        encoding="utf-8",
    )

    metadata = load_release_metadata(path, trusted_public_keys=_test_public_keys())

    assert metadata.installer_signing is not None
    assert metadata.installer_signing["publisher"] == "NavierTwin Contributors"
    assert metadata.installer_signing["certificate_thumbprint"] == ("A1" * 20)


def test_update_metadata_rejects_tampered_installer_identity(tmp_path) -> None:
    from naviertwin.utils.updater import load_release_metadata

    payload = _signed_metadata(installer_signing=_installer_signing())
    installer_signing = payload["installer_signing"]
    assert isinstance(installer_signing, dict)
    installer_signing["publisher"] = "Unexpected Publisher"
    path = tmp_path / "release.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="signature verification failed"):
        load_release_metadata(path, trusted_public_keys=_test_public_keys())


def test_update_metadata_rejects_unsigned_metadata(tmp_path) -> None:
    from naviertwin.utils.updater import load_release_metadata

    path = tmp_path / "release.json"
    path.write_text(json.dumps(_metadata()), encoding="utf-8")

    with pytest.raises(ValueError, match="requires signature"):
        load_release_metadata(path, trusted_public_keys=_test_public_keys())


def test_update_metadata_rejects_tampered_signed_metadata(tmp_path) -> None:
    from naviertwin.utils.updater import load_release_metadata

    payload = _signed_metadata()
    payload["version"] = "4.2.60"
    payload["url"] = "https://github.com/naviertwin/naviertwin/releases/download/v4.2.60/NavierTwinSetup.exe"
    path = tmp_path / "release.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="signature verification failed"):
        load_release_metadata(path, trusted_public_keys=_test_public_keys())


def test_update_metadata_rejects_untrusted_signature_key(tmp_path) -> None:
    from naviertwin.utils.updater import load_release_metadata

    path = tmp_path / "release.json"
    path.write_text(json.dumps(_signed_metadata()), encoding="utf-8")

    with pytest.raises(ValueError, match="untrusted"):
        load_release_metadata(path)


def test_update_metadata_rejects_invalid_integrity_hash(tmp_path) -> None:
    from naviertwin.utils.updater import load_release_metadata

    payload = _signed_metadata()
    payload["sha256"] = "not-a-hash"
    path = tmp_path / "release.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="sha256"):
        load_release_metadata(path, trusted_public_keys=_test_public_keys())


def test_update_metadata_rejects_url_version_mismatch(tmp_path) -> None:
    from naviertwin.utils.updater import load_release_metadata

    payload = _signed_metadata(
        version="4.2.60",
        url="https://github.com/naviertwin/naviertwin/releases/download/v4.2.59/NavierTwinSetup.exe",
    )
    path = tmp_path / "release.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="url tag"):
        load_release_metadata(path, trusted_public_keys=_test_public_keys())


def test_update_metadata_rejects_unexpected_installer_name(tmp_path) -> None:
    from naviertwin.utils.updater import load_release_metadata

    payload = _signed_metadata(
        url="https://github.com/naviertwin/naviertwin/releases/download/v4.2.59/OtherSetup.exe",
    )
    path = tmp_path / "release.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="NavierTwinSetup.exe"):
        load_release_metadata(path, trusted_public_keys=_test_public_keys())


def test_update_check_channel_mismatch_is_not_update(tmp_path) -> None:
    from naviertwin.utils.updater import check_for_update

    path = tmp_path / "release.json"
    path.write_text(json.dumps(_signed_metadata(channel="beta")), encoding="utf-8")

    result = check_for_update(
        path,
        current_version="4.2.58",
        channel="stable",
        trusted_public_keys=_test_public_keys(),
    )

    assert result.update_available is False
    assert result.latest_version == "4.2.58"
    assert result.url == ""


@pytest.mark.parametrize("latest", ["4.2.57", "4.2.58"])
def test_update_check_equal_or_older_version_is_not_update(tmp_path, latest) -> None:
    from naviertwin.utils.updater import check_for_update

    path = tmp_path / "release.json"
    path.write_text(
        json.dumps(
            _signed_metadata(
                version=latest,
                url=(
                    "https://github.com/naviertwin/naviertwin/releases/"
                    f"download/v{latest}/NavierTwinSetup.exe"
                ),
            )
        ),
        encoding="utf-8",
    )

    result = check_for_update(
        path,
        current_version="4.2.58",
        channel="stable",
        trusted_public_keys=_test_public_keys(),
    )

    assert result.update_available is False
    assert result.latest_version == latest
    assert result.url.endswith("/NavierTwinSetup.exe")


def test_update_check_cli_outputs_json(capsys) -> None:
    from pathlib import Path

    from naviertwin.main import _build_parser, _run_update_check

    root = Path(__file__).resolve().parents[1]
    path = root / "examples" / "release-metadata.example.json"
    args = _build_parser().parse_args(["update-check", "--metadata", str(path)])

    code = _run_update_check(
        metadata=args.metadata,
        channel=args.channel,
        current_version="4.2.58",
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    assert output["update_available"] is True
    assert output["channel"] == "stable"


def test_update_check_cli_verifies_downloaded_artifact(tmp_path, capsys) -> None:
    from naviertwin.main import _run_update_check

    data = b"downloaded installer bytes"
    artifact = tmp_path / "NavierTwinSetup.exe"
    artifact.write_bytes(data)
    metadata = tmp_path / "release.json"
    metadata.write_text(
        json.dumps(_signed_metadata(sha256=hashlib.sha256(data).hexdigest())),
        encoding="utf-8",
    )

    code = _run_update_check(
        metadata=str(metadata),
        channel="stable",
        current_version="4.2.58",
        verify_artifact=str(artifact),
        trusted_public_keys=_test_public_keys(),
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    assert output["artifact_verification"]["verified"] is True
    assert output["artifact_verification"]["path"] == str(artifact)


def test_update_check_cli_reports_installer_identity_status(
    tmp_path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import naviertwin.utils.updater as updater
    from naviertwin.main import _run_update_check

    data = b"downloaded installer bytes"
    artifact = tmp_path / "NavierTwinSetup.exe"
    artifact.write_bytes(data)
    metadata = tmp_path / "release.json"
    metadata.write_text(
        json.dumps(
            _signed_metadata(
                sha256=hashlib.sha256(data).hexdigest(),
                installer_signing=_installer_signing(),
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(updater.platform, "system", lambda: "Linux")

    code = _run_update_check(
        metadata=str(metadata),
        channel="stable",
        current_version="4.2.58",
        verify_artifact=str(artifact),
        trusted_public_keys=_test_public_keys(),
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    authenticode = output["artifact_verification"]["authenticode"]
    assert authenticode["status"] == "unavailable"
    assert authenticode["authenticode_required"] is True


def test_update_check_cli_reports_downloaded_artifact_mismatch(tmp_path, capsys) -> None:
    from naviertwin.main import _run_update_check

    artifact = tmp_path / "NavierTwinSetup.exe"
    artifact.write_bytes(b"unexpected installer bytes")
    metadata = tmp_path / "release.json"
    metadata.write_text(
        json.dumps(_signed_metadata(sha256="f" * 64)),
        encoding="utf-8",
    )

    code = _run_update_check(
        metadata=str(metadata),
        channel="stable",
        current_version="4.2.58",
        verify_artifact=str(artifact),
        trusted_public_keys=_test_public_keys(),
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 3
    assert output["artifact_verification"]["verified"] is False
    assert output["artifact_verification"]["expected_sha256"] == "f" * 64


def test_update_check_cli_reports_invalid_metadata_without_traceback(tmp_path, capsys) -> None:
    from naviertwin.main import _run_update_check

    missing = tmp_path / "missing-release.json"
    missing_code = _run_update_check(
        metadata=str(missing),
        channel="stable",
        current_version="4.2.58",
    )
    missing_output = capsys.readouterr()

    assert missing_code == 2
    assert missing_output.out == ""
    assert "update-check error:" in missing_output.err
    assert "Traceback" not in missing_output.err

    invalid = tmp_path / "invalid-release.json"
    payload = _signed_metadata()
    payload["sha256"] = "invalid"
    invalid.write_text(json.dumps(payload), encoding="utf-8")

    invalid_code = _run_update_check(
        metadata=str(invalid),
        channel="stable",
        current_version="4.2.58",
    )
    invalid_output = capsys.readouterr()

    assert invalid_code == 2
    assert invalid_output.out == ""
    assert "update-check error:" in invalid_output.err
    assert "Traceback" not in invalid_output.err


def test_release_metadata_example_is_valid() -> None:
    from pathlib import Path

    from naviertwin.utils.updater import check_for_update, load_release_metadata

    root = Path(__file__).resolve().parents[1]
    example = root / "examples" / "release-metadata.example.json"

    metadata = load_release_metadata(example)
    result = check_for_update(example, current_version="4.2.58")

    assert metadata.channel == "stable"
    assert metadata.signature_key_id == "naviertwin-release-2026q2"
    assert result.update_available is True
