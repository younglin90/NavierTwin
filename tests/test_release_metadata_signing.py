"""Release metadata signing tool tests."""

from __future__ import annotations

import base64
import json

_TEST_KEY_ID = "test-release-key"
_TEST_PRIVATE_KEY = base64.b64encode(bytes([29]) * 32).decode("ascii")


def _unsigned_metadata() -> dict[str, object]:
    return {
        "version": "4.2.59",
        "channel": "stable",
        "url": "https://github.com/naviertwin/naviertwin/releases/download/v4.2.59/NavierTwinSetup.exe",
        "sha256": "c" * 64,
        "notes": "release ops smoke",
    }


def test_sign_release_metadata_payload_verifies_with_matching_public_key(tmp_path) -> None:
    from naviertwin.utils.updater import load_release_metadata
    from scripts.sign_release_metadata import public_key_base64, sign_release_metadata_payload

    signed = sign_release_metadata_payload(
        _unsigned_metadata(),
        key_id=_TEST_KEY_ID,
        private_key_base64=_TEST_PRIVATE_KEY,
    )
    path = tmp_path / "signed-release.json"
    path.write_text(json.dumps(signed), encoding="utf-8")

    metadata = load_release_metadata(
        path,
        trusted_public_keys={_TEST_KEY_ID: public_key_base64(_TEST_PRIVATE_KEY)},
    )

    assert metadata.signature_key_id == _TEST_KEY_ID
    assert signed["signature"]["algorithm"] == "ed25519"  # type: ignore[index]


def test_sign_release_metadata_main_writes_signed_file(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    from naviertwin.utils.updater import load_release_metadata
    from scripts.sign_release_metadata import main, public_key_base64

    unsigned = tmp_path / "release-unsigned.json"
    signed = tmp_path / "release-signed.json"
    unsigned.write_text(json.dumps(_unsigned_metadata()), encoding="utf-8")
    monkeypatch.setenv("NAVIER_TWIN_RELEASE_PRIVATE_KEY_B64", _TEST_PRIVATE_KEY)

    code = main(
        [
            "--input",
            str(unsigned),
            "--output",
            str(signed),
            "--key-id",
            _TEST_KEY_ID,
            "--print-public-key",
        ]
    )
    output = capsys.readouterr()

    assert code == 0
    assert output.out.strip() == public_key_base64(_TEST_PRIVATE_KEY)
    assert output.err == ""
    metadata = load_release_metadata(
        signed,
        trusted_public_keys={_TEST_KEY_ID: output.out.strip()},
    )
    assert metadata.version == "4.2.59"


def test_sign_release_metadata_main_reports_missing_private_key_without_traceback(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    from scripts.sign_release_metadata import main

    unsigned = tmp_path / "release-unsigned.json"
    signed = tmp_path / "release-signed.json"
    unsigned.write_text(json.dumps(_unsigned_metadata()), encoding="utf-8")
    monkeypatch.delenv("NAVIER_TWIN_RELEASE_PRIVATE_KEY_B64", raising=False)

    code = main(
        [
            "--input",
            str(unsigned),
            "--output",
            str(signed),
            "--key-id",
            _TEST_KEY_ID,
        ]
    )
    output = capsys.readouterr()

    assert code == 2
    assert output.out == ""
    assert "sign-release-metadata error:" in output.err
    assert "Traceback" not in output.err
