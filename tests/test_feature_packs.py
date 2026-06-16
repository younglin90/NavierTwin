"""Feature-pack runtime installation and activation tests."""

from __future__ import annotations

import json
import zipfile


def test_feature_pack_archive_installs_and_activates(tmp_path) -> None:
    from naviertwin.utils.feature_packs import (
        activate_installed_feature_packs,
        install_feature_pack_archive,
        installed_site_dir,
    )

    root = tmp_path / "packs"
    archive = tmp_path / "NavierTwinFeaturePack-ml-cpu-test.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "manifest.json",
            json.dumps({"id": "ml-cpu", "version": "test", "layout": "site"}),
        )
        zf.writestr("site/naviertwin_feature_pack_probe.py", "VALUE = 42\n")

    status = install_feature_pack_archive(archive, root=root, expected_pack_id="ml-cpu")

    assert status["installed"] is True
    assert installed_site_dir("ml-cpu", root).exists()
    activated = activate_installed_feature_packs(root)
    assert installed_site_dir("ml-cpu", root) in activated

    import naviertwin_feature_pack_probe

    assert naviertwin_feature_pack_probe.VALUE == 42


def test_feature_pack_status_checks_installer_managed_root(tmp_path, monkeypatch) -> None:
    from naviertwin.utils.feature_packs import (
        activate_installed_feature_packs,
        feature_pack_status,
        install_feature_pack_archive,
    )

    system_root = tmp_path / "programdata" / "feature-packs"
    archive = tmp_path / "NavierTwinFeaturePack-serving-test.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "manifest.json",
            json.dumps({"id": "serving", "version": "test", "layout": "site"}),
        )
        zf.writestr("site/naviertwin_installer_pack_probe.py", "VALUE = 84\n")

    install_feature_pack_archive(archive, root=system_root, expected_pack_id="serving")
    monkeypatch.setenv("NAVIER_TWIN_SYSTEM_FEATURE_PACK_DIR", str(system_root))

    status = feature_pack_status("serving")
    activated = activate_installed_feature_packs()

    assert status["installed"] is True
    assert str(system_root / "serving") in status["installed_paths"]
    assert system_root / "serving" / "site" in activated


def test_feature_pack_online_install_uses_pip_target_layout(
    tmp_path,
    monkeypatch,
) -> None:
    from naviertwin.utils import feature_packs

    root = tmp_path / "packs"
    log_file = tmp_path / "install.log"

    def fake_pip_install(packages, target_site, log_path=None, extra_index_urls=()):
        assert packages == ("fastapi", "uvicorn")
        assert log_path == log_file
        assert extra_index_urls == ()
        target_site.mkdir(parents=True, exist_ok=True)
        (target_site / "fastapi.py").write_text("VALUE = 'fastapi'\n", encoding="utf-8")
        (target_site / "uvicorn.py").write_text("VALUE = 'uvicorn'\n", encoding="utf-8")
        return 0

    monkeypatch.setattr(feature_packs, "_run_pip_install", fake_pip_install)

    status = feature_packs.install_feature_pack_online(
        "serving",
        root=root,
        log_file=log_file,
    )

    manifest = root / "serving" / "manifest.json"
    site = root / "serving" / "site"
    assert status["installed"] is True
    assert site.exists()
    assert "online-pip" in manifest.read_text(encoding="utf-8")
    assert "Installing feature pack 'serving'" in log_file.read_text(encoding="utf-8")


def test_physicsnemo_pack_uses_correct_pypi_name() -> None:
    """PhysicsNeMo PyPI 패키지명은 ``nvidia-physicsnemo`` (Python import 명은 ``physicsnemo``).

    확인 방법: ``curl -I https://pypi.org/pypi/nvidia-physicsnemo/json`` → 200,
    ``https://pypi.org/pypi/physicsnemo/json`` → 404.
    """
    from naviertwin.utils.feature_packs import FEATURE_PACKS

    pack = FEATURE_PACKS["physicsnemo"]
    assert "nvidia-physicsnemo" in pack.packages
    assert "physicsnemo" not in pack.packages
    # PyTorch CPU 휠 인덱스 — GPU 없는 PC 에서도 작은 wheel 로 설치되도록.
    assert any("download.pytorch.org/whl/cpu" in u for u in pack.extra_index_urls)


def test_ml_cpu_pack_pulls_cpu_torch_wheel_index(tmp_path, monkeypatch) -> None:
    """ml-cpu 팩은 PyTorch CPU 휠 인덱스를 ``--extra-index-url`` 로 전달해야 한다."""
    from naviertwin.utils import feature_packs

    captured: dict[str, object] = {}

    def fake_pip_install(packages, target_site, log_path=None, extra_index_urls=()):
        captured["packages"] = tuple(packages)
        captured["extra_index_urls"] = tuple(extra_index_urls)
        target_site.mkdir(parents=True, exist_ok=True)
        return 0

    monkeypatch.setattr(feature_packs, "_run_pip_install", fake_pip_install)
    feature_packs.install_feature_pack_online("ml-cpu", root=tmp_path / "packs")
    assert "torch" in captured["packages"]
    assert any(
        "download.pytorch.org/whl/cpu" in url
        for url in captured["extra_index_urls"]  # type: ignore[union-attr]
    )


def test_feature_pack_pip_install_patches_distlib_before_running(
    tmp_path,
    monkeypatch,
) -> None:
    from pip._internal.cli import main as pip_main_module

    from naviertwin.utils import feature_packs

    calls: list[str] = []

    monkeypatch.setattr(
        feature_packs,
        "_patch_pip_distlib_resource_finder",
        lambda: calls.append("patched"),
    )
    monkeypatch.setattr(pip_main_module, "main", lambda args: 0)

    exit_code = feature_packs._run_pip_install(("fastapi",), tmp_path / "site")

    assert exit_code == 0
    assert calls == ["patched"]


def test_feature_pack_default_release_url_is_versioned() -> None:
    from naviertwin import __version__
    from naviertwin.utils.feature_packs import default_release_asset_url

    url = default_release_asset_url("ml-cpu")

    assert f"/releases/download/v{__version__}/" in url
    assert f"NavierTwinFeaturePack-ml-cpu-{__version__}.zip" in url
