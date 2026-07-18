"""R658 — Op 파라미터 프리셋 검증."""

from __future__ import annotations

import json

import pytest


class TestFactoryPresets:
    def test_psd_welch_has_presets(self) -> None:
        from naviertwin.core.post_process_presets import factory_presets

        p = factory_presets("psd_welch")
        assert "high_resolution" in p
        assert p["high_resolution"]["fs"] == 1000.0

    def test_unknown_op_empty(self) -> None:
        from naviertwin.core.post_process_presets import factory_presets

        assert factory_presets("nonexistent_op") == {}

    def test_list_factory_ops(self) -> None:
        from naviertwin.core.post_process_presets import list_factory_preset_ops

        ops = list_factory_preset_ops()
        assert len(ops) >= 10
        assert "psd_welch" in ops
        assert "denoise" in ops


class TestPresetStore:
    def test_in_memory_add_get(self) -> None:
        from naviertwin.core.post_process_presets import PresetStore

        store = PresetStore()
        store.add("psd_welch", "my_setup", {"fs": 500.0, "nperseg": 512})
        assert store.get("psd_welch", "my_setup") == {
            "fs": 500.0, "nperseg": 512,
        }

    def test_user_overrides_factory(self) -> None:
        from naviertwin.core.post_process_presets import PresetStore

        store = PresetStore()
        # factory에 "default"가 있는 op (eof) 덮어쓰기
        store.add("eof", "default", {"n_modes": 99})
        merged = store.merged("eof")
        assert merged["default"]["n_modes"] == 99

    def test_list_presets_includes_factory(self) -> None:
        from naviertwin.core.post_process_presets import PresetStore

        store = PresetStore()
        names = store.list_presets("psd_welch")
        # factory에 3개 (low/high/no_window)
        assert "high_resolution" in names

    def test_remove(self) -> None:
        from naviertwin.core.post_process_presets import PresetStore

        store = PresetStore()
        store.add("psd_welch", "foo", {"fs": 1.0})
        assert store.remove("psd_welch", "foo") is True
        assert store.user_presets("psd_welch") == {}

    def test_remove_nonexistent(self) -> None:
        from naviertwin.core.post_process_presets import PresetStore

        store = PresetStore()
        assert store.remove("psd_welch", "nope") is False

    def test_save_load_round_trip(self, tmp_path) -> None:
        from naviertwin.core.post_process_presets import PresetStore

        path = tmp_path / "presets.json"
        store = PresetStore(path=path)
        store.add("eof", "experiment_a", {"n_modes": 7})
        store.add("denoise", "smooth_x", {"window_length": 21, "polyorder": 5})
        store.save()
        # JSON 유효
        data = json.loads(path.read_text())
        assert "eof" in data

        # 새 인스턴스로 로드
        store2 = PresetStore(path=path)
        assert store2.get("eof", "experiment_a") == {"n_modes": 7}
        assert store2.user_presets("denoise")["smooth_x"]["polyorder"] == 5

    def test_corrupt_json_resets_user(self, tmp_path) -> None:
        from naviertwin.core.post_process_presets import PresetStore

        path = tmp_path / "bad.json"
        path.write_text("not valid json {{{")
        store = PresetStore(path=path)
        # 손상된 파일 → 빈 user
        assert store.user_presets("psd_welch") == {}

    def test_empty_preset_name_raises(self) -> None:
        from naviertwin.core.post_process_presets import PresetStore

        with pytest.raises(ValueError, match="preset_name"):
            PresetStore().add("psd_welch", "", {"fs": 1.0})

    def test_no_path_save_noop(self) -> None:
        from naviertwin.core.post_process_presets import PresetStore

        store = PresetStore()  # path 없음
        store.add("eof", "foo", {"n_modes": 3})
        store.save()  # 예외 안 던짐


class TestPresetMatching:
    def test_all_factory_presets_have_valid_keys(self) -> None:
        """모든 factory 프리셋의 키가 facade의 scalar param에 존재하는지."""
        from naviertwin.core.post_process_facade import PostProcessFacade
        from naviertwin.core.post_process_presets import (
            factory_presets,
            list_factory_preset_ops,
        )

        facade = PostProcessFacade()
        for op in list_factory_preset_ops():
            spec = facade.scalar_param_specs(op)
            for preset_name, preset in factory_presets(op).items():
                for k in preset:
                    assert k in spec, (
                        f"{op}.{preset_name} 의 키 '{k}'가 "
                        f"facade scalar params에 없음"
                    )
