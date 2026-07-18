"""Round 646 — ROM serialization (NPZ) + size + downcast + metadata."""

from __future__ import annotations

import numpy as np
import pytest


class TestSaveLoad:
    def test_round_trip_minimal(self, tmp_path) -> None:
        from naviertwin.core.dimensionality_reduction.rom_serialization import (
            load_rom,
            save_rom,
        )

        rng = np.random.default_rng(0)
        modes = rng.standard_normal((20, 5))
        sv = np.array([5.0, 3.0, 2.0, 1.0, 0.5])
        path = tmp_path / "rom.npz"
        save_rom(path, modes=modes, singular_values=sv)
        data = load_rom(path)
        np.testing.assert_allclose(data["modes"], modes)
        np.testing.assert_allclose(data["singular_values"], sv)

    def test_round_trip_with_mean(self, tmp_path) -> None:
        from naviertwin.core.dimensionality_reduction.rom_serialization import (
            load_rom,
            save_rom,
        )

        rng = np.random.default_rng(1)
        modes = rng.standard_normal((10, 3))
        sv = np.array([2.0, 1.0, 0.5])
        mean = np.full(10, 5.0)
        path = tmp_path / "rom.npz"
        save_rom(path, modes=modes, singular_values=sv, mean=mean)
        data = load_rom(path)
        np.testing.assert_allclose(data["mean"], mean)

    def test_round_trip_with_temporal(self, tmp_path) -> None:
        from naviertwin.core.dimensionality_reduction.rom_serialization import (
            load_rom,
            save_rom,
        )

        rng = np.random.default_rng(2)
        modes = rng.standard_normal((10, 3))
        sv = np.array([2.0, 1.0, 0.5])
        T = rng.standard_normal((3, 50))
        path = tmp_path / "rom.npz"
        save_rom(path, modes=modes, singular_values=sv, temporal_coefficients=T)
        data = load_rom(path)
        np.testing.assert_allclose(data["temporal_coefficients"], T)

    def test_metadata_preserved(self, tmp_path) -> None:
        from naviertwin.core.dimensionality_reduction.rom_serialization import (
            load_rom,
            save_rom,
        )

        rng = np.random.default_rng(3)
        modes = rng.standard_normal((10, 2))
        sv = np.array([1.0, 0.5])
        path = tmp_path / "rom.npz"
        save_rom(
            path, modes=modes, singular_values=sv,
            metadata={"experiment": "cavity_re100", "seed": 42},
        )
        data = load_rom(path)
        meta = data["metadata"]
        assert meta["schema_version"] == "1.0"
        assert meta["n_modes"] == 2
        assert meta["n_space"] == 10
        assert meta["user_metadata"]["experiment"] == "cavity_re100"

    def test_modes_sv_shape_mismatch(self, tmp_path) -> None:
        from naviertwin.core.dimensionality_reduction.rom_serialization import (
            save_rom,
        )

        with pytest.raises(ValueError, match="modes/singular_values"):
            save_rom(
                tmp_path / "x.npz",
                modes=np.zeros((10, 3)),
                singular_values=np.zeros(5),
            )

    def test_mean_length_mismatch(self, tmp_path) -> None:
        from naviertwin.core.dimensionality_reduction.rom_serialization import (
            save_rom,
        )

        with pytest.raises(ValueError, match="mean"):
            save_rom(
                tmp_path / "x.npz",
                modes=np.zeros((10, 3)),
                singular_values=np.zeros(3),
                mean=np.zeros(8),
            )

    def test_temporal_shape_mismatch(self, tmp_path) -> None:
        from naviertwin.core.dimensionality_reduction.rom_serialization import (
            save_rom,
        )

        with pytest.raises(ValueError, match="temporal"):
            save_rom(
                tmp_path / "x.npz",
                modes=np.zeros((10, 3)),
                singular_values=np.zeros(3),
                temporal_coefficients=np.zeros((4, 50)),
            )

    def test_load_missing_raises(self, tmp_path) -> None:
        from naviertwin.core.dimensionality_reduction.rom_serialization import (
            load_rom,
        )

        with pytest.raises(FileNotFoundError):
            load_rom(tmp_path / "missing.npz")

    def test_non_serializable_metadata(self, tmp_path) -> None:
        from naviertwin.core.dimensionality_reduction.rom_serialization import (
            save_rom,
        )

        # 람다는 JSON 직렬화 안 됨
        with pytest.raises(ValueError, match="JSON"):
            save_rom(
                tmp_path / "x.npz",
                modes=np.zeros((5, 2)),
                singular_values=np.zeros(2),
                metadata={"f": lambda x: x},
            )


class TestRomSize:
    def test_basic(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_serialization import (
            rom_size_bytes,
        )

        modes = np.zeros((100, 10), dtype=np.float64)  # 8000 bytes
        sv = np.zeros(10)  # 80 bytes
        size = rom_size_bytes(modes, sv)
        assert size == 100 * 10 * 8 + 10 * 8

    def test_with_optional(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_serialization import (
            rom_size_bytes,
        )

        modes = np.zeros((10, 3))
        sv = np.zeros(3)
        mean = np.zeros(10)
        T = np.zeros((3, 50))
        size = rom_size_bytes(modes, sv, mean=mean, temporal_coefficients=T)
        # 240 + 24 + 80 + 1200 = 1544
        assert size == 1544


class TestCompressFloat32:
    def test_smooth_modes_compress(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_serialization import (
            compress_modes_float32,
        )

        rng = np.random.default_rng(0)
        modes = rng.standard_normal((20, 5))
        compressed = compress_modes_float32(modes, rel_tol=1e-3)
        assert compressed.dtype == np.float32

    def test_keeps_float64_if_high_precision_needed(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_serialization import (
            compress_modes_float32,
        )

        rng = np.random.default_rng(1)
        modes = rng.standard_normal((20, 5))
        # 매우 엄격한 tolerance → float32로 못 줄임
        result = compress_modes_float32(modes, rel_tol=1e-15)
        assert result.dtype == np.float64


class TestMetadataCompat:
    def test_match(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_serialization import (
            metadata_compatible,
        )

        e = {"schema_version": "1.0", "n_modes": 5, "n_space": 100}
        a = {"schema_version": "1.0", "n_modes": 5, "n_space": 100, "extra": 42}
        assert metadata_compatible(e, a)

    def test_mismatch_n_modes(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_serialization import (
            metadata_compatible,
        )

        e = {"schema_version": "1.0", "n_modes": 5, "n_space": 100}
        a = {"schema_version": "1.0", "n_modes": 4, "n_space": 100}
        assert not metadata_compatible(e, a)

    def test_custom_keys(self) -> None:
        from naviertwin.core.dimensionality_reduction.rom_serialization import (
            metadata_compatible,
        )

        e = {"a": 1, "b": 2}
        a = {"a": 1, "b": 99}
        assert metadata_compatible(e, a, keys=["a"])
        assert not metadata_compatible(e, a, keys=["b"])
