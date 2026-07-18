"""Round 443 — signed config."""

from __future__ import annotations


class TestSignedCfg:
    def test_round_trip(self) -> None:
        from naviertwin.utils.signed_config import sign_config, verify_config

        cfg = {"epochs": 10, "lr": 0.01}
        sig = sign_config(cfg, key="topsecret")
        assert verify_config(cfg, sig, key="topsecret")

    def test_tamper_detected(self) -> None:
        from naviertwin.utils.signed_config import sign_config, verify_config

        sig = sign_config({"x": 1}, key="k")
        assert not verify_config({"x": 2}, sig, key="k")

    def test_wrong_key(self) -> None:
        from naviertwin.utils.signed_config import sign_config, verify_config

        sig = sign_config({"x": 1}, key="k1")
        assert not verify_config({"x": 1}, sig, key="k2")
