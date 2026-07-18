"""Round 565 — supply-chain audit composition."""

from __future__ import annotations

from naviertwin.utils.signed_config import verify_config


class TestSupplyAudit:
    def test_basic(self) -> None:
        from naviertwin.utils.supply_audit import build_audit_report

        report = build_audit_report(
            name="run1",
            config={"lr": 0.01, "epochs": 5},
            artifacts=[(b"alpha-data", "train.npz"), (b"beta-data", "val.npz")],
            key="topsecret",
        )
        assert report["name"] == "run1"
        assert len(report["artifacts"]) == 2
        # different blobs → different hashes
        assert report["artifacts"][0]["sha256"] != report["artifacts"][1]["sha256"]
        # signature verifies
        assert verify_config(report["config"], report["signature"], key="topsecret")

    def test_tampered_signature(self) -> None:
        from naviertwin.utils.supply_audit import build_audit_report

        r = build_audit_report(name="x", config={"a": 1},
                                  artifacts=[], key="k1")
        assert not verify_config(r["config"], r["signature"], key="k2")
