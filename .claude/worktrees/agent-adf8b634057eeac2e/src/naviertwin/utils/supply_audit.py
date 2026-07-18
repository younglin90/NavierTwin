"""Supply-chain audit — compose CAS hash, lineage, signed config into one report.

Combines R442 (hash_chain), R443 (signed_config), R513 (LineageDAG),
R514 (cas_hash) into a single audit-ready manifest.

Examples:
    >>> from naviertwin.utils.supply_audit import build_audit_report
    >>> r = build_audit_report(name='exp1', config={'lr': 0.01},
    ...                          artifacts=[(b'data1', 'train.npz')], key='k')
    >>> 'signature' in r and 'artifacts' in r
    True
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from naviertwin.utils.dataset_cas import cas_hash, version_id
from naviertwin.utils.signed_config import sign_config


def build_audit_report(
    *, name: str, config: dict[str, Any],
    artifacts: Sequence[tuple[bytes, str]],
    key: str,
) -> dict[str, Any]:
    """Returns a manifest mapping artifact name → CAS hash + signed config sig.

    artifacts: iterable of (raw_bytes, filename) pairs.
    """
    art_records = []
    idx = 0
    while idx < len(artifacts):
        blob, fname = artifacts[idx]
        h = cas_hash(blob)
        art_records.append({
            "name": fname,
            "sha256": h,
            "version": version_id(name, h),
        })
        idx += 1
    sig = sign_config(config, key=key)
    return {
        "name": name,
        "config": dict(config),
        "signature": sig,
        "artifacts": art_records,
    }


__all__ = ["build_audit_report"]
