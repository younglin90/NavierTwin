"""Artifact zipper + manifest.

Examples:
    >>> import tempfile, pathlib
    >>> from naviertwin.utils.workflow.artifact_zip import zip_artifacts
    >>> with tempfile.TemporaryDirectory() as d:
    ...     p = pathlib.Path(d) / 'f.txt'
    ...     p.write_text('hi')
    ...     out = zip_artifacts([p], pathlib.Path(d) / 'a.zip')
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from naviertwin.utils.dataset_cas import cas_hash_file


def zip_artifacts(
    files: list[str | Path], out_path: str | Path,
    *, manifest_name: str = "MANIFEST.json",
) -> Path:
    out = Path(out_path)
    manifest = []
    with zipfile.ZipFile(out, "w") as zf:
        for f in files:
            fp = Path(f)
            zf.write(fp, arcname=fp.name)
            manifest.append({"name": fp.name, "sha256": cas_hash_file(fp)})
        zf.writestr(manifest_name, json.dumps(manifest, indent=2))
    return out


def read_manifest(zip_path: str | Path, *, manifest_name: str = "MANIFEST.json") -> list[dict]:
    with zipfile.ZipFile(zip_path) as zf:
        return json.loads(zf.read(manifest_name))


__all__ = ["read_manifest", "zip_artifacts"]
