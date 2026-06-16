"""Build a downloadable NavierTwin feature-pack ZIP.

The output archive layout is:

```
manifest.json
site/<packages installed with pip --target>
```

Release asset names follow ``NavierTwinFeaturePack-<pack>-<version>.zip``.
Customers install the ZIP with ``naviertwin feature-pack install --archive`` or
through the GUI Library panel.  Runtime activation prepends ``site/`` to
``sys.path`` and native-library lookup paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from naviertwin import __version__  # noqa: E402
from naviertwin.utils.feature_packs import (  # noqa: E402
    FEATURE_PACKS,
    build_archive_manifest,
    get_feature_pack_spec,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", required=True, choices=sorted(FEATURE_PACKS))
    parser.add_argument("--output-dir", default="dist/feature-packs")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--version", default=__version__)
    parser.add_argument("--skip-pip-install", action="store_true", default=False)
    parser.add_argument(
        "--package",
        action="append",
        default=None,
        help="Override/add package requirement. Can be passed multiple times.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    spec = get_feature_pack_spec(args.pack)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / spec.asset_name(args.version)

    with tempfile.TemporaryDirectory(prefix=f"naviertwin-{args.pack}-") as tmp:
        staging = Path(tmp)
        site = staging / "site"
        site.mkdir()
        packages = tuple(args.package or spec.packages)
        if not args.skip_pip_install:
            cmd = [
                args.python,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--target",
                str(site),
                *packages,
            ]
            subprocess.run(cmd, check=True)
        manifest = build_archive_manifest(args.pack, version=args.version)
        manifest["packages"] = list(packages)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_zip(staging, output_path)

    digest = _sha256(output_path)
    print(
        json.dumps(
            {
                "status": "ok",
                "pack": args.pack,
                "output": str(output_path),
                "sha256": digest,
                "bytes": output_path.stat().st_size,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def _write_zip(source_dir: Path, output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(source_dir.rglob("*")):
            if path.is_dir():
                continue
            zf.write(path, path.relative_to(source_dir).as_posix())


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
