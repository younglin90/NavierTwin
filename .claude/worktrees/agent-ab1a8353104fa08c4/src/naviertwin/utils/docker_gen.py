"""Dockerfile generator (CUDA-ready NavierTwin runtime).

Examples:
    >>> from naviertwin.utils.docker_gen import dockerfile_text
    >>> "FROM" in dockerfile_text()
    True
"""

from __future__ import annotations

from pathlib import Path

_TEMPLATE = """\
FROM nvidia/cuda:{cuda_version}-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \\
    python3 python3-pip git \\
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/naviertwin
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .{extras}

CMD ["python3", "-m", "naviertwin"]
"""


def dockerfile_text(
    cuda_version: str = "12.1.0", extras: str | None = None,
) -> str:
    return _TEMPLATE.format(
        cuda_version=cuda_version,
        extras=f"[{extras}]" if extras else "",
    )


def write_dockerfile(
    path: str | Path = "Dockerfile",
    cuda_version: str = "12.1.0",
    extras: str | None = None,
) -> None:
    Path(path).write_text(dockerfile_text(cuda_version, extras))


__all__ = ["dockerfile_text", "write_dockerfile"]
