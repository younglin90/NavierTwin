"""FMI/FMU export adapter exporting NavierTwin digital twins.

The exporter writes a deterministic FMI 2.0 Co-Simulation FMU archive containing
``modelDescription.xml`` and a pickled ``TwinEngine`` resource. It is intentionally
dependency-light: native FMI binaries can be added later behind an optional extra
without changing the public export surface.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape


@dataclass(frozen=True)
class FMUExportInfo:
    """Summary of a generated FMU archive."""

    path: Path
    model_name: str
    guid: str
    input_names: list[str]
    output_names: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary."""
        return {
            "path": str(self.path),
            "model_name": self.model_name,
            "guid": self.guid,
            "input_names": list(self.input_names),
            "output_names": list(self.output_names),
        }


def export_to_fmu(
    engine: object,
    path: str | Path,
    *,
    model_name: str = "NavierTwinDigitalTwin",
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    description: str = "NavierTwin reduced-order CFD digital twin",
) -> FMUExportInfo:
    """Export a fitted ``TwinEngine``-like object as an FMI 2.0 FMU archive.

    Args:
        engine: Fitted object exposing ``predict`` and preferably ``is_fitted``.
        path: Output ``.fmu`` path.
        model_name: FMI model name.
        input_names: Optional parameter names. Defaults to ``param_0..``.
        output_names: Optional output names. Defaults to ``field_0..``.
        description: Human-readable FMI model description.

    Returns:
        FMU export summary.

    Raises:
        RuntimeError: If the engine is not fitted or does not expose predict.
        ValueError: If requested names are invalid.
    """
    if not hasattr(engine, "predict"):
        raise RuntimeError("FMU export requires an engine with predict(params)")
    if getattr(engine, "is_fitted", True) is False:
        raise RuntimeError("FMU export requires a fitted engine")

    out = Path(path)
    if out.suffix.lower() != ".fmu":
        out = out.with_suffix(".fmu")
    out.parent.mkdir(parents=True, exist_ok=True)

    in_names = _resolve_input_names(engine, input_names)
    out_names = _resolve_output_names(engine, output_names)
    guid = _stable_guid(engine, model_name, in_names, out_names)
    manifest = _build_manifest(engine, model_name, guid, in_names, out_names, description)
    model_description = _render_model_description(
        model_name=model_name,
        guid=guid,
        input_names=in_names,
        output_names=out_names,
        description=description,
    )

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("modelDescription.xml", model_description)
        archive.writestr(
            "resources/naviertwin_fmu.json",
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
        archive.writestr("resources/README.txt", _resource_readme())
        archive.writestr("resources/engine.pkl", pickle.dumps(engine, protocol=pickle.HIGHEST_PROTOCOL))
        archive.writestr("documentation/index.html", _documentation_html(manifest))

    return FMUExportInfo(
        path=out,
        model_name=model_name,
        guid=guid,
        input_names=in_names,
        output_names=out_names,
    )


def inspect_fmu(path: str | Path) -> dict[str, Any]:
    """Read the NavierTwin manifest embedded in an exported FMU."""
    with zipfile.ZipFile(path) as archive:
        data = archive.read("resources/naviertwin_fmu.json")
    return json.loads(data.decode("utf-8"))


def validate_fmu_archive(path: str | Path) -> dict[str, Any]:
    """Validate NavierTwin FMU archive structure and metadata consistency."""
    required = {
        "modelDescription.xml",
        "resources/naviertwin_fmu.json",
        "resources/README.txt",
        "resources/engine.pkl",
        "documentation/index.html",
    }
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        missing = sorted(required - names)
        if missing:
            return {"status": "error", "errors": [f"missing: {','.join(missing)}"]}
        xml_text = archive.read("modelDescription.xml").decode("utf-8")
        manifest = json.loads(archive.read("resources/naviertwin_fmu.json").decode("utf-8"))

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        return {"status": "error", "errors": [f"invalid modelDescription.xml: {exc}"]}

    errors: list[str] = []
    if root.attrib.get("fmiVersion") != "2.0":
        errors.append("fmiVersion must be 2.0")
    if root.attrib.get("modelName") != manifest.get("model_name"):
        errors.append("modelName does not match manifest")
    if root.attrib.get("guid") != manifest.get("guid"):
        errors.append("guid does not match manifest")
    if root.find("CoSimulation") is None:
        errors.append("CoSimulation element missing")

    scalar_names = set(
        map(
            lambda element: element.attrib.get("name"),
            root.findall("./ModelVariables/ScalarVariable"),
        )
    )
    variable_names = [*manifest.get("input_names", []), *manifest.get("output_names", [])]
    variable_idx = 0
    while variable_idx < len(variable_names):
        name = variable_names[variable_idx]
        if name not in scalar_names:
            errors.append(f"ScalarVariable missing: {name}")
        variable_idx += 1

    return {
        "status": "error" if errors else "ok",
        "errors": errors,
        "entries": sorted(names),
        "model_name": manifest.get("model_name"),
        "guid": manifest.get("guid"),
    }


def _resolve_input_names(engine: object, names: list[str] | None) -> list[str]:
    if names is not None:
        return _validate_names(names, label="input_names")
    surrogate = getattr(engine, "surrogate", None)
    dim = int(getattr(surrogate, "input_dim", 0) or getattr(engine, "input_dim", 0) or 1)
    return list(map(lambda idx: f"param_{idx}", range(max(1, dim))))


def _resolve_output_names(engine: object, names: list[str] | None) -> list[str]:
    if names is not None:
        return _validate_names(names, label="output_names")
    reducer = getattr(engine, "reducer", None)
    modes = getattr(reducer, "modes_", None)
    if modes is not None:
        count = int(getattr(modes, "shape", [1])[0])
    else:
        count = int(getattr(engine, "output_dim", 0) or 1)
    # Keep modelDescription compact with large CFD fields while documenting the full count.
    return list(map(lambda idx: f"field_{idx}", range(min(max(1, count), 16))))


def _validate_names(names: list[str], *, label: str) -> list[str]:
    cleaned = list(filter(None, map(str.strip, names)))
    if not cleaned:
        raise ValueError(f"{label} must include at least one name")
    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"{label} must be unique")
    return cleaned


def _stable_guid(
    engine: object,
    model_name: str,
    input_names: list[str],
    output_names: list[str],
) -> str:
    payload = {
        "model_name": model_name,
        "input_names": input_names,
        "output_names": output_names,
        "engine": {
            "type": type(engine).__name__,
            "params": getattr(engine, "get_params", lambda: {})(),
        },
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def _build_manifest(
    engine: object,
    model_name: str,
    guid: str,
    input_names: list[str],
    output_names: list[str],
    description: str,
) -> dict[str, Any]:
    return {
        "format": "FMI 2.0 Co-Simulation FMU",
        "generator": "NavierTwin",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "guid": guid,
        "description": description,
        "engine_type": type(engine).__name__,
        "engine_params": getattr(engine, "get_params", lambda: {})(),
        "input_names": input_names,
        "output_names": output_names,
        "runtime": {
            "kind": "python-resource",
            "entry_resource": "resources/engine.pkl",
            "predict_method": "predict(params)",
        },
    }


def _render_model_description(
    *,
    model_name: str,
    guid: str,
    input_names: list[str],
    output_names: list[str],
    description: str,
) -> str:
    variables: list[str] = []
    value_ref = 1
    input_idx = 0
    while input_idx < len(input_names):
        name = input_names[input_idx]
        variables.append(_scalar_variable(name, value_ref, causality="input"))
        value_ref += 1
        input_idx += 1
    output_idx = 0
    while output_idx < len(output_names):
        name = output_names[output_idx]
        variables.append(_scalar_variable(name, value_ref, causality="output"))
        value_ref += 1
        output_idx += 1
    model_structure = "\n".join(
        map(
            lambda idx: f'    <Unknown index="{idx}"/>',
            range(len(input_names) + 1, len(input_names) + len(output_names) + 1),
        )
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<fmiModelDescription fmiVersion="2.0"\n'
        f'    modelName="{escape(model_name)}"\n'
        f'    guid="{guid}"\n'
        '    generationTool="NavierTwin"\n'
        f'    generationDateAndTime="{datetime.now(timezone.utc).isoformat()}"\n'
        f'    description="{escape(description)}">\n'
        f'  <CoSimulation modelIdentifier="{escape(model_name)}" canHandleVariableCommunicationStepSize="true"/>\n'
        "  <ModelVariables>\n"
        + "\n".join(variables)
        + "\n  </ModelVariables>\n"
        "  <ModelStructure>\n"
        f"{model_structure}\n"
        "  </ModelStructure>\n"
        "</fmiModelDescription>\n"
    )


def _scalar_variable(name: str, value_ref: int, *, causality: str) -> str:
    return (
        f'    <ScalarVariable name="{escape(name)}" valueReference="{value_ref}" '
        f'causality="{causality}" variability="continuous">\n'
        "      <Real start=\"0.0\"/>\n"
        "    </ScalarVariable>"
    )


def _resource_readme() -> str:
    return (
        "NavierTwin FMU resource archive\n"
        "===============================\n\n"
        "This FMU contains FMI 2.0 model metadata plus a Python TwinEngine resource.\n"
        "Native platform binaries are intentionally not bundled; importers can use\n"
        "resources/engine.pkl with NavierTwin to evaluate predict(params).\n"
    )


def _documentation_html(manifest: dict[str, Any]) -> str:
    model_name = escape(str(manifest["model_name"]))
    description = escape(str(manifest["description"]))
    return (
        "<!doctype html>\n"
        "<html><head><meta charset=\"utf-8\"><title>NavierTwin FMU</title></head>\n"
        "<body>\n"
        f"<h1>{model_name}</h1>\n"
        f"<p>{description}</p>\n"
        "<p>This FMU contains FMI 2.0 metadata and a NavierTwin Python resource "
        "to evaluate the exported digital twin.</p>\n"
        "</body></html>\n"
    )


__all__ = ["FMUExportInfo", "export_to_fmu", "inspect_fmu", "validate_fmu_archive"]
