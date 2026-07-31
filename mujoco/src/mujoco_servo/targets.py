from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from .config import GraspPoint, TargetPart, TargetSpec

TARGETS: dict[str, TargetSpec] = {
    "grasp-cube": TargetSpec(
        "grasp-cube",
        "box",
        (0.045, 0.045, 0.065),
        (0.12, 0.72, 0.92, 1.0),
        ("grasp cube", "training block", "cyan block"),
        mass=0.08,
        friction=(1.2, 0.01, 0.002),
        dynamics="physical",
        grasp_points=(GraspPoint("center", width_m=0.045),),
    ),
    "cup": TargetSpec(
        "cup",
        "compound",
        (0.095, 0.075, 0.105),
        (0.95, 0.32, 0.18, 1.0),
        ("mug", "red cup"),
        parts=(
            TargetPart("cylinder", (0.075, 0.075, 0.105), rgba=(0.95, 0.32, 0.18, 1.0)),
            TargetPart(
                "capsule",
                (0.014, 0.014, 0.070),
                pos=(0.048, 0.0, 0.005),
                rgba=(0.95, 0.32, 0.18, 1.0),
                quat=(0.7071, 0.0, 0.7071, 0.0),
            ),
        ),
    ),
    "apple": TargetSpec(
        "apple",
        "compound",
        (0.085, 0.085, 0.105),
        (0.9, 0.08, 0.10, 1.0),
        ("red apple", "fruit"),
        parts=(
            TargetPart("sphere", (0.080, 0.080, 0.080), rgba=(0.9, 0.08, 0.10, 1.0)),
            TargetPart(
                "capsule",
                (0.010, 0.010, 0.038),
                pos=(0.0, 0.0, 0.052),
                rgba=(0.34, 0.18, 0.08, 1.0),
            ),
            TargetPart(
                "box",
                (0.035, 0.014, 0.006),
                pos=(0.020, 0.0, 0.066),
                rgba=(0.12, 0.55, 0.16, 1.0),
                quat=(0.9239, 0.0, 0.3827, 0.0),
            ),
        ),
    ),
    "box": TargetSpec(
        "box",
        "box",
        (0.11, 0.085, 0.09),
        (0.18, 0.45, 0.92, 1.0),
        ("blue box", "cube", "block"),
    ),
    "bottle": TargetSpec(
        "bottle",
        "cylinder",
        (0.052, 0.052, 0.22),
        (0.10, 0.55, 0.85, 1.0),
        ("blue bottle",),
    ),
    "phone": TargetSpec(
        "phone",
        "box",
        (0.075, 0.014, 0.145),
        (0.08, 0.08, 0.09, 1.0),
        ("mobile", "black phone"),
    ),
    "capsule": TargetSpec(
        "capsule",
        "capsule",
        (0.045, 0.045, 0.16),
        (0.55, 0.85, 0.25, 1.0),
        ("pill", "green capsule"),
    ),
    "sphere": TargetSpec(
        "sphere",
        "sphere",
        (0.085, 0.085, 0.085),
        (0.96, 0.85, 0.18, 1.0),
        ("ball", "yellow sphere"),
    ),
    "cylinder": TargetSpec(
        "cylinder",
        "cylinder",
        (0.075, 0.075, 0.13),
        (0.55, 0.25, 0.9, 1.0),
        ("can", "purple cylinder"),
    ),
    "hammer": TargetSpec(
        "hammer",
        "compound",
        (0.18, 0.05, 0.12),
        (0.45, 0.26, 0.12, 1.0),
        ("tool", "mallet"),
        parts=(
            TargetPart(
                "capsule",
                (0.018, 0.018, 0.18),
                pos=(0.0, 0.0, 0.0),
                rgba=(0.45, 0.26, 0.12, 1.0),
                quat=(0.7071, 0.0, 0.7071, 0.0),
            ),
            TargetPart(
                "box",
                (0.095, 0.040, 0.040),
                pos=(0.075, 0.0, 0.0),
                rgba=(0.15, 0.15, 0.16, 1.0),
            ),
        ),
    ),
    "dumbbell": TargetSpec(
        "dumbbell",
        "compound",
        (0.18, 0.055, 0.055),
        (0.10, 0.70, 0.62, 1.0),
        ("barbell", "weight"),
        parts=(
            TargetPart(
                "capsule",
                (0.015, 0.015, 0.16),
                rgba=(0.10, 0.70, 0.62, 1.0),
                quat=(0.7071, 0.0, 0.7071, 0.0),
            ),
            TargetPart(
                "sphere",
                (0.052, 0.052, 0.052),
                pos=(-0.085, 0.0, 0.0),
                rgba=(0.08, 0.45, 0.40, 1.0),
            ),
            TargetPart(
                "sphere",
                (0.052, 0.052, 0.052),
                pos=(0.085, 0.0, 0.0),
                rgba=(0.08, 0.45, 0.40, 1.0),
            ),
        ),
    ),
    "tower": TargetSpec(
        "tower",
        "compound",
        (0.08, 0.08, 0.18),
        (0.92, 0.55, 0.12, 1.0),
        ("stack", "stacked blocks"),
        parts=(
            TargetPart(
                "box",
                (0.090, 0.090, 0.045),
                pos=(0.0, 0.0, -0.045),
                rgba=(0.90, 0.30, 0.18, 1.0),
            ),
            TargetPart(
                "box",
                (0.070, 0.070, 0.045),
                pos=(0.0, 0.0, 0.000),
                rgba=(0.18, 0.48, 0.90, 1.0),
            ),
            TargetPart(
                "box",
                (0.052, 0.052, 0.045),
                pos=(0.0, 0.0, 0.045),
                rgba=(0.95, 0.82, 0.20, 1.0),
            ),
        ),
    ),
}


_PRIMITIVE_SHAPES = frozenset({"box", "sphere", "cylinder", "capsule"})
_TARGET_SHAPES = _PRIMITIVE_SHAPES | {"compound", "mesh"}
_PART_SHAPES = _PRIMITIVE_SHAPES | {"mesh"}
_MESH_SUFFIXES = frozenset({".obj", ".stl"})
_TARGET_FIELDS = frozenset(
    {
        "name",
        "shape",
        "size",
        "rgba",
        "aliases",
        "base_position",
        "parts",
        "mesh_file",
        "mesh_path",
        "scale",
        "mesh_scale",
        "schema_version",
        "quat",
        "mass",
        "friction",
        "dynamics",
        "grasp_points",
    }
)
_PART_FIELDS = frozenset(
    {
        "shape",
        "size",
        "pos",
        "offset",
        "rgba",
        "quat",
        "mesh_file",
        "mesh_path",
        "scale",
        "mesh_scale",
    }
)

BASE_POSITIONS: dict[str, np.ndarray] = {
    "cup": np.array([0.48, 0.02, 0.34], dtype=float),
    "apple": np.array([0.44, 0.13, 0.33], dtype=float),
    "box": np.array([0.50, -0.10, 0.34], dtype=float),
    "bottle": np.array([0.43, -0.18, 0.42], dtype=float),
    "phone": np.array([0.52, 0.08, 0.33], dtype=float),
    "capsule": np.array([0.48, 0.12, 0.35], dtype=float),
    "sphere": np.array([0.46, -0.12, 0.34], dtype=float),
    "cylinder": np.array([0.51, 0.00, 0.36], dtype=float),
    "hammer": np.array([0.50, -0.10, 0.35], dtype=float),
    "dumbbell": np.array([0.48, 0.12, 0.34], dtype=float),
    "tower": np.array([0.44, 0.02, 0.40], dtype=float),
}


def load_target_specs(path: str | Path | None) -> dict[str, TargetSpec]:
    if path is None:
        return {}
    source = Path(path).expanduser()
    payload = json.loads(
        source.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_json_object,
        parse_constant=_reject_json_constant,
    )
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        unknown = set(payload) - {"targets", "schema_version"}
        if unknown:
            raise ValueError(
                f"target file contains unknown top-level fields: {', '.join(sorted(unknown))}"
            )
        _schema_version(payload.get("schema_version", 1), "target file")
        entries = payload.get("targets")
    else:
        entries = None
    if not isinstance(entries, list):
        raise ValueError(
            "target file must contain a list or a {'targets': [...]} object"
        )
    base_dir = source.resolve().parent
    specs: dict[str, TargetSpec] = {}
    tokens: dict[str, str] = {}
    for entry in entries:
        spec = _target_from_mapping(entry, base_dir)
        if spec.name in specs:
            raise ValueError(f"duplicate target name '{spec.name}'")
        _register_target_tokens(tokens, spec)
        specs[spec.name] = spec
    return specs


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key '{key}'")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid non-finite JSON number '{value}'")


def _target_from_mapping(entry: Any, base_dir: Path | None = None) -> TargetSpec:
    if not isinstance(entry, dict):
        raise ValueError("target entries must be objects")
    _reject_unknown_fields(entry, _TARGET_FIELDS, "target")
    root = Path.cwd() if base_dir is None else base_dir
    schema_version = _schema_version(entry.get("schema_version", 1), "target")
    name = _required_text(entry, "name")
    raw_parts = entry.get("parts", [])
    if not isinstance(raw_parts, list):
        raise ValueError("parts must be a list")
    default_shape = "compound" if raw_parts else "box"
    shape = _shape_value(entry.get("shape", default_shape), "shape", _TARGET_SHAPES)
    raw_size = entry.get("size")
    rgba = _rgba_tuple(entry.get("rgba", (0.85, 0.25, 0.25, 1.0)), "rgba")
    aliases = _aliases_tuple(entry.get("aliases", []))
    base = entry.get("base_position")
    base_position = None if base is None else _float_tuple(base, 3, "base_position")
    parts = tuple(_target_part_from_mapping(part, root) for part in raw_parts)
    mesh_path = _mesh_path_from_mapping(entry, root, "target")
    mesh_scale = _mesh_scale_from_mapping(entry, "target")
    if shape == "mesh" and raw_size is None and mesh_path is not None:
        size = _infer_mesh_size(mesh_path, mesh_scale)
    else:
        size = _positive_float_tuple(
            (0.10, 0.10, 0.10) if raw_size is None else raw_size, 3, "size"
        )
    _validate_shape_size(shape, size, "size")
    if shape == "compound":
        if not parts:
            raise ValueError("compound target must contain at least one part")
        if mesh_path is not None or "scale" in entry or "mesh_scale" in entry:
            raise ValueError(
                "compound target cannot define a top-level mesh or mesh scale"
            )
    elif shape == "mesh":
        if parts:
            raise ValueError("mesh target cannot also contain parts")
        if mesh_path is None:
            raise ValueError("mesh target requires mesh_file or mesh_path")
    else:
        if parts:
            raise ValueError(
                f"{shape} target cannot also contain parts; use shape 'compound'"
            )
        if mesh_path is not None or "scale" in entry or "mesh_scale" in entry:
            raise ValueError(
                f"{shape} target cannot define mesh_file, mesh_path, or mesh scale"
            )
    quat = _quat_tuple(entry.get("quat", (1.0, 0.0, 0.0, 0.0)), "quat")
    mass = _positive_number(entry.get("mass", 0.10), "mass")
    friction = _positive_or_zero_tuple(
        entry.get("friction", (0.8, 0.005, 0.0001)), 3, "friction"
    )
    dynamics = _dynamics_value(entry.get("dynamics", "visual"))
    grasp_points = _grasp_points_tuple(entry.get("grasp_points"), size)
    return TargetSpec(
        name=name,
        shape=shape,
        size=size,
        rgba=rgba,
        aliases=aliases,
        parts=parts,
        base_position=base_position,
        mesh_path=mesh_path,
        mesh_scale=mesh_scale,
        quat=quat,
        mass=mass,
        friction=friction,
        dynamics=dynamics,
        grasp_points=grasp_points,
        schema_version=schema_version,
    )


def _target_part_from_mapping(entry: Any, base_dir: Path | None = None) -> TargetPart:
    if not isinstance(entry, dict):
        raise ValueError("target parts must be objects")
    _reject_unknown_fields(entry, _PART_FIELDS, "target part")
    if "pos" in entry and "offset" in entry:
        raise ValueError("target part must use only one of pos or offset")
    root = Path.cwd() if base_dir is None else base_dir
    shape = _shape_value(entry.get("shape", "box"), "part.shape", _PART_SHAPES)
    rgba = entry.get("rgba")
    quat = entry.get("quat")
    pos_value = entry.get("pos", entry.get("offset", (0.0, 0.0, 0.0)))
    mesh_path = _mesh_path_from_mapping(entry, root, "target part")
    mesh_scale = _mesh_scale_from_mapping(entry, "target part")
    raw_size = entry.get("size")
    if shape == "mesh":
        if mesh_path is None:
            raise ValueError("mesh target part requires mesh_file or mesh_path")
    elif mesh_path is not None or "scale" in entry or "mesh_scale" in entry:
        raise ValueError(
            f"{shape} target part cannot define mesh_file, mesh_path, or mesh scale"
        )
    if shape == "mesh" and raw_size is None and mesh_path is not None:
        size = _infer_mesh_size(mesh_path, mesh_scale)
    else:
        size = _positive_float_tuple(
            (0.05, 0.05, 0.05) if raw_size is None else raw_size, 3, "part.size"
        )
    _validate_shape_size(shape, size, "part.size")
    return TargetPart(
        shape=shape,
        size=size,
        pos=_float_tuple(pos_value, 3, "part.pos"),
        rgba=None if rgba is None else _rgba_tuple(rgba, "part.rgba"),
        quat=None if quat is None else _quat_tuple(quat, "part.quat"),
        mesh_path=mesh_path,
        mesh_scale=mesh_scale,
    )


def _required_text(entry: dict[str, Any], key: str) -> str:
    raw_value = entry.get(key)
    if not isinstance(raw_value, str):
        raise ValueError(f"target entry '{key}' must be a string")
    value = _normalize_text(raw_value)
    if not value:
        raise ValueError(f"target entry missing '{key}'")
    return value


def _aliases_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("aliases must be a list of strings")
    aliases: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"aliases[{index}] must be a string")
        alias = _normalize_text(item)
        if not alias:
            raise ValueError(f"aliases[{index}] must be non-empty")
        aliases.append(alias)
    return tuple(aliases)


def _float_tuple(value: Any, expected: int, field: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != expected:
        raise ValueError(f"{field} must contain {expected} numbers")
    if any(isinstance(item, bool) or not isinstance(item, Real) for item in value):
        raise ValueError(f"{field} must contain {expected} numbers")
    values = tuple(float(item) for item in value)
    if not np.isfinite(values).all():
        raise ValueError(f"{field} must contain finite numbers")
    return values


def _positive_float_tuple(value: Any, expected: int, field: str) -> tuple[float, ...]:
    values = _float_tuple(value, expected, field)
    if any(item <= 0.0 for item in values):
        raise ValueError(f"{field} values must be positive")
    return values


def _positive_or_zero_tuple(value: Any, expected: int, field: str) -> tuple[float, ...]:
    values = _float_tuple(value, expected, field)
    if any(item < 0.0 for item in values):
        raise ValueError(f"{field} values must be non-negative")
    return values


def _positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite positive number")
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field} must be a finite positive number")
    return number


def _quat_tuple(value: Any, field: str) -> tuple[float, float, float, float]:
    quat = np.asarray(_float_tuple(value, 4, field), dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        raise ValueError(f"{field} must be non-zero")
    normalized = quat / norm
    return tuple(float(item) for item in normalized)


def _rgba_tuple(value: Any, field: str) -> tuple[float, float, float, float]:
    rgba = _float_tuple(value, 4, field)
    if any(item < 0.0 or item > 1.0 for item in rgba):
        raise ValueError(f"{field} values must be in [0, 1]")
    return rgba


def _dynamics_value(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("dynamics must be a string")
    normalized = value.strip().lower()
    if normalized not in {"visual", "physical"}:
        raise ValueError("dynamics must be one of physical, visual")
    return normalized


def _grasp_points_tuple(
    value: Any, size: tuple[float, float, float]
) -> tuple[GraspPoint, ...]:
    if value is None:
        return (GraspPoint(width_m=float(min(size[0], size[1]))),)
    if not isinstance(value, list) or not value:
        raise ValueError("grasp_points must be a non-empty list")
    points: list[GraspPoint] = []
    names: set[str] = set()
    allowed = {"name", "position", "pos", "approach", "width", "width_m"}
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"grasp_points[{index}] must be an object")
        unknown = set(item) - allowed
        if unknown:
            raise ValueError(
                f"grasp_points[{index}] contains unknown fields: {', '.join(sorted(unknown))}"
            )
        if "position" in item and "pos" in item:
            raise ValueError(
                f"grasp_points[{index}] must use only one of position or pos"
            )
        if "width" in item and "width_m" in item:
            raise ValueError(
                f"grasp_points[{index}] must use only one of width or width_m"
            )
        name_value = item.get("name", f"grasp-{index}")
        if not isinstance(name_value, str) or not name_value.strip():
            raise ValueError(f"grasp_points[{index}].name must be non-empty text")
        name = _normalize_text(name_value)
        if name in names:
            raise ValueError(f"duplicate grasp point name '{name}'")
        names.add(name)
        position = _float_tuple(
            item.get("position", item.get("pos", (0.0, 0.0, 0.0))),
            3,
            f"grasp_points[{index}].position",
        )
        approach_array = np.asarray(
            _float_tuple(
                item.get("approach", (0.0, 0.0, 1.0)),
                3,
                f"grasp_points[{index}].approach",
            ),
            dtype=float,
        )
        norm = float(np.linalg.norm(approach_array))
        if norm <= 1e-12:
            raise ValueError(f"grasp_points[{index}].approach must be non-zero")
        width_value = item.get("width_m", item.get("width"))
        width = (
            None
            if width_value is None
            else _positive_number(width_value, f"grasp_points[{index}].width_m")
        )
        points.append(
            GraspPoint(
                name=name,
                position=position,
                approach=tuple(float(component) for component in approach_array / norm),
                width_m=width,
            )
        )
    return tuple(points)


def _schema_version(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field}.schema_version must be an integer")
    if value != 1:
        raise ValueError(
            f"{field}.schema_version {value} is unsupported; supported versions: 1"
        )
    return value


def _shape_value(value: Any, field: str, supported: frozenset[str]) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    shape = value.strip().lower()
    if shape not in supported:
        raise ValueError(f"{field} must be one of {', '.join(sorted(supported))}")
    return shape


def _validate_shape_size(shape: str, size: tuple[float, ...], field: str) -> None:
    if shape == "sphere" and not np.allclose(size, size[0], rtol=1e-6, atol=1e-9):
        raise ValueError(f"{field} for sphere must contain three equal diameters")
    if shape in {"cylinder", "capsule"} and not np.isclose(
        size[0], size[1], rtol=1e-6, atol=1e-9
    ):
        raise ValueError(f"{field} for {shape} must use equal x/y diameters")
    if shape == "capsule" and size[2] <= size[0]:
        raise ValueError(
            f"{field} for capsule must have height greater than its diameter"
        )


def _reject_unknown_fields(
    entry: dict[str, Any], allowed: frozenset[str], label: str
) -> None:
    unknown = set(entry) - allowed
    if unknown:
        raise ValueError(
            f"{label} contains unknown fields: {', '.join(sorted(unknown))}"
        )


def _mesh_path_from_mapping(
    entry: dict[str, Any], base_dir: Path, label: str
) -> Path | None:
    keys = [key for key in ("mesh_file", "mesh_path") if key in entry]
    if len(keys) > 1:
        raise ValueError(f"{label} must use only one of mesh_file or mesh_path")
    if not keys:
        return None
    raw_path = entry[keys[0]]
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{label} {keys[0]} must be a non-empty path string")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if path.suffix.lower() not in _MESH_SUFFIXES:
        raise ValueError(f"{label} mesh must be an OBJ or STL file")
    if not path.is_file():
        raise ValueError(f"{label} mesh file does not exist: {path}")
    return path


def _mesh_scale_from_mapping(
    entry: dict[str, Any], label: str
) -> tuple[float, float, float]:
    keys = [key for key in ("scale", "mesh_scale") if key in entry]
    if len(keys) > 1:
        raise ValueError(f"{label} must use only one of scale or mesh_scale")
    if not keys:
        return (1.0, 1.0, 1.0)
    values = _positive_float_tuple(entry[keys[0]], 3, f"{label}.scale")
    return tuple(float(item) for item in values)


def _infer_mesh_size(
    mesh_path: Path, scale: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Read OBJ or STL vertices and return their scaled axis-aligned full extents."""

    try:
        if mesh_path.suffix.lower() == ".obj":
            vertices = _obj_vertices(mesh_path)
        else:
            vertices = _stl_vertices(mesh_path)
    except (OSError, UnicodeDecodeError, struct.error, ValueError) as exc:
        raise ValueError(
            f"could not infer target mesh bounding box for {mesh_path}: {exc}"
        ) from exc
    if vertices.size == 0:
        raise ValueError(
            f"could not infer target mesh bounding box for {mesh_path}: mesh has no vertices"
        )
    scaled = vertices * np.asarray(scale, dtype=float)[None, :]
    full_extents = np.maximum(np.max(scaled, axis=0) - np.min(scaled, axis=0), 1e-6)
    if not np.isfinite(full_extents).all() or np.any(full_extents <= 0.0):
        raise ValueError(
            f"could not infer a finite positive target mesh bounding box for {mesh_path}"
        )
    return tuple(float(component) for component in full_extents)


def _obj_vertices(path: Path) -> np.ndarray:
    vertices: list[tuple[float, float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        fields = line.strip().split()
        if fields and fields[0] == "v" and len(fields) >= 4:
            vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
    return np.asarray(vertices, dtype=float).reshape(-1, 3)


def _stl_vertices(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    if len(payload) >= 84:
        triangle_count = struct.unpack_from("<I", payload, 80)[0]
        expected = 84 + 50 * triangle_count
        if triangle_count > 0 and len(payload) == expected:
            vertices = np.empty((triangle_count * 3, 3), dtype=float)
            for triangle in range(triangle_count):
                offset = 84 + 50 * triangle + 12
                vertices[3 * triangle : 3 * triangle + 3] = np.asarray(
                    struct.unpack_from("<9f", payload, offset),
                    dtype=float,
                ).reshape(3, 3)
            return vertices
    text = payload.decode("utf-8")
    vertices = []
    for match in re.finditer(
        r"^\s*vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    ):
        vertices.append(tuple(float(match.group(index)) for index in (1, 2, 3)))
    return np.asarray(vertices, dtype=float).reshape(-1, 3)


def _register_target_tokens(tokens: dict[str, str], spec: TargetSpec) -> None:
    for token in (spec.name, *spec.aliases):
        normalized = _normalize_text(token)
        if not normalized:
            continue
        owner = tokens.get(normalized)
        if owner is not None:
            raise ValueError(
                f"duplicate target name or alias '{normalized}' used by '{owner}' and '{spec.name}'"
            )
        tokens[normalized] = spec.name


def resolve_target(
    name_or_prompt: str, extra_targets: dict[str, TargetSpec] | None = None
) -> TargetSpec:
    if not isinstance(name_or_prompt, str):
        raise ValueError("target name or prompt must be a string")
    text = _normalize_text(name_or_prompt)
    if not text:
        raise ValueError("target name or prompt must be non-empty")
    extra = extra_targets or {}
    for collection in (extra, TARGETS):
        exact = _exact_target_match(text, collection)
        if exact is not None:
            return exact
        phrase = _phrase_target_match(text, collection)
        if phrase is not None:
            return phrase
    available = ", ".join(sorted({*TARGETS, *extra}))
    raise ValueError(
        f"unknown target '{name_or_prompt}'; available targets: {available}"
    )


def _exact_target_match(text: str, targets: dict[str, TargetSpec]) -> TargetSpec | None:
    for key, spec in targets.items():
        if text == _normalize_text(key) or any(
            text == _normalize_text(alias) for alias in spec.aliases
        ):
            return spec
    return None


def _phrase_target_match(
    text: str, targets: dict[str, TargetSpec]
) -> TargetSpec | None:
    words = _word_tokens(text)
    matches: list[tuple[int, int, TargetSpec]] = []
    for key, spec in targets.items():
        for token in (key, spec.name, *spec.aliases):
            normalized = _normalize_text(token)
            token_words = _word_tokens(normalized)
            if token_words and _contains_word_phrase(words, token_words):
                matches.append((len(token_words), len(normalized), spec))
    if not matches:
        return None
    best_rank = max((word_count, char_count) for word_count, char_count, _ in matches)
    best_specs = {
        spec.name: spec
        for word_count, char_count, spec in matches
        if (word_count, char_count) == best_rank
    }
    if len(best_specs) > 1:
        raise ValueError(
            f"ambiguous target prompt '{text}': {', '.join(sorted(best_specs))}"
        )
    return next(iter(best_specs.values()))


def _contains_word_phrase(words: list[str], phrase: list[str]) -> bool:
    width = len(phrase)
    return any(
        words[index : index + width] == phrase
        for index in range(len(words) - width + 1)
    )


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().strip().split())


def _word_tokens(value: str) -> list[str]:
    return re.findall(r"[^\W_]+|_+", value.lower(), flags=re.UNICODE)


def base_position(target: TargetSpec) -> np.ndarray:
    if target.base_position is not None:
        return np.array(target.base_position, dtype=float)
    return BASE_POSITIONS.get(
        target.name, np.array([0.48, 0.02, 0.35], dtype=float)
    ).copy()


@dataclass
class TargetMotion:
    target: TargetSpec
    mode: str
    seed: int = 7
    base_override: np.ndarray | None = None

    def __post_init__(self) -> None:
        self._base = (
            np.array(self.base_override, dtype=float).reshape(3)
            if self.base_override is not None
            else base_position(self.target)
        )
        self._rng = np.random.default_rng(self.seed)
        self._random_velocity = np.array([0.035, -0.025, 0.018], dtype=float)
        self._random_pos = self._base.copy()
        self._last_time = 0.0
        self._waypoints = np.array(
            [
                self._base + np.array([0.00, 0.00, 0.00]),
                self._base + np.array([0.10, 0.05, 0.02]),
                self._base + np.array([0.04, -0.12, -0.015]),
                self._base + np.array([-0.08, -0.04, 0.025]),
            ],
            dtype=float,
        )

    def position(self, time_s: float) -> np.ndarray:
        mode = self.mode.strip().lower()
        base = self._base
        t = float(max(time_s, 0.0))
        phase = (sum(ord(ch) for ch in self.target.name) % 360) * np.pi / 180.0
        if mode == "static":
            return base
        if mode == "circle":
            return base + np.array(
                [
                    0.075 * np.cos(0.58 * t + phase),
                    0.055 * np.sin(0.58 * t + phase),
                    0.020 * np.sin(0.30 * t),
                ],
                dtype=float,
            )
        if mode == "figure-eight":
            return base + np.array(
                [
                    0.085 * np.sin(0.54 * t),
                    0.060 * np.sin(1.08 * t + 0.4),
                    0.024 * np.sin(0.38 * t + phase),
                ],
                dtype=float,
            )
        if mode == "random-walk":
            return self._random_walk(t)
        if mode == "waypoints":
            return self._waypoint_position(t)
        raise ValueError(f"unknown target trajectory '{self.mode}'")

    def _random_walk(self, time_s: float) -> np.ndarray:
        dt = max(0.0, min(0.05, time_s - self._last_time))
        self._last_time = time_s
        jitter = self._rng.normal(0.0, 0.06, size=3)
        jitter[2] *= 0.35
        self._random_velocity = 0.985 * self._random_velocity + 0.015 * jitter
        self._random_pos = self._random_pos + self._random_velocity * dt
        low = self._base + np.array([-0.13, -0.16, -0.05])
        high = self._base + np.array([0.13, 0.16, 0.06])
        for i in range(3):
            if self._random_pos[i] < low[i] or self._random_pos[i] > high[i]:
                self._random_velocity[i] *= -0.65
        self._random_pos = np.clip(self._random_pos, low, high)
        return self._random_pos.copy()

    def _waypoint_position(self, time_s: float) -> np.ndarray:
        segment_s = 2.4
        scaled = time_s / segment_s
        index = int(np.floor(scaled)) % len(self._waypoints)
        nxt = (index + 1) % len(self._waypoints)
        local = scaled - np.floor(scaled)
        blend = 0.5 - 0.5 * np.cos(np.pi * local)
        return (1.0 - blend) * self._waypoints[index] + blend * self._waypoints[nxt]
