from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def sanitize_export_name(name: str) -> str:
    cleaned = re.sub(r"[\\/]+", "_", name.strip())
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned)
    return cleaned.strip("._")


def ensure_export_directory(output_root: str | Path, export_name: str) -> Path:
    raw_output = str(output_root).strip()
    if not raw_output:
        raise ValueError("La carpeta de salida no puede estar vacia.")
    root_path = Path(raw_output).expanduser()

    safe_name = sanitize_export_name(export_name)
    if not safe_name:
        raise ValueError("El nombre de la subcarpeta de exportacion es invalido.")

    if root_path.exists() and not root_path.is_dir():
        raise ValueError(f"La ruta de salida no es un directorio: {root_path}")
    root_path.mkdir(parents=True, exist_ok=True)

    export_dir = root_path / safe_name
    if export_dir.exists():
        if not export_dir.is_dir():
            raise ValueError(f"La ruta de exportacion no es un directorio: {export_dir}")
        if any(export_dir.iterdir()):
            raise ValueError(
                f"La carpeta de exportacion ya existe y no esta vacia: {export_dir}"
            )
    else:
        export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


def ns_to_iso8601(timestamp_ns: int | None) -> str | None:
    if timestamp_ns is None:
        return None
    seconds, nanoseconds = divmod(int(timestamp_ns), 1_000_000_000)
    dt = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return f"{dt.strftime('%Y-%m-%dT%H:%M:%S')}.{nanoseconds:09d}Z"


def timestamp_payload(timestamp_ns: int | None) -> dict[str, object] | None:
    if timestamp_ns is None:
        return None
    return {
        "ns": int(timestamp_ns),
        "iso_utc": ns_to_iso8601(timestamp_ns),
    }


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    output_path = Path(path)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def validate_export_directory(export_dir: str | Path, sample_size: int = 3) -> dict[str, object]:
    path = Path(export_dir).expanduser()
    if not path.exists():
        raise ValueError(f"La carpeta no existe: {path}")
    if not path.is_dir():
        raise ValueError(f"La ruta no es un directorio: {path}")

    pcd_files = sorted(path.glob("*.pcd"))
    metadata_path = path / "metadata.json"

    result: dict[str, object] = {
        "export_dir": str(path.resolve()),
        "metadata_exists": metadata_path.exists(),
        "pcd_count": len(pcd_files),
        "valid": False,
        "sample_files": [],
    }

    if not pcd_files:
        return result

    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "Open3D no esta instalado. No se puede validar la exportacion."
        ) from exc

    sample_files = []
    for file_path in pcd_files[: max(sample_size, 1)]:
        point_count = _read_point_count(o3d, file_path)
        sample_files.append({"file": file_path.name, "points": point_count})

    result["sample_files"] = sample_files
    result["valid"] = True
    return result


def discover_pcd_directories(output_root: str | Path) -> list[Path]:
    root = Path(output_root).expanduser()
    if not str(root).strip():
        return []
    if not root.exists() or not root.is_dir():
        return []

    candidates: set[Path] = set()
    resolved_root = root.resolve()
    if any(resolved_root.glob("*.pcd")):
        candidates.add(resolved_root)

    for child in resolved_root.iterdir():
        if child.is_dir() and any(child.glob("*.pcd")):
            candidates.add(child.resolve())

    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def _read_point_count(o3d: object, file_path: Path) -> int:
    if hasattr(o3d, "t") and hasattr(o3d.t, "io"):
        point_cloud = o3d.t.io.read_point_cloud(str(file_path))
        return int(point_cloud.point["positions"].shape[0])
    legacy_cloud = o3d.io.read_point_cloud(str(file_path))
    return int(len(legacy_cloud.points))


def _main() -> None:
    parser = argparse.ArgumentParser(description="Valida una carpeta de PCD exportados.")
    parser.add_argument("export_dir", help="Carpeta que contiene los .pcd y metadata.json")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3,
        help="Cantidad de archivos .pcd a leer para validar.",
    )
    args = parser.parse_args()
    result = validate_export_directory(args.export_dir, sample_size=args.sample_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
