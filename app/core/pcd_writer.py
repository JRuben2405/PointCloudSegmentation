from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class PCDWriteError(RuntimeError):
    """Raised when a point cloud cannot be written to disk."""


@dataclass(frozen=True)
class PCDWriteResult:
    path: Path
    point_count: int
    includes_intensity: bool


@dataclass(frozen=True)
class PCDPreviewData:
    path: Path
    xyz: np.ndarray
    intensity: np.ndarray | None
    original_point_count: int
    rendered_point_count: int
    has_intensity: bool
    centroid: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray


def build_pcd_path(export_dir: str | Path, index: int) -> Path:
    return Path(export_dir) / f"{index:06d}.pcd"


def write_pcd(
    output_path: str | Path,
    xyz: np.ndarray,
    intensity: np.ndarray | None = None,
) -> PCDWriteResult:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    xyz_array = np.asarray(xyz, dtype=np.float32)
    if xyz_array.ndim != 2 or xyz_array.shape[1] != 3:
        raise PCDWriteError("La nube de puntos debe tener forma (N, 3).")

    intensity_array = None
    if intensity is not None:
        intensity_array = np.asarray(intensity, dtype=np.float32).reshape(-1)
        if intensity_array.shape[0] != xyz_array.shape[0]:
            raise PCDWriteError("La intensidad debe tener el mismo numero de elementos que xyz.")

    o3d = _load_open3d()

    try:
        if intensity_array is not None:
            _write_tensor_pointcloud(o3d, path, xyz_array, intensity_array)
        else:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(xyz_array.astype(np.float64, copy=False))
            ok = o3d.io.write_point_cloud(
                str(path),
                cloud,
                write_ascii=False,
                compressed=False,
                print_progress=False,
            )
            if not ok:
                raise PCDWriteError(f"Open3D no pudo escribir el archivo {path.name}.")
    except PCDWriteError:
        raise
    except Exception as exc:
        raise PCDWriteError(f"No se pudo escribir {path.name}: {exc}") from exc

    return PCDWriteResult(
        path=path,
        point_count=int(xyz_array.shape[0]),
        includes_intensity=intensity_array is not None,
    )


def load_pcd_preview_data(
    pcd_path: str | Path,
    *,
    max_points: int | None = 30_000,
) -> PCDPreviewData:
    path = Path(pcd_path)
    if not path.exists():
        raise PCDWriteError(f"El archivo PCD no existe: {path}")
    if not path.is_file():
        raise PCDWriteError(f"La ruta seleccionada no es un archivo: {path}")
    if path.suffix.lower() != ".pcd":
        raise PCDWriteError(f"El archivo seleccionado no es un .pcd: {path.name}")

    o3d = _load_open3d()
    xyz, intensity = _read_pcd_arrays(o3d, path)
    if xyz.size == 0:
        return PCDPreviewData(
            path=path,
            xyz=xyz,
            intensity=intensity,
            original_point_count=0,
            rendered_point_count=0,
            has_intensity=intensity is not None,
            centroid=np.zeros(3, dtype=np.float32),
            bounds_min=np.zeros(3, dtype=np.float32),
            bounds_max=np.zeros(3, dtype=np.float32),
        )

    sampled_xyz, sampled_intensity = _sample_points(xyz, intensity, max_points=max_points)
    return PCDPreviewData(
        path=path,
        xyz=sampled_xyz,
        intensity=sampled_intensity,
        original_point_count=int(xyz.shape[0]),
        rendered_point_count=int(sampled_xyz.shape[0]),
        has_intensity=intensity is not None,
        centroid=np.mean(xyz, axis=0),
        bounds_min=np.min(xyz, axis=0),
        bounds_max=np.max(xyz, axis=0),
    )


def _write_tensor_pointcloud(
    o3d: object,
    path: Path,
    xyz: np.ndarray,
    intensity: np.ndarray,
) -> None:
    if not hasattr(o3d, "t") or not hasattr(o3d.t, "geometry") or not hasattr(o3d.t, "io"):
        raise PCDWriteError(
            "La version instalada de Open3D no soporta escritura tensorial con intensidad."
        )

    dtype = o3d.core.Dtype.Float32
    intensity_tensor = o3d.core.Tensor(intensity.reshape(-1, 1), dtype=dtype)

    write_errors: list[str] = []
    for attribute_name in ("intensity", "intensities"):
        point_cloud = o3d.t.geometry.PointCloud()
        point_cloud.point["positions"] = o3d.core.Tensor(xyz, dtype=dtype)
        point_cloud.point[attribute_name] = intensity_tensor
        try:
            ok = o3d.t.io.write_point_cloud(str(path), point_cloud)
            if ok:
                return
        except Exception as exc:
            write_errors.append(f"{attribute_name}: {exc}")
        else:
            write_errors.append(f"{attribute_name}: Open3D devolvio False")

    raise PCDWriteError(
        "No se pudo escribir la intensidad en el PCD. "
        + " | ".join(write_errors)
    )


def _load_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise PCDWriteError(
            "Open3D no esta instalado. Instale las dependencias con requirements.txt."
        ) from exc
    return o3d


def _read_pcd_arrays(o3d: object, path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    if hasattr(o3d, "t") and hasattr(o3d.t, "io"):
        point_cloud = o3d.t.io.read_point_cloud(str(path))
        xyz = _tensor_to_numpy(point_cloud.point["positions"])
        intensity = _extract_optional_intensity(point_cloud)
        return _ensure_float32_xyz(xyz), _ensure_float32_optional_vector(intensity)

    point_cloud = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(point_cloud.points, dtype=np.float32)
    return _ensure_float32_xyz(xyz), None


def _extract_optional_intensity(point_cloud: object) -> np.ndarray | None:
    for attribute_name in ("intensity", "intensities"):
        try:
            values = point_cloud.point[attribute_name]
        except Exception:
            continue
        return _tensor_to_numpy(values).reshape(-1)
    return None


def _tensor_to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _ensure_float32_xyz(xyz: np.ndarray) -> np.ndarray:
    xyz_array = np.asarray(xyz, dtype=np.float32)
    if xyz_array.ndim != 2 or xyz_array.shape[1] != 3:
        raise PCDWriteError("El archivo PCD no contiene puntos con forma (N, 3).")
    return xyz_array


def _ensure_float32_optional_vector(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    return np.asarray(values, dtype=np.float32).reshape(-1)


def _sample_points(
    xyz: np.ndarray,
    intensity: np.ndarray | None,
    *,
    max_points: int | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if max_points is None or max_points <= 0 or xyz.shape[0] <= max_points:
        return xyz, intensity

    rng = np.random.default_rng(42)
    indices = np.sort(rng.choice(xyz.shape[0], size=max_points, replace=False))
    sampled_xyz = xyz[indices]
    sampled_intensity = intensity[indices] if intensity is not None else None
    return sampled_xyz, sampled_intensity
