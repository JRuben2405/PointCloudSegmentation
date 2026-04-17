from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


class SegmentationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SensorConfig:
    width: int
    height: int
    fov_up: float
    fov_down: float
    img_means: list[float]
    img_stds: list[float]


@dataclass(frozen=True)
class SegmentationConfig:
    sensor: SensorConfig
    nclasses: int
    learning_map_inv: dict[int, int]
    color_map: dict[int, list[int]]
    labels: dict[int, str]

    @property
    def train_id_labels(self) -> dict[int, str]:
        return {
            tid: self.labels.get(raw, f"class_{tid}")
            for tid, raw in self.learning_map_inv.items()
        }

    @property
    def train_id_colors(self) -> dict[int, list[int]]:
        return {
            tid: self.color_map.get(raw, [160, 160, 160])
            for tid, raw in self.learning_map_inv.items()
        }


@dataclass(frozen=True)
class SegmentationResult:
    labels: np.ndarray
    num_points: int
    class_counts: dict[int, int]
    config: SegmentationConfig


def load_configs(model_dir: str | Path) -> SegmentationConfig:
    model_dir = Path(model_dir)
    arch_path = model_dir / "arch_cfg.yaml"
    data_path = model_dir / "data_cfg.yaml"

    if not arch_path.exists():
        raise SegmentationError(f"No se encontro arch_cfg.yaml en {model_dir}")
    if not data_path.exists():
        raise SegmentationError(f"No se encontro data_cfg.yaml en {model_dir}")

    with open(arch_path) as f:
        arch = yaml.safe_load(f)
    with open(data_path) as f:
        data = yaml.safe_load(f)

    sensor_cfg = arch["dataset"]["sensor"]
    sensor = SensorConfig(
        width=sensor_cfg["img_prop"]["width"],
        height=sensor_cfg["img_prop"]["height"],
        fov_up=sensor_cfg["fov_up"],
        fov_down=sensor_cfg["fov_down"],
        img_means=arch["dataset"]["sensor"]["img_means"],
        img_stds=arch["dataset"]["sensor"]["img_stds"],
    )

    return SegmentationConfig(
        sensor=sensor,
        nclasses=arch["head"]["nclasses"],
        learning_map_inv={int(k): int(v) for k, v in data["learning_map_inv"].items()},
        color_map={int(k): v for k, v in data["color_map"].items()},
        labels={int(k): str(v) for k, v in data["labels"].items()},
    )


def spherical_projection(
    xyz: np.ndarray,
    intensity: np.ndarray | None,
    sensor: SensorConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Matches LaserScan.do_range_projection() from training notebook exactly."""
    fov_up = sensor.fov_up / 180.0 * np.pi
    fov_down = sensor.fov_down / 180.0 * np.pi
    fov = abs(fov_down) + abs(fov_up)

    points = xyz.astype(np.float32)
    if intensity is not None:
        remissions = intensity.astype(np.float32).reshape(-1)
        intensity_max = remissions.max()
        if intensity_max > 1.0:
            remissions = remissions / intensity_max
    else:
        remissions = np.zeros(points.shape[0], dtype=np.float32)

    depth = np.linalg.norm(points, 2, axis=1)
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]

    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(np.clip(scan_z / (depth + 1e-8), -1, 1))

    proj_x = 0.5 * (yaw / np.pi + 1.0) * sensor.width
    proj_y = (1.0 - (pitch + abs(fov_down)) / fov) * sensor.height

    proj_x = np.floor(proj_x).clip(0, sensor.width - 1).astype(np.int32)
    proj_y = np.floor(proj_y).clip(0, sensor.height - 1).astype(np.int32)

    proj_range = np.full((sensor.height, sensor.width), -1, dtype=np.float32)
    proj_xyz = np.full((sensor.height, sensor.width, 3), -1, dtype=np.float32)
    proj_remission = np.full((sensor.height, sensor.width), -1, dtype=np.float32)
    proj_idx = np.full((sensor.height, sensor.width), -1, dtype=np.int32)

    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = np.arange(len(depth))[order]
    points = points[order]
    remissions_sorted = remissions[order]
    proj_y_sorted = proj_y[order]
    proj_x_sorted = proj_x[order]

    proj_range[proj_y_sorted, proj_x_sorted] = depth
    proj_xyz[proj_y_sorted, proj_x_sorted] = points
    proj_remission[proj_y_sorted, proj_x_sorted] = remissions_sorted
    proj_idx[proj_y_sorted, proj_x_sorted] = indices
    proj_mask = (proj_idx >= 0).astype(np.float32)

    # Build 5-channel tensor: [range, x, y, z, remission]
    proj = np.stack([
        proj_range,
        proj_xyz[:, :, 0],
        proj_xyz[:, :, 1],
        proj_xyz[:, :, 2],
        proj_remission,
    ], axis=0).astype(np.float32)  # [5, H, W]

    means = np.array(sensor.img_means, dtype=np.float32).reshape(5, 1, 1)
    stds = np.array(sensor.img_stds, dtype=np.float32).reshape(5, 1, 1)
    proj = (proj - means) / stds
    proj = proj * proj_mask[np.newaxis, :, :]

    range_tensor = proj[np.newaxis]  # [1, 5, H, W]
    return range_tensor, proj_idx


def unproject_labels(
    predictions: np.ndarray,
    proj_idx: np.ndarray,
    num_points: int,
) -> np.ndarray:
    # predictions: [1, nclasses, H, W] -> argmax -> [H, W]
    pred_labels = np.argmax(predictions[0], axis=0).astype(np.int32)
    point_labels = np.zeros(num_points, dtype=np.int32)

    valid_mask = proj_idx >= 0
    point_indices = proj_idx[valid_mask]
    pixel_labels = pred_labels[valid_mask]
    point_labels[point_indices] = pixel_labels

    return point_labels


def run_onnx_segmentation(
    model_path: str | Path,
    range_image: np.ndarray,
) -> np.ndarray:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SegmentationError(
            "onnxruntime no esta instalado. Ejecute: pip install onnxruntime"
        ) from exc

    model_path = str(model_path)
    try:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise SegmentationError(f"No se pudo cargar el modelo ONNX: {exc}") from exc

    input_name = session.get_inputs()[0].name
    try:
        outputs = session.run(None, {input_name: range_image})
    except Exception as exc:
        raise SegmentationError(f"Error durante la inferencia: {exc}") from exc

    return outputs[0]


def segment_pcd(
    xyz: np.ndarray,
    intensity: np.ndarray | None,
    model_path: str | Path,
    config: SegmentationConfig,
) -> SegmentationResult:
    range_image, proj_idx = spherical_projection(
        xyz, intensity, config.sensor,
    )

    predictions = run_onnx_segmentation(model_path, range_image)
    labels = unproject_labels(predictions, proj_idx, xyz.shape[0])

    unique, counts = np.unique(labels, return_counts=True)
    class_counts = {int(u): int(c) for u, c in zip(unique, counts)}

    return SegmentationResult(
        labels=labels,
        num_points=xyz.shape[0],
        class_counts=class_counts,
        config=config,
    )
