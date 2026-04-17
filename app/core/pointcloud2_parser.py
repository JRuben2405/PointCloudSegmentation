from __future__ import annotations

from dataclasses import dataclass

import numpy as np

POINTFIELD_DTYPES: dict[int, np.dtype] = {
    1: np.dtype("i1"),
    2: np.dtype("u1"),
    3: np.dtype("i2"),
    4: np.dtype("u2"),
    5: np.dtype("i4"),
    6: np.dtype("u4"),
    7: np.dtype("f4"),
    8: np.dtype("f8"),
}


class PointCloud2ParseError(RuntimeError):
    """Raised when a PointCloud2 message cannot be converted safely."""


@dataclass(frozen=True)
class ParsedPointCloudFrame:
    xyz: np.ndarray
    intensity: np.ndarray | None
    field_names: list[str]
    point_count_raw: int
    point_count_valid: int
    has_intensity_field: bool
    frame_id: str | None
    header_stamp_ns: int | None
    width: int
    height: int
    point_step: int

    @property
    def frame_metadata(self) -> dict[str, object]:
        return {
            "field_names": self.field_names,
            "point_count_raw": self.point_count_raw,
            "point_count_valid": self.point_count_valid,
            "has_intensity_field": self.has_intensity_field,
            "frame_id": self.frame_id,
            "header_stamp_ns": self.header_stamp_ns,
            "width": self.width,
            "height": self.height,
            "point_step": self.point_step,
        }


def parse_pointcloud2(
    message: object,
    *,
    include_intensity: bool = True,
    discard_invalid: bool = True,
) -> ParsedPointCloudFrame:
    fields = list(getattr(message, "fields", []))
    field_names = [field.name for field in fields]
    fields_by_name = {field.name.lower(): field for field in fields}

    missing = [name for name in ("x", "y", "z") if name not in fields_by_name]
    if missing:
        missing_text = ", ".join(missing)
        raise PointCloud2ParseError(
            f"El mensaje PointCloud2 no contiene los campos requeridos: {missing_text}."
        )

    for axis in ("x", "y", "z"):
        _validate_scalar_field(fields_by_name[axis], axis)

    intensity_field = fields_by_name.get("intensity")
    if include_intensity and intensity_field is not None:
        _validate_scalar_field(intensity_field, "intensity")

    width = int(getattr(message, "width", 0))
    height = int(getattr(message, "height", 0))
    point_step = int(getattr(message, "point_step", 0))
    row_step = int(getattr(message, "row_step", 0))
    raw_point_count = width * height

    if point_step <= 0:
        raise PointCloud2ParseError("El mensaje PointCloud2 tiene point_step invalido.")
    if height < 0 or width < 0:
        raise PointCloud2ParseError("El mensaje PointCloud2 tiene width/height invalidos.")
    if raw_point_count == 0:
        return ParsedPointCloudFrame(
            xyz=np.empty((0, 3), dtype=np.float32),
            intensity=np.empty((0,), dtype=np.float32) if include_intensity and intensity_field else None,
            field_names=field_names,
            point_count_raw=0,
            point_count_valid=0,
            has_intensity_field=intensity_field is not None,
            frame_id=getattr(getattr(message, "header", None), "frame_id", None),
            header_stamp_ns=_header_stamp_to_ns(message),
            width=width,
            height=height,
            point_step=point_step,
        )

    if row_step < point_step * width:
        raise PointCloud2ParseError(
            "El mensaje PointCloud2 tiene row_step menor que width * point_step."
        )

    buffer = _as_uint8_buffer(getattr(message, "data", b""))
    required_size = row_step * height
    if buffer.size < required_size:
        raise PointCloud2ParseError(
            "El bloque binario del mensaje PointCloud2 es mas pequeno que row_step * height."
        )

    endian = ">" if bool(getattr(message, "is_bigendian", False)) else "<"
    selected_fields = [fields_by_name["x"], fields_by_name["y"], fields_by_name["z"]]
    if include_intensity and intensity_field is not None:
        selected_fields.append(intensity_field)

    dtype = np.dtype(
        {
            "names": [field.name.lower() for field in selected_fields],
            "formats": [_field_numpy_dtype(field, endian) for field in selected_fields],
            "offsets": [int(field.offset) for field in selected_fields],
            "itemsize": point_step,
        }
    )

    try:
        structured = np.ndarray(
            shape=(height, width),
            dtype=dtype,
            buffer=buffer,
            strides=(row_step, point_step),
        )
    except Exception as exc:
        raise PointCloud2ParseError(f"No se pudo interpretar el buffer PointCloud2: {exc}") from exc

    flat = structured.reshape(-1)
    xyz = np.stack(
        [
            np.asarray(flat["x"], dtype=np.float32),
            np.asarray(flat["y"], dtype=np.float32),
            np.asarray(flat["z"], dtype=np.float32),
        ],
        axis=1,
    )

    intensity = None
    if include_intensity and intensity_field is not None:
        intensity = np.asarray(flat["intensity"], dtype=np.float32).reshape(-1)

    if discard_invalid:
        valid_mask = np.isfinite(xyz).all(axis=1)
        if intensity is not None:
            valid_mask &= np.isfinite(intensity)
        xyz = xyz[valid_mask]
        if intensity is not None:
            intensity = intensity[valid_mask]

    return ParsedPointCloudFrame(
        xyz=xyz,
        intensity=intensity,
        field_names=field_names,
        point_count_raw=raw_point_count,
        point_count_valid=int(xyz.shape[0]),
        has_intensity_field=intensity_field is not None,
        frame_id=getattr(getattr(message, "header", None), "frame_id", None),
        header_stamp_ns=_header_stamp_to_ns(message),
        width=width,
        height=height,
        point_step=point_step,
    )


def _as_uint8_buffer(data: object) -> np.ndarray:
    if isinstance(data, np.ndarray):
        array = np.asarray(data, dtype=np.uint8).reshape(-1)
        return np.ascontiguousarray(array)
    return np.frombuffer(data, dtype=np.uint8)


def _field_numpy_dtype(field: object, endian: str) -> np.dtype:
    datatype = int(getattr(field, "datatype", -1))
    if datatype not in POINTFIELD_DTYPES:
        raise PointCloud2ParseError(
            f"Datatype PointField no soportado en el campo {field.name}: {datatype}."
        )
    return POINTFIELD_DTYPES[datatype].newbyteorder(endian)


def _validate_scalar_field(field: object, name: str) -> None:
    count = int(getattr(field, "count", 1))
    if count != 1:
        raise PointCloud2ParseError(
            f'El campo "{name}" debe tener count=1 y se encontro count={count}.'
        )


def _header_stamp_to_ns(message: object) -> int | None:
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    if stamp is None:
        return None
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", None)
    if sec is None or nanosec is None:
        return None
    return int(sec) * 1_000_000_000 + int(nanosec)
