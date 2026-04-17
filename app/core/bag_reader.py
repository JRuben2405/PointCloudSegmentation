from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from rosbags.highlevel import AnyReader
from rosbags.interfaces import Connection, TopicInfo
from rosbags.typesys import Stores, get_typestore

POINTCLOUD2_TYPE_ALIASES = {
    "sensor_msgs/PointCloud2",
    "sensor_msgs/msg/PointCloud2",
}


class BagReaderError(RuntimeError):
    """Raised when a bag cannot be opened or interpreted."""


@dataclass(frozen=True)
class BagSource:
    original_path: Path
    reader_path: Path
    bag_type: str

    def to_dict(self) -> dict[str, str]:
        return {
            "original_path": str(self.original_path),
            "reader_path": str(self.reader_path),
            "bag_type": self.bag_type,
        }


@dataclass(frozen=True)
class TopicDetails:
    topic: str
    msgtype: str | None
    message_count: int
    connection_count: int


@dataclass(frozen=True)
class ScanResult:
    source: BagSource
    topics: list[TopicDetails]
    pointcloud_topics: list[TopicDetails]


@dataclass(frozen=True)
class DecodedBagMessage:
    connection: Connection
    timestamp_ns: int
    message: object


def normalize_msgtype(msgtype: str | None) -> str | None:
    if msgtype == "sensor_msgs/PointCloud2":
        return "sensor_msgs/msg/PointCloud2"
    return msgtype


def is_pointcloud2_msgtype(msgtype: str | None) -> bool:
    return normalize_msgtype(msgtype) in POINTCLOUD2_TYPE_ALIASES


def resolve_bag_source(input_path: str | Path) -> BagSource:
    raw_input = str(input_path).strip()
    if not raw_input:
        raise BagReaderError("Seleccione una ruta de entrada antes de escanear.")
    path = Path(raw_input).expanduser()
    if not path.exists():
        raise BagReaderError(f"La ruta no existe: {path}")

    resolved = path.resolve()
    if resolved.is_file():
        if resolved.suffix == ".bag":
            return BagSource(original_path=resolved, reader_path=resolved, bag_type="rosbag1")
        if resolved.suffix == ".db3":
            reader_path = resolved.parent.resolve()
            if not _looks_like_rosbag2_dir(reader_path):
                raise BagReaderError(
                    "El archivo .db3 no pertenece a un directorio rosbag2 valido."
                )
            return BagSource(original_path=resolved, reader_path=reader_path, bag_type="rosbag2")
        raise BagReaderError(
            "Entrada no soportada. Use un archivo .bag de ROS1 o un directorio rosbag2."
        )

    if resolved.is_dir():
        if _looks_like_rosbag2_dir(resolved):
            return BagSource(original_path=resolved, reader_path=resolved, bag_type="rosbag2")
        raise BagReaderError(
            "El directorio no parece ser un rosbag2 valido. Debe contener metadata.yaml o archivos .db3."
        )

    raise BagReaderError(f"No se pudo interpretar la ruta de entrada: {resolved}")


def scan_bag(input_path: str | Path) -> ScanResult:
    source = resolve_bag_source(input_path)
    with open_reader(source) as reader:
        topics = [_topic_details_from_info(name, info) for name, info in sorted(reader.topics.items())]
    pointcloud_topics = [topic for topic in topics if is_pointcloud2_msgtype(topic.msgtype)]
    return ScanResult(source=source, topics=topics, pointcloud_topics=pointcloud_topics)


def iter_topic_messages(source: BagSource | str | Path, topic: str) -> Iterator[DecodedBagMessage]:
    source_obj = source if isinstance(source, BagSource) else resolve_bag_source(source)
    reader = _create_reader(source_obj)
    try:
        reader.open()
        connections = [conn for conn in reader.connections if conn.topic == topic]
        if not connections:
            raise BagReaderError(f'No se encontro el topico "{topic}" en el bag.')
        for connection, timestamp_ns, rawdata in reader.messages(connections=connections):
            yield DecodedBagMessage(
                connection=connection,
                timestamp_ns=timestamp_ns,
                message=reader.deserialize(rawdata, connection.msgtype),
            )
    except BagReaderError:
        raise
    except Exception as exc:
        raise BagReaderError(f"No se pudieron leer los mensajes del topico {topic}: {exc}") from exc
    finally:
        if reader.isopen:
            reader.close()


@contextmanager
def open_reader(source: BagSource | str | Path):
    source_obj = source if isinstance(source, BagSource) else resolve_bag_source(source)
    reader = _create_reader(source_obj)
    try:
        reader.open()
        yield reader
    except BagReaderError:
        raise
    except Exception as exc:
        raise BagReaderError(f"No se pudo abrir el bag: {exc}") from exc
    finally:
        if reader.isopen:
            reader.close()


def _create_reader(source: BagSource) -> AnyReader:
    try:
        return AnyReader(
            [source.reader_path],
            default_typestore=_default_typestore(source.bag_type),
        )
    except Exception as exc:
        raise BagReaderError(f"No se pudo crear el reader para {source.reader_path}: {exc}") from exc


def _default_typestore(bag_type: str):
    store = Stores.ROS1_NOETIC if bag_type == "rosbag1" else Stores.LATEST
    return get_typestore(store)


def _looks_like_rosbag2_dir(path: Path) -> bool:
    return path.is_dir() and ((path / "metadata.yaml").exists() or any(path.glob("*.db3")))


def _topic_details_from_info(topic_name: str, info: TopicInfo) -> TopicDetails:
    msgtype = normalize_msgtype(info.msgtype)
    if msgtype is None:
        connection_types = {normalize_msgtype(connection.msgtype) for connection in info.connections}
        if len(connection_types) == 1:
            msgtype = next(iter(connection_types))
    return TopicDetails(
        topic=topic_name,
        msgtype=msgtype,
        message_count=info.msgcount,
        connection_count=len(info.connections),
    )
