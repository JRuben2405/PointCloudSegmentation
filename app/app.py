from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.offline import get_plotlyjs

try:
    from core.bag_reader import BagReaderError, ScanResult, iter_topic_messages, scan_bag
    from core.pcd_writer import (
        PCDPreviewData,
        PCDWriteError,
        build_pcd_path,
        load_pcd_preview_data,
        write_pcd,
    )
    from core.pointcloud2_parser import PointCloud2ParseError, parse_pointcloud2
    from core.segmentation import (
        SegmentationConfig,
        SegmentationError,
        SegmentationResult,
        load_configs,
        segment_pcd,
    )
    from core.utils import (
        discover_pcd_directories,
        ensure_export_directory,
        timestamp_payload,
        validate_export_directory,
        write_json,
    )
except ModuleNotFoundError:
    from app.core.bag_reader import BagReaderError, ScanResult, iter_topic_messages, scan_bag
    from app.core.pcd_writer import (
        PCDPreviewData,
        PCDWriteError,
        build_pcd_path,
        load_pcd_preview_data,
        write_pcd,
    )
    from app.core.pointcloud2_parser import PointCloud2ParseError, parse_pointcloud2
    from app.core.segmentation import (
        SegmentationConfig,
        SegmentationError,
        SegmentationResult,
        load_configs,
        segment_pcd,
    )
    from app.core.utils import (
        discover_pcd_directories,
        ensure_export_directory,
        timestamp_payload,
        validate_export_directory,
        write_json,
    )


@dataclass(frozen=True)
class OutputBrowserState:
    root_raw: str
    root: Path | None
    export_dirs: list[Path]
    selected_export_dir: Path | None


@dataclass(frozen=True)
class ExportDirectoryDetails:
    export_dir: Path
    pcd_files: list[Path]
    metadata_path: Path | None
    metadata: dict[str, Any] | None
    metadata_error: str | None
    last_modified_label: str


def main() -> None:
    st.set_page_config(
        page_title="ForestSeg",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_state()
    _inject_styles()
    output_state = _build_output_browser_state()
    _render_sidebar(output_state)
    _render_global_messages()

    page = st.session_state.get("active_page", "Convertir")
    if page == "Convertir":
        _render_convert_page(output_state)
    elif page == "Segmentar":
        _render_segment_page(output_state)
    elif page == "Explorar PCD":
        _render_explore_page(output_state)
    elif page == "Metadata":
        _render_metadata_page(output_state)


def _init_state() -> None:
    defaults = {
        "active_page": "Convertir",
        "input_mode": "Autodetectar",
        "input_path": "",
        "output_root": str(Path.cwd() / "output"),
        "export_name": "exportacion_lidar",
        "save_intensity": True,
        "skip_empty_frames": True,
        "extraction_mode": "Todos los frames",
        "max_frames": 100,
        "specific_frame": 0,
        "scan_result": None,
        "scan_error": None,
        "scan_success": None,
        "preview_cache": {},
        "selected_topic": None,
        "last_conversion": None,
        "selected_pcd_export_dir": None,
        "selected_pcd_preview_file": None,
        "pcd_preview_use_downsampling": False,
        "pcd_preview_max_points": 30000,
        "pcd_preview_color_mode": "Altura Z",
        "pcd_preview_point_size": 2.0,
        "pcd_preview_background": "Oscuro",
        "pcd_preview_show_bounds": True,
        "pcd_preview_chart_nonce": 0,
        "metadata_show_only_real_errors": False,
        "validation_cache": {},
        "seg_model_dir": str(Path(__file__).parent / "models"),
        "seg_last_result": None,
        "seg_preview_point_size": 2.0,
        "seg_preview_background": "Oscuro",
        "seg_preview_show_bounds": False,
        "seg_preview_max_points": 50000,
        "seg_preview_use_downsampling": False,
        "seg_chart_nonce": 0,
        "seg_class_color_signature": None,
        "seg_class_visibility_signature": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2.5rem;
            padding-bottom: 2.2rem;
        }

        /* ── Stat cards ── */
        .stat-card {
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 10px;
            padding: 0.9rem 1rem;
            text-align: center;
            transition: box-shadow 0.2s;
        }
        .stat-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        .stat-card .label {
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.15rem;
        }
        .stat-card .value {
            font-size: 1.45rem;
            font-weight: 700;
            margin: 0.1rem 0;
            line-height: 1.2;
        }
        .stat-card .detail {
            font-size: 0.72rem;
            opacity: 0.65;
        }

        .stat-card.tone-ready  { border-left: 4px solid #22c55e; }
        .stat-card.tone-active { border-left: 4px solid #3b82f6; }
        .stat-card.tone-warn   { border-left: 4px solid #f59e0b; }
        .stat-card.tone-pending { border-left: 4px solid #94a3b8; }

        .stat-card.tone-ready  .value { color: #16a34a; }
        .stat-card.tone-active .value { color: #2563eb; }
        .stat-card.tone-warn   .value { color: #d97706; }
        .stat-card.tone-pending .value { color: #64748b; }

        /* ── Stepper ── */
        .step-header {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin: 1.4rem 0 0.5rem 0;
        }
        .step-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.7rem;
            height: 1.7rem;
            border-radius: 50%;
            background: #3b82f6;
            color: white;
            font-weight: 700;
            font-size: 0.85rem;
            flex-shrink: 0;
        }
        .step-badge.done { background: #22c55e; }
        .step-badge.disabled { background: #94a3b8; }
        .step-title {
            font-size: 1.05rem;
            font-weight: 600;
        }

        /* ── Segmentation class rows ── */
        .seg-class-row {
            min-height: 3.15rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .seg-class-row.inactive {
            opacity: 0.48;
        }
        .seg-class-title {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            font-weight: 600;
            line-height: 1.15;
        }
        .seg-class-dot {
            display: inline-block;
            width: 0.8rem;
            height: 0.8rem;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.16);
            flex-shrink: 0;
        }
        .seg-class-meta {
            margin-left: 1.35rem;
            margin-top: 0.12rem;
            font-size: 0.82rem;
            opacity: 0.72;
        }
        .seg-class-divider {
            height: 1px;
            background: rgba(128, 128, 128, 0.14);
            margin: 0.28rem 0 0.45rem 0;
        }
        [data-testid="stColorPicker"] {
            transform: scale(0.82);
            transform-origin: center center;
        }

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
        }
        section[data-testid="stSidebar"] hr {
            margin: 0.8rem 0;
        }

        /* ── Section dividers ── */
        .section-divider {
            border: none;
            border-top: 1px solid rgba(128,128,128,0.15);
            margin: 1.2rem 0;
        }

        /* ── Sidebar nav separator before tools ── */
        section[data-testid="stSidebar"] .nav-section-label {
            font-size: 0.65rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            opacity: 0.5;
            margin: 0.6rem 0 0.2rem 0;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(output_state: OutputBrowserState) -> None:
    with st.sidebar:
        st.markdown("### :material/forest: ForestSeg")
        st.caption("Segmentacion semantica de nubes de puntos LiDAR")
        st.divider()

        all_pages = ["Convertir", "Segmentar", "Explorar PCD", "Metadata"]
        current = st.session_state.get("active_page", "Convertir")
        if current not in all_pages:
            current = "Convertir"

        def _on_nav_change():
            st.session_state["active_page"] = st.session_state["_nav_radio"]

        main_pages = ["Convertir", "Segmentar"]
        radio_index = main_pages.index(current) if current in main_pages else None

        st.radio(
            "Navegacion",
            options=main_pages,
            index=radio_index,
            key="_nav_radio",
            label_visibility="collapsed",
            captions=["Rosbag → archivos .pcd", "Inferencia semantica"],
            on_change=_on_nav_change,
        )

        st.markdown('<p class="nav-section-label">Herramientas</p>', unsafe_allow_html=True)
        tool_col1, tool_col2 = st.columns(2)
        tool_col1.button(
            "Explorar",
            icon=":material/view_in_ar:",
            key="nav_explore",
            use_container_width=True,
            on_click=lambda: st.session_state.update(active_page="Explorar PCD"),
        )
        tool_col2.button(
            "Metadata",
            icon=":material/description:",
            key="nav_metadata",
            use_container_width=True,
            on_click=lambda: st.session_state.update(active_page="Metadata"),
        )

        st.divider()

        st.markdown("##### :material/folder_open: Workspace")
        st.text_input(
            "Directorio de output",
            key="output_root",
            placeholder=str(Path.cwd() / "output"),
            help="Carpeta raiz donde se guardan y buscan las exportaciones PCD.",
        )
        if st.button(
            ":material/folder_open: Explorar",
            key="browse_output_btn",
            use_container_width=True,
        ):
            _file_browser_dialog("output_root", "directory")

        st.divider()

        col1, col2 = st.columns(2)
        col1.metric("Exportaciones", len(output_state.export_dirs))
        scan_result = st.session_state.get("scan_result")
        scan_label = scan_result.source.bag_type if scan_result else "-"
        col2.metric("Bag", scan_label)

        st.divider()
        st.caption("ForestSeg v1.0")


@st.dialog("Explorador de archivos", width="medium")
def _file_browser_dialog(target_key: str, mode: str = "directory") -> None:
    st.markdown(
        """
        <style>
        [data-testid="stDialog"] button[data-testid="stBaseButton-secondary"],
        [data-testid="stDialog"] button[data-testid="stBaseButton-secondary"]:focus,
        [data-testid="stDialog"] button[data-testid="stBaseButton-secondary"]:focus-visible,
        [data-testid="stDialog"] button[data-testid="stBaseButton-secondary"]:focus-within,
        [data-testid="stDialog"] button[data-testid="stBaseButton-secondary"]:active,
        [data-testid="stDialog"] button[data-testid="stBaseButton-secondary"]:visited,
        [data-testid="stDialog"] button[data-testid="stBaseButton-secondary"][kind] {
            background: transparent !important;
            border: 1px solid transparent !important;
            box-shadow: none !important;
            outline: none !important;
            padding: 0.4rem 0.7rem !important;
            border-radius: 6px !important;
            justify-content: flex-start !important;
            text-align: left !important;
            color: inherit !important;
        }
        [data-testid="stDialog"] button[data-testid="stBaseButton-secondary"] p,
        [data-testid="stDialog"] button[data-testid="stBaseButton-secondary"] > div {
            text-align: left !important;
            justify-content: flex-start !important;
        }
        [data-testid="stDialog"] button[data-testid="stBaseButton-secondary"]:hover {
            background: rgba(128, 128, 128, 0.15) !important;
            border-color: rgba(128, 128, 128, 0.3) !important;
        }
        [data-testid="stDialog"] > div > div[data-testid="stVerticalBlockBorderWrapper"] {
            overflow: hidden !important;
        }
        [data-testid="stDialog"] [data-testid="stVerticalBlockBorderWrapper"]
            [data-testid="stVerticalBlockBorderWrapper"] div[style*="overflow"] {
            overflow-y: scroll !important;
        }
        [data-testid="stDialog"] * {
            scrollbar-width: auto !important;
            scrollbar-color: rgba(128,128,128,0.5) transparent !important;
        }
        [data-testid="stDialog"] *::-webkit-scrollbar {
            width: 10px !important;
            display: block !important;
        }
        [data-testid="stDialog"] *::-webkit-scrollbar-track {
            background: rgba(128,128,128,0.1) !important;
            border-radius: 5px !important;
        }
        [data-testid="stDialog"] *::-webkit-scrollbar-thumb {
            background: rgba(128,128,128,0.45) !important;
            border-radius: 5px !important;
        }
        [data-testid="stDialog"] *::-webkit-scrollbar-thumb:hover {
            background: rgba(128,128,128,0.65) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    current_dir_key = f"_browser_{target_key}"
    current_dir = Path(st.session_state.get(current_dir_key, str(Path.home())))
    if not current_dir.is_dir():
        current_dir = Path.home()
        st.session_state[current_dir_key] = str(current_dir)

    # ── Navigation bar ──
    nav_up, nav_home, nav_path = st.columns([0.08, 0.08, 0.84])
    if nav_up.button(":material/arrow_upward:", key="dlg_up", help="Subir un nivel"):
        parent = current_dir.parent
        if parent != current_dir:
            st.session_state[current_dir_key] = str(parent)
            st.rerun(scope="fragment")
    if nav_home.button(":material/home:", key="dlg_home", help="Ir al inicio"):
        st.session_state[current_dir_key] = str(Path.home())
        st.rerun(scope="fragment")
    nav_path.markdown(f"`{current_dir}`")

    # ── Select current folder ──
    if mode in ("directory", "both"):
        if st.button(
            ":material/check: Seleccionar esta carpeta",
            key="dlg_pick_current",
            type="primary",
            use_container_width=True,
        ):
            st.session_state[target_key] = str(current_dir)
            st.rerun()

    st.divider()

    # ── List entries ──
    try:
        entries = sorted(current_dir.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        st.warning("Sin permisos para leer este directorio.")
        return

    dirs = [e for e in entries if e.is_dir() and not e.name.startswith(".")]
    files: list[Path] = []
    if mode in ("file", "both"):
        bag_extensions = {".bag", ".db3", ".yaml", ".yml"}
        files = [
            e for e in entries
            if e.is_file() and not e.name.startswith(".") and e.suffix.lower() in bag_extensions
        ]

    if not dirs and not files:
        st.caption("Carpeta vacia.")
        return

    with st.container(height=500):
        for d in dirs:
            col_name, col_pick = st.columns([0.9, 0.1])
            if col_name.button(
                d.name,
                icon=":material/folder:",
                key=f"dlg_nav_{d.name}",
                use_container_width=True,
            ):
                st.session_state[current_dir_key] = str(d)
                st.rerun(scope="fragment")
            if mode in ("directory", "both"):
                if col_pick.button(":material/check:", key=f"dlg_pick_{d.name}", help="Seleccionar"):
                    st.session_state[target_key] = str(d)
                    st.rerun()

        if files:
            st.divider()
            for f in files:
                col_name, col_pick = st.columns([0.9, 0.1])
                col_name.markdown(f":material/description: {f.name}")
                if col_pick.button(":material/check:", key=f"dlg_file_{f.name}", help="Seleccionar"):
                    st.session_state[target_key] = str(f)
                    st.rerun()




def _build_output_browser_state() -> OutputBrowserState:
    root_raw = str(st.session_state["output_root"]).strip()
    if not root_raw:
        st.session_state["selected_pcd_export_dir"] = None
        st.session_state["selected_pcd_preview_file"] = None
        return OutputBrowserState(
            root_raw="",
            root=None,
            export_dirs=[],
            selected_export_dir=None,
        )

    root = Path(root_raw).expanduser()
    export_dirs = discover_pcd_directories(root)

    selected_export_dir = None
    if export_dirs:
        valid_paths = {str(path): path for path in export_dirs}
        current = st.session_state.get("selected_pcd_export_dir")
        if current not in valid_paths:
            current = str(export_dirs[0])
            st.session_state["selected_pcd_preview_file"] = None
        st.session_state["selected_pcd_export_dir"] = current
        selected_export_dir = valid_paths[current]
    else:
        st.session_state["selected_pcd_export_dir"] = None
        st.session_state["selected_pcd_preview_file"] = None

    return OutputBrowserState(
        root_raw=root_raw,
        root=root,
        export_dirs=export_dirs,
        selected_export_dir=selected_export_dir,
    )


@st.cache_resource(show_spinner=False)
def _get_plotly_js_bundle() -> str:
    return get_plotlyjs()


def _render_persistent_plotly_chart(
    fig: go.Figure,
    *,
    storage_key: str,
    height: int | None = None,
) -> None:
    html = _build_persistent_plotly_html(fig, storage_key=storage_key)
    chart_height = height
    if chart_height is None:
        chart_height = int(fig.layout.height) if fig.layout.height is not None else 720
    components.html(html, height=chart_height + 8, scrolling=False)


def _build_persistent_plotly_html(
    fig: go.Figure,
    *,
    storage_key: str,
) -> str:
    plot_height = int(fig.layout.height) if fig.layout.height is not None else 720
    plot_div_id = f"plotly-{re.sub(r'[^a-zA-Z0-9_-]+', '-', storage_key)}"
    figure_json = fig.to_json().replace("</script>", "<\\/script>")
    plotly_js = _get_plotly_js_bundle().replace("</script>", "<\\/script>")
    config_json = json.dumps({"displaylogo": False, "responsive": True})
    storage_key_json = json.dumps(storage_key)
    plot_div_id_json = json.dumps(plot_div_id)
    plot_height_px = json.dumps(f"{plot_height}px")

    return f"""
    <div id={plot_div_id_json} style="width: 100%; height: {plot_height_px};"></div>
    <script>{plotly_js}</script>
    <script>
    const storageKey = {storage_key_json};
    const plotDiv = document.getElementById({plot_div_id_json});
    const figure = {figure_json};
    const config = {config_json};

    function cloneValue(value) {{
        return value ? JSON.parse(JSON.stringify(value)) : null;
    }}

    function readSavedCamera() {{
        try {{
            const raw = window.localStorage.getItem(storageKey);
            return raw ? JSON.parse(raw) : null;
        }} catch (error) {{
            return null;
        }}
    }}

    function writeSavedCamera(camera) {{
        try {{
            window.localStorage.setItem(storageKey, JSON.stringify(camera));
        }} catch (error) {{
            return;
        }}
    }}

    function mergeCameraUpdate(currentCamera, eventData) {{
        if (!eventData) {{
            return null;
        }}
        if (eventData["scene.camera"]) {{
            return cloneValue(eventData["scene.camera"]);
        }}

        const nextCamera = cloneValue(currentCamera) || {{}};
        let changed = false;
        for (const section of ["center", "eye", "up"]) {{
            for (const axis of ["x", "y", "z"]) {{
                const key = `scene.camera.${{section}}.${{axis}}`;
                if (eventData[key] !== undefined) {{
                    nextCamera[section] = nextCamera[section] || {{}};
                    nextCamera[section][axis] = eventData[key];
                    changed = true;
                }}
            }}
        }}
        const projectionKey = "scene.camera.projection.type";
        if (eventData[projectionKey] !== undefined) {{
            nextCamera.projection = nextCamera.projection || {{}};
            nextCamera.projection.type = eventData[projectionKey];
            changed = true;
        }}
        return changed ? nextCamera : null;
    }}

    const savedCamera = readSavedCamera();
    if (savedCamera) {{
        figure.layout = figure.layout || {{}};
        figure.layout.scene = figure.layout.scene || {{}};
        figure.layout.scene.camera = savedCamera;
    }}

    Plotly.newPlot(plotDiv, figure.data, figure.layout, config).then((gd) => {{
        let currentCamera = cloneValue(savedCamera || figure.layout?.scene?.camera);

        gd.on("plotly_relayout", (eventData) => {{
            const nextCamera = mergeCameraUpdate(currentCamera, eventData);
            if (!nextCamera) {{
                return;
            }}
            currentCamera = nextCamera;
            writeSavedCamera(currentCamera);
        }});

        const resizePlot = () => Plotly.Plots.resize(gd);
        window.addEventListener("resize", resizePlot);
        requestAnimationFrame(resizePlot);
    }});
    </script>
    """


def _render_global_messages() -> None:
    if st.session_state.get("scan_error"):
        st.error(st.session_state["scan_error"])
    elif st.session_state.get("scan_success"):
        st.success(st.session_state["scan_success"])


def _render_convert_page(output_state: OutputBrowserState) -> None:
    st.markdown("#### :material/swap_horiz: Convertir Bag → PCD")
    scan_result = st.session_state.get("scan_result")
    selected_topic = st.session_state.get("selected_topic")

    # ── Step 1: Seleccionar bag ──
    _render_step_header(1, "Seleccionar bag", done=scan_result is not None)

    with st.container(border=True):
        st.radio(
            "Tipo esperado",
            options=["Autodetectar", "Archivo .bag", "Directorio rosbag2"],
            key="input_mode",
            horizontal=True,
            help="En la mayoria de los casos, Autodetectar funciona correctamente.",
        )
        path_col, browse_col, scan_col = st.columns([3.5, 0.22, 0.8])
        path_col.text_input(
            "Ruta de entrada",
            key="input_path",
            placeholder="/ruta/al/archivo.bag o /ruta/al/directorio_rosbag2",
            label_visibility="collapsed",
        )
        input_mode = st.session_state.get("input_mode", "Autodetectar")
        browser_mode = "directory" if input_mode == "Directorio rosbag2" else "both"
        if browse_col.button(":material/folder_open:", key="browse_input_btn"):
            _file_browser_dialog("input_path", browser_mode)
        if scan_col.button(
            ":material/search: Escanear",
            key="scan_topics_primary",
            type="primary",
            use_container_width=True,
        ):
            _handle_scan()

    if scan_result and _scan_result_is_stale(scan_result):
        st.warning("La ruta de entrada cambio desde el ultimo escaneo. Vuelva a escanear.")

    if scan_result:
        _render_scan_summary(scan_result)
    elif not st.session_state.get("scan_error"):
        st.info("Ingrese la ruta a un archivo .bag o directorio rosbag2 y presione **Escanear**.")

    # ── Step 2: Elegir topico ──
    has_topics = bool(scan_result and scan_result.pointcloud_topics)
    _render_step_header(
        2,
        "Elegir topico",
        done=selected_topic is not None,
        disabled=not scan_result,
    )

    if not scan_result:
        st.caption("Escanee el bag en el paso anterior para habilitar este paso.")
    elif not scan_result.pointcloud_topics:
        st.warning("El bag no contiene topicos PointCloud2.")
    else:
        topic = _render_topic_selector(scan_result)
        _render_preview_panel(scan_result, topic)

        # ── Step 3: Configurar exportacion ──
        _render_step_header(3, "Configurar exportacion")

        with st.container(border=True):
            export_col1, export_col2 = st.columns([2.5, 1.5])
            export_col1.text_input(
                "Nombre de subcarpeta",
                key="export_name",
                placeholder="exportacion_lidar",
                help="Subcarpeta dentro del directorio de output donde se guardaran los .pcd.",
            )
            opt_col1, opt_col2 = export_col2.columns(2)
            opt_col1.checkbox("Intensidad", key="save_intensity", help="Guardar intensidad si el campo existe.")
            opt_col2.checkbox("Saltar vacios", key="skip_empty_frames", help="Omitir frames sin puntos validos.")

            st.divider()

            preview = st.session_state.get("preview_cache", {})
            total_msgs = 0
            for cached in preview.values():
                if "message_count" in cached:
                    total_msgs = cached["message_count"]
                    break

            st.radio(
                "Modo de extraccion",
                options=["Todos los frames", "Cantidad maxima", "Frame especifico"],
                key="extraction_mode",
                horizontal=True,
                help="Elija cuantos frames exportar del bag.",
            )

            extraction_mode = st.session_state.get("extraction_mode", "Todos los frames")
            if extraction_mode == "Cantidad maxima":
                st.number_input(
                    "Maximo de frames a extraer",
                    min_value=1,
                    max_value=max(total_msgs, 1) if total_msgs else 100000,
                    key="max_frames",
                    help="Se extraeran los primeros N frames del topico.",
                )
            elif extraction_mode == "Frame especifico":
                st.number_input(
                    "Indice del frame (base 0)",
                    min_value=0,
                    max_value=max(total_msgs - 1, 0) if total_msgs else 100000,
                    key="specific_frame",
                    help="Se extraera unicamente el frame con este indice.",
                )

        # ── Step 4: Convertir ──
        _render_step_header(4, "Convertir", disabled=not output_state.root_raw)

        if not output_state.root_raw:
            st.warning("Defina un directorio de output en el panel lateral antes de convertir.")
        elif st.button(
            ":material/play_arrow: Convertir a PCD",
            type="primary",
            key="convert_to_pcd",
            use_container_width=True,
        ):
            _handle_conversion(scan_result, topic)

    _render_conversion_summary()


def _render_step_header(
    number: int,
    title: str,
    *,
    done: bool = False,
    disabled: bool = False,
) -> None:
    badge_class = "done" if done else ("disabled" if disabled else "")
    st.markdown(
        f'<div class="step-header">'
        f'<span class="step-badge {badge_class}">{number}</span>'
        f'<span class="step-title">{title}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )



def _scan_result_is_stale(scan_result: ScanResult) -> bool:
    input_path_raw = str(st.session_state["input_path"]).strip()
    if not input_path_raw:
        return False
    try:
        resolved_input = Path(input_path_raw).expanduser().resolve()
    except Exception:
        return False
    return resolved_input not in {
        scan_result.source.original_path,
        scan_result.source.reader_path,
    }


def _render_scan_summary(scan_result: ScanResult) -> None:
    st.caption(
        f"Bag: {scan_result.source.bag_type} | Topicos: {len(scan_result.topics)} | PointCloud2: {len(scan_result.pointcloud_topics)}"
    )


def _render_topic_selector(scan_result: ScanResult) -> str:
    topic_labels = {
        f"{topic.topic} | {topic.message_count} mensajes": topic.topic
        for topic in scan_result.pointcloud_topics
    }
    labels = list(topic_labels.keys())
    selected_topic = st.session_state.get("selected_topic")
    default_index = 0
    if selected_topic:
        for index, label in enumerate(labels):
            if topic_labels[label] == selected_topic:
                default_index = index
                break

    default_label = labels[default_index]
    if st.session_state.get("convert_topic_selectbox") not in labels:
        st.session_state["convert_topic_selectbox"] = default_label

    selected_label = st.selectbox(
        "Topico PointCloud2",
        options=labels,
        key="convert_topic_selectbox",
    )
    selected_topic = topic_labels[selected_label]
    st.session_state["selected_topic"] = selected_topic
    return selected_topic


def _render_preview_panel(scan_result: ScanResult, topic: str) -> None:
    preview = _load_preview(scan_result, topic)

    if "error" in preview:
        st.warning(preview["error"])
        return

    cards = [
        {
            "label": "Mensajes del topico",
            "value": str(preview["message_count"]),
            "detail": "Mensajes disponibles para exportar",
            "tone": "active",
        },
        {
            "label": "Primer frame",
            "value": str(preview["first_frame_points"]),
            "detail": "Puntos declarados",
            "tone": "ready",
        },
        {
            "label": "Puntos validos",
            "value": str(preview["first_valid_points"]),
            "detail": "Sin NaN ni Inf",
            "tone": "ready",
        },
        {
            "label": "Intensidad",
            "value": "Si" if preview["has_intensity"] else "No",
            "detail": "Campo intensity en el mensaje",
            "tone": "active" if preview["has_intensity"] else "pending",
        },
    ]
    _render_stat_cards(cards)

    with st.expander(":material/info: Detalles del primer frame", expanded=False):
        detail_col1, detail_col2 = st.columns([2.0, 1.1])
        with detail_col1:
            st.markdown("**Campos detectados**")
            fields = preview["field_names"] or []
            if fields:
                st.markdown(" ".join(f"`{field}`" for field in fields))
            else:
                st.caption("Sin campos")

        with detail_col2:
            st.markdown("**Primeros timestamps**")
            sample_timestamps = preview["sample_timestamps"] or []
            if sample_timestamps:
                rows = [
                    {
                        "iso_utc": item.get("iso_utc", "-"),
                        "ns": item.get("ns", "-"),
                    }
                    for item in sample_timestamps
                    if item
                ]
                st.dataframe(rows, use_container_width=True)
            else:
                st.caption("No se pudieron extraer timestamps de muestra.")


def _load_preview(scan_result: ScanResult, topic: str) -> dict[str, object]:
    cache_key = f"{scan_result.source.reader_path}:{topic}"
    preview_cache = st.session_state["preview_cache"]
    if cache_key in preview_cache:
        return preview_cache[cache_key]

    topic_info = next(item for item in scan_result.pointcloud_topics if item.topic == topic)
    preview: dict[str, object] = {
        "message_count": topic_info.message_count,
        "field_names": [],
        "first_frame_points": 0,
        "first_valid_points": 0,
        "sample_timestamps": [],
        "has_intensity": False,
    }

    try:
        for index, bag_message in enumerate(iter_topic_messages(scan_result.source, topic)):
            if index == 0:
                parsed = parse_pointcloud2(bag_message.message, include_intensity=True)
                preview["field_names"] = parsed.field_names
                preview["first_frame_points"] = parsed.point_count_raw
                preview["first_valid_points"] = parsed.point_count_valid
                preview["has_intensity"] = parsed.has_intensity_field

            preview["sample_timestamps"].append(timestamp_payload(bag_message.timestamp_ns))
            if index >= 4:
                break
    except (BagReaderError, PointCloud2ParseError) as exc:
        preview = {"error": f"No se pudo generar la vista previa: {exc}"}

    preview_cache[cache_key] = preview
    return preview


def _handle_scan() -> None:
    try:
        result = scan_bag(st.session_state["input_path"])
    except BagReaderError as exc:
        st.session_state["scan_result"] = None
        st.session_state["scan_error"] = str(exc)
        st.session_state["scan_success"] = None
        st.session_state["preview_cache"] = {}
        st.session_state["selected_topic"] = None
        return

    st.session_state["scan_result"] = result
    st.session_state["scan_error"] = None
    st.session_state["scan_success"] = (
        f"Bag detectado como {result.source.bag_type} con {len(result.topics)} topicos totales."
    )
    st.session_state["preview_cache"] = {}
    st.session_state["selected_topic"] = (
        result.pointcloud_topics[0].topic if result.pointcloud_topics else None
    )


def _handle_conversion(scan_result: ScanResult, topic: str) -> None:
    topic_info = next(item for item in scan_result.pointcloud_topics if item.topic == topic)

    try:
        export_dir = ensure_export_directory(
            st.session_state["output_root"],
            st.session_state["export_name"],
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    progress_bar = st.progress(0.0)
    status = st.empty()
    errors: list[str] = []
    suppressed_errors = 0

    generated_count = 0
    skipped_empty = 0
    detected_fields: list[str] = []
    had_intensity = False
    first_timestamp_ns = None
    last_timestamp_ns = None
    first_header_timestamp_ns = None
    last_header_timestamp_ns = None

    extraction_mode = st.session_state.get("extraction_mode", "Todos los frames")
    max_frames = int(st.session_state.get("max_frames", 100))
    specific_frame = int(st.session_state.get("specific_frame", 0))

    try:
        for frame_index, bag_message in enumerate(iter_topic_messages(scan_result.source, topic)):
            if extraction_mode == "Cantidad maxima" and generated_count >= max_frames:
                break
            if extraction_mode == "Frame especifico" and frame_index > specific_frame:
                break
            if extraction_mode == "Frame especifico" and frame_index != specific_frame:
                if frame_index < specific_frame:
                    continue

            processed = frame_index + 1
            total_messages = max(topic_info.message_count, 1)
            progress_bar.progress(min(processed / total_messages, 1.0))
            status.info(f"Procesando frame {processed}/{topic_info.message_count}")

            if first_timestamp_ns is None:
                first_timestamp_ns = bag_message.timestamp_ns
            last_timestamp_ns = bag_message.timestamp_ns

            try:
                parsed = parse_pointcloud2(
                    bag_message.message,
                    include_intensity=st.session_state["save_intensity"],
                )
                if not detected_fields:
                    detected_fields = parsed.field_names
                had_intensity = had_intensity or parsed.has_intensity_field
                if parsed.header_stamp_ns is not None:
                    if first_header_timestamp_ns is None:
                        first_header_timestamp_ns = parsed.header_stamp_ns
                    last_header_timestamp_ns = parsed.header_stamp_ns

                if parsed.point_count_valid == 0 and st.session_state["skip_empty_frames"]:
                    skipped_empty += 1
                    errors.append(f"Frame {frame_index}: sin puntos validos, omitido.")
                    continue

                write_pcd(
                    build_pcd_path(export_dir, generated_count),
                    parsed.xyz,
                    parsed.intensity if st.session_state["save_intensity"] else None,
                )
                generated_count += 1
            except (PointCloud2ParseError, PCDWriteError, ValueError) as exc:
                if len(errors) < 200:
                    errors.append(f"Frame {frame_index}: {exc}")
                else:
                    suppressed_errors += 1
                if isinstance(exc, PCDWriteError) and "Open3D" in str(exc):
                    break
    except BagReaderError as exc:
        errors.append(str(exc))

    if suppressed_errors:
        errors.append(f"Se suprimieron {suppressed_errors} errores adicionales.")

    metadata = {
        "source": scan_result.source.to_dict(),
        "topic": topic,
        "total_messages": topic_info.message_count,
        "pcd_generated": generated_count,
        "detected_fields": detected_fields,
        "had_intensity": had_intensity,
        "save_intensity_requested": bool(st.session_state["save_intensity"]),
        "skip_empty_frames": bool(st.session_state["skip_empty_frames"]),
        "extraction_mode": extraction_mode,
        "max_frames": max_frames if extraction_mode == "Cantidad maxima" else None,
        "specific_frame": specific_frame if extraction_mode == "Frame especifico" else None,
        "skipped_empty_frames": skipped_empty,
        "timestamp_initial": timestamp_payload(first_timestamp_ns),
        "timestamp_final": timestamp_payload(last_timestamp_ns),
        "header_timestamp_initial": timestamp_payload(first_header_timestamp_ns),
        "header_timestamp_final": timestamp_payload(last_header_timestamp_ns),
        "errors": errors,
    }
    write_json(export_dir / "metadata.json", metadata)

    validation_summary: dict[str, object] | None = None
    if generated_count > 0:
        try:
            validation_summary = validate_export_directory(
                export_dir,
                sample_size=min(generated_count, 3),
            )
        except Exception as exc:
            validation_summary = {"valid": False, "error": str(exc)}

    st.session_state["last_conversion"] = {
        "export_dir": str(export_dir),
        "generated_count": generated_count,
        "skipped_empty": skipped_empty,
        "total_messages": topic_info.message_count,
        "errors": errors,
        "metadata_path": str(export_dir / "metadata.json"),
        "validation": validation_summary,
    }
    if validation_summary is not None:
        st.session_state["validation_cache"][str(export_dir.resolve())] = validation_summary

    st.session_state["selected_pcd_export_dir"] = str(export_dir.resolve())
    st.session_state["selected_pcd_preview_file"] = None
    status.success(f"Conversion finalizada. Se generaron {generated_count} archivos .pcd.")
    progress_bar.progress(1.0)
    st.rerun()


def _render_conversion_summary() -> None:
    summary = st.session_state.get("last_conversion")
    if not summary:
        return

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("##### :material/check_circle: Resumen de la ultima conversion")
    cards = [
        {
            "label": "Mensajes leidos",
            "value": str(summary["total_messages"]),
            "detail": "Frames procesados",
            "tone": "active",
        },
        {
            "label": "PCD generados",
            "value": str(summary["generated_count"]),
            "detail": "Archivos creados",
            "tone": "ready",
        },
        {
            "label": "Frames omitidos",
            "value": str(summary["skipped_empty"]),
            "detail": "Frames vacios ignorados",
            "tone": "warn" if summary["skipped_empty"] else "pending",
        },
    ]
    _render_stat_cards(cards)

    validation = summary.get("validation")
    if validation:
        if validation.get("valid"):
            st.success("Validacion rapida de PCD completada.")
        else:
            st.warning(
                f"No se pudo validar la exportacion: {validation.get('error', 'sin detalle')}"
            )

    errors = summary.get("errors") or []
    if errors:
        with st.expander(f":material/warning: {len(errors)} errores y omisiones"):
            st.write("\n".join(errors))


def _render_segment_page(output_state: OutputBrowserState) -> None:
    st.markdown("#### :material/forest: Segmentar")
    if not output_state.root_raw:
        st.info("Defina un directorio de output en el panel lateral para segmentar nubes de puntos.")
        return

    if not output_state.export_dirs:
        st.info("No se encontraron exportaciones con archivos .pcd. Convierta un rosbag primero.")
        return

    selection_col, model_col = st.columns(2, gap="large")

    with selection_col:
        # ── Step 1: Seleccionar PCD ──
        _render_step_header(1, "Seleccionar nube de puntos")

        with st.container(border=True):
            selected_export_dir = _render_export_directory_selector(
                output_state,
                widget_key="seg_export_selector",
                label="Carpeta de exportacion",
            )
            details = _load_export_details(selected_export_dir)
            if not details.pcd_files:
                st.warning("La carpeta seleccionada no contiene archivos .pcd.")
                return

            file_names = [p.name for p in details.pcd_files]
            selected_file = st.selectbox(
                "Archivo PCD",
                options=file_names,
                key="seg_selected_pcd",
                help=f"{len(file_names)} archivos disponibles.",
            )
            selected_pcd_path = details.export_dir / selected_file

    with model_col:
        # ── Step 2: Seleccionar modelo ──
        _render_step_header(2, "Seleccionar modelo")

        with st.container(border=True):
            model_dir_col, browse_col = st.columns([3.7, 0.3])
            default_model_dir = str(Path(__file__).parent / "models")
            model_dir_col.text_input(
                "Directorio del modelo",
                key="seg_model_dir",
                placeholder=default_model_dir,
                help="Carpeta con el modelo ONNX y los archivos arch_cfg.yaml / data_cfg.yaml.",
            )
            browse_col.markdown("<div style='height: 1.8rem;'></div>", unsafe_allow_html=True)
            if browse_col.button(
                ":material/folder_open:",
                key="browse_model_dir_btn",
                use_container_width=True,
            ):
                _file_browser_dialog("seg_model_dir", "directory")

            model_dir_raw = st.session_state.get("seg_model_dir", "").strip()
            if not model_dir_raw:
                model_dir_raw = str(Path(__file__).parent / "models")
            model_dir = Path(model_dir_raw)
            onnx_files = sorted(model_dir.glob("*.onnx")) if model_dir.is_dir() else []

            if not onnx_files:
                st.warning("No se encontraron archivos .onnx en el directorio seleccionado.")
                return

            onnx_names = [f.name for f in onnx_files]
            selected_onnx = st.selectbox(
                "Modelo ONNX",
                options=onnx_names,
                key="seg_selected_onnx",
            )
            model_path = model_dir / selected_onnx

            try:
                seg_config = load_configs(model_dir)
            except SegmentationError as exc:
                st.error(str(exc))
                return

    # ── Step 3: Ejecutar segmentacion ──
    _render_step_header(3, "Ejecutar segmentacion")

    if st.button(
        ":material/play_arrow: Segmentar",
        type="primary",
        key="run_segmentation",
        use_container_width=True,
    ):
        _handle_segmentation(selected_pcd_path, model_path, seg_config)

    # ── Resultado ──
    seg_result = st.session_state.get("seg_last_result")
    if seg_result and seg_result.get("pcd_path") == str(selected_pcd_path):
        _render_segmentation_result(seg_result, seg_config)


def _handle_segmentation(
    pcd_path: Path,
    model_path: Path,
    config: SegmentationConfig,
) -> None:
    status = st.empty()
    progress = st.progress(0.0)

    status.info("Cargando nube de puntos...")
    progress.progress(0.1)
    try:
        preview = load_pcd_preview_data(pcd_path, max_points=None)
    except PCDWriteError as exc:
        status.error(f"No se pudo cargar el PCD: {exc}")
        return

    status.info("Ejecutando proyeccion esferica e inferencia...")
    progress.progress(0.3)
    try:
        result = segment_pcd(preview.xyz, preview.intensity, model_path, config)
    except SegmentationError as exc:
        status.error(f"Error en la segmentacion: {exc}")
        return

    progress.progress(1.0)
    total_classified = sum(
        c for tid, c in result.class_counts.items() if tid != 0
    )
    status.success(
        f"Segmentacion completada: {total_classified:,} puntos clasificados "
        f"de {result.num_points:,} totales."
    )

    # Reset per-result viewer defaults so the controls start from a consistent baseline.
    st.session_state["seg_preview_point_size"] = 2.0

    st.session_state["seg_last_result"] = {
        "pcd_path": str(pcd_path),
        "labels": result.labels,
        "class_counts": result.class_counts,
        "num_points": result.num_points,
        "xyz": preview.xyz,
        "intensity": preview.intensity,
    }


def _render_segmentation_result(
    seg_result: dict,
    config: SegmentationConfig,
) -> None:
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("##### :material/check_circle: Resultado de segmentacion")

    train_labels = config.train_id_labels
    class_counts = seg_result["class_counts"]

    controls_col, viewer_col = st.columns([1.05, 1.95], gap="large")

    xyz = seg_result["xyz"]
    labels = seg_result["labels"]
    num_points = xyz.shape[0]
    class_colors = _get_segmentation_class_colors(config)
    class_visibility = _get_segmentation_class_visibility(config)

    with controls_col:
        st.markdown("##### :material/tune: Controles")
        with st.expander(":material/category: Clases", expanded=True):
            class_colors, class_visibility = _render_segmentation_class_controls(
                train_labels=train_labels,
                class_counts=class_counts,
                num_points=num_points,
                class_colors=class_colors,
                class_visibility=class_visibility,
            )

        with st.expander(":material/filter_alt: Muestreo", expanded=True):
            use_ds = st.checkbox(
                "Downsampling",
                key="seg_preview_use_downsampling",
            )
            max_pts: int | None = None
            if use_ds:
                max_pts = st.slider(
                    "Max puntos",
                    min_value=5000,
                    max_value=200000,
                    step=5000,
                    key="seg_preview_max_points",
                )
            st.slider(
                "Tamano de punto",
                min_value=0.5,
                max_value=6.0,
                step=0.25,
                key="seg_preview_point_size",
            )

        with st.expander(":material/palette: Visualizacion", expanded=True):
            st.selectbox(
                "Fondo",
                options=["Oscuro", "Claro"],
                key="seg_preview_background",
            )
            st.checkbox("Bounds", key="seg_preview_show_bounds")
            if st.button(
                ":material/refresh: Resetear vista",
                use_container_width=True,
                key="seg_reset_camera",
            ):
                st.session_state["seg_chart_nonce"] += 1

    sample_xyz, sample_labels = _sample_segmentation(xyz, labels, max_pts)

    with viewer_col:
        st.markdown("##### :material/forest: Visor de segmentacion")
        fig = _build_segmentation_figure(
            sample_xyz,
            sample_labels,
            config,
            class_colors=class_colors,
            class_visibility=class_visibility,
            point_size=float(st.session_state["seg_preview_point_size"]),
            background_mode=st.session_state["seg_preview_background"],
            show_bounds=bool(st.session_state["seg_preview_show_bounds"]),
            camera_nonce=int(st.session_state["seg_chart_nonce"]),
        )
        _render_persistent_plotly_chart(
            fig,
            storage_key=(
                f"forestseg-seg-camera-{seg_result['pcd_path']}-"
                f"{st.session_state['seg_chart_nonce']}"
            ),
        )


def _sample_segmentation(
    xyz: np.ndarray,
    labels: np.ndarray,
    max_points: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if max_points is None or max_points <= 0 or xyz.shape[0] <= max_points:
        return xyz, labels
    rng = np.random.default_rng(42)
    indices = np.sort(rng.choice(xyz.shape[0], size=max_points, replace=False))
    return xyz[indices], labels[indices]


def _build_segmentation_figure(
    xyz: np.ndarray,
    labels: np.ndarray,
    config: SegmentationConfig,
    *,
    class_colors: dict[int, list[int]],
    class_visibility: dict[int, bool],
    point_size: float,
    background_mode: str,
    show_bounds: bool,
    camera_nonce: int,
) -> go.Figure:
    background = _resolve_background_palette(background_mode)
    train_labels = config.train_id_labels

    fig = go.Figure()

    for tid in sorted(train_labels.keys()):
        if not class_visibility.get(tid, True):
            continue
        mask = labels == tid
        if not np.any(mask):
            continue
        rgb = class_colors.get(tid, [160, 160, 160])
        color_str = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
        pts = xyz[mask]
        label = _format_class_display_name(train_labels[tid])
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            name=label,
            marker={"size": point_size, "opacity": 0.85, "color": color_str},
            hovertemplate=f"{label}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>z=%{{z:.3f}}<extra></extra>",
        ))

    if show_bounds and xyz.shape[0] > 0:
        bounds_min = np.min(xyz, axis=0)
        bounds_max = np.max(xyz, axis=0)
        fig.add_trace(_build_bbox_trace(bounds_min, bounds_max, background["accent"]))

    fig.update_layout(
        height=720,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        paper_bgcolor=background["paper"],
        plot_bgcolor=background["paper"],
        font={"color": background["font"]},
        showlegend=True,
        legend={"itemsizing": "constant"},
        uirevision=f"seg-view-{camera_nonce}",
        scene={
            "uirevision": f"seg-view-{camera_nonce}",
            "aspectmode": "data",
            "bgcolor": background["scene"],
            "xaxis": {
                "title": "X",
                "gridcolor": background["grid"],
                "zerolinecolor": background["grid"],
                "backgroundcolor": background["scene"],
            },
            "yaxis": {
                "title": "Y",
                "gridcolor": background["grid"],
                "zerolinecolor": background["grid"],
                "backgroundcolor": background["scene"],
            },
            "zaxis": {
                "title": "Z",
                "gridcolor": background["grid"],
                "zerolinecolor": background["grid"],
                "backgroundcolor": background["scene"],
            },
        },
    )
    return fig


def _render_explore_page(output_state: OutputBrowserState) -> None:
    st.markdown("#### :material/view_in_ar: Explorar PCD")
    if not output_state.root_raw:
        st.info("Defina un directorio de output en el panel lateral para explorar exportaciones.")
        return

    if not output_state.export_dirs:
        st.info("No se encontraron carpetas con archivos .pcd en el directorio actual.")
        return

    selected_export_dir = _render_export_directory_selector(
        output_state,
        widget_key="explore_export_selector",
        label="Carpeta de exportacion",
    )
    details = _load_export_details(selected_export_dir)

    _render_output_overview(output_state, details)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    _render_pcd_directory_viewer(details)


def _render_output_overview(
    output_state: OutputBrowserState,
    details: ExportDirectoryDetails,
) -> None:
    metadata_topic = details.metadata.get("topic") if details.metadata else "-"
    parts = [
        f"{len(output_state.export_dirs)} exportaciones",
        f"{len(details.pcd_files)} PCD",
        f"ultima modificacion {details.last_modified_label}",
    ]
    if metadata_topic not in {"-", None, ""}:
        parts.append(f"topico {metadata_topic}")
    st.caption(" | ".join(parts))


def _render_export_directory_selector(
    output_state: OutputBrowserState,
    *,
    widget_key: str,
    label: str,
) -> Path:
    export_dirs = output_state.export_dirs
    selected_dir = st.session_state.get("selected_pcd_export_dir")
    available_paths = [str(path) for path in export_dirs]
    if selected_dir not in available_paths:
        selected_dir = available_paths[0]

    labels = {_export_dir_label(output_state.root, path): str(path) for path in export_dirs}
    label_list = list(labels.keys())
    default_label = next(text for text, path in labels.items() if path == selected_dir)

    if st.session_state.get(widget_key) not in label_list:
        st.session_state[widget_key] = default_label

    chosen_label = st.selectbox(
        label,
        options=label_list,
        key=widget_key,
    )
    chosen_dir = Path(labels[chosen_label])
    if str(chosen_dir) != st.session_state.get("selected_pcd_export_dir"):
        st.session_state["selected_pcd_export_dir"] = str(chosen_dir)
        st.session_state["selected_pcd_preview_file"] = None
        st.rerun()
    return chosen_dir


def _export_dir_label(output_root: Path | None, export_dir: Path) -> str:
    if output_root is None:
        return str(export_dir)

    resolved_root = output_root.resolve()
    if export_dir == resolved_root:
        return f"[directorio actual] {export_dir.name or export_dir}"
    try:
        return str(export_dir.relative_to(resolved_root))
    except ValueError:
        return str(export_dir)


def _load_export_details(export_dir: Path) -> ExportDirectoryDetails:
    pcd_files = sorted(export_dir.glob("*.pcd"))
    metadata_path = export_dir / "metadata.json"
    metadata_payload = None
    metadata_error = None

    if metadata_path.exists():
        try:
            metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as exc:
            metadata_error = f"No se pudo leer metadata.json: {exc}"

    timestamps = [path.stat().st_mtime for path in pcd_files]
    if metadata_path.exists():
        timestamps.append(metadata_path.stat().st_mtime)
    if not timestamps:
        timestamps.append(export_dir.stat().st_mtime)

    return ExportDirectoryDetails(
        export_dir=export_dir,
        pcd_files=pcd_files,
        metadata_path=metadata_path if metadata_path.exists() else None,
        metadata=metadata_payload,
        metadata_error=metadata_error,
        last_modified_label=_format_datetime(max(timestamps)),
    )


def _render_pcd_directory_viewer(details: ExportDirectoryDetails) -> None:
    if not details.pcd_files:
        st.info("La carpeta seleccionada no contiene archivos .pcd.")
        return

    file_names = [path.name for path in details.pcd_files]
    if st.session_state.get("selected_pcd_preview_file") not in file_names:
        st.session_state["selected_pcd_preview_file"] = file_names[0]

    controls_col, viewer_col = st.columns([1.05, 1.95], gap="large")

    with controls_col:
        st.markdown("##### :material/tune: Controles")
        with st.container(border=True):
            st.selectbox(
                "Archivo PCD",
                options=file_names,
                key="selected_pcd_preview_file",
                help=f"{len(file_names)} archivos disponibles en esta exportacion.",
            )

        with st.expander(":material/filter_alt: Muestreo y puntos", expanded=True):
            use_downsampling = st.checkbox(
                "Downsampling",
                key="pcd_preview_use_downsampling",
                help="Limita la cantidad de puntos renderizados para mejorar la fluidez.",
            )
            max_points: int | None = None
            if use_downsampling:
                max_points = st.slider(
                    "Max puntos",
                    min_value=5000,
                    max_value=100000,
                    step=5000,
                    key="pcd_preview_max_points",
                )

            st.slider(
                "Tamano de punto",
                min_value=0.5,
                max_value=6.0,
                step=0.25,
                key="pcd_preview_point_size",
            )

    selected_path = details.export_dir / st.session_state["selected_pcd_preview_file"]

    try:
        preview = load_pcd_preview_data(selected_path, max_points=max_points)
    except PCDWriteError as exc:
        st.warning(f"No se pudo abrir el PCD seleccionado: {exc}")
        return

    if preview.has_intensity and preview.intensity is not None:
        available_color_modes = ["Altura Z", "Intensidad"]
    else:
        available_color_modes = ["Altura Z"]

    current_color_mode = st.session_state.get("pcd_preview_color_mode", available_color_modes[0])
    if current_color_mode not in available_color_modes:
        st.session_state["pcd_preview_color_mode"] = available_color_modes[0]

    with controls_col:
        with st.expander(":material/palette: Visualizacion", expanded=True):
            col_bg, col_color = st.columns(2)
            col_bg.selectbox(
                "Fondo",
                options=["Oscuro", "Claro"],
                key="pcd_preview_background",
            )
            color_mode = col_color.selectbox(
                "Color",
                options=available_color_modes,
                key="pcd_preview_color_mode",
            )

            st.checkbox("Bounds y centroide", key="pcd_preview_show_bounds")
            if st.button(
                ":material/refresh: Resetear vista",
                use_container_width=True,
                key="reset_camera_button",
            ):
                st.session_state["pcd_preview_chart_nonce"] += 1

        with st.expander(":material/info: Info del archivo"):
            m1, m2 = st.columns(2)
            m1.metric("Puntos", f"{preview.original_point_count:,}")
            m2.metric("Renderizados", f"{preview.rendered_point_count:,}")
            m3, m4 = st.columns(2)
            m3.metric("Intensidad", "Si" if preview.has_intensity else "No")
            m4.metric("Centroide", _format_xyz(preview.centroid))

            if use_downsampling and preview.original_point_count > preview.rendered_point_count:
                st.caption(
                    f"Muestreo: {preview.rendered_point_count:,} de {preview.original_point_count:,} puntos."
                )
            st.caption(
                f"Bounds min=({_format_xyz(preview.bounds_min)}) max=({_format_xyz(preview.bounds_max)})"
            )

    with viewer_col:
        st.markdown("##### :material/view_in_ar: Visor 3D")
        _render_persistent_plotly_chart(
            _build_pcd_figure(
                preview,
                color_mode=color_mode,
                point_size=float(st.session_state["pcd_preview_point_size"]),
                background_mode=st.session_state["pcd_preview_background"],
                show_bounds=bool(st.session_state["pcd_preview_show_bounds"]),
                camera_nonce=int(st.session_state["pcd_preview_chart_nonce"]),
            ),
            storage_key=(
                f"forestseg-pcd-camera-{selected_path.resolve()}-"
                f"{st.session_state['pcd_preview_chart_nonce']}"
            ),
        )


def _build_pcd_figure(
    preview: PCDPreviewData,
    *,
    color_mode: str,
    point_size: float,
    background_mode: str,
    show_bounds: bool,
    camera_nonce: int,
) -> go.Figure:
    xyz = preview.xyz
    if xyz.shape[0] == 0:
        figure = go.Figure()
        figure.update_layout(
            height=700,
            margin={"l": 0, "r": 0, "t": 30, "b": 0},
            title="El archivo PCD no contiene puntos.",
        )
        return figure

    color_values, colorbar_title = _resolve_pcd_colors(preview, color_mode=color_mode)
    background = _resolve_background_palette(background_mode)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="markers",
            name="Puntos",
            marker={
                "size": point_size,
                "opacity": 0.86,
                "color": color_values,
                "colorscale": "Turbo",
                "colorbar": {"title": colorbar_title},
            },
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
        )
    )

    if show_bounds:
        figure.add_trace(
            go.Scatter3d(
                x=[preview.centroid[0]],
                y=[preview.centroid[1]],
                z=[preview.centroid[2]],
                mode="markers",
                name="Centroide",
                marker={"size": max(point_size * 3.5, 4.0), "color": "#ef4444"},
                hovertemplate="Centroide<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
            )
        )
        figure.add_trace(_build_bbox_trace(preview.bounds_min, preview.bounds_max, background["accent"]))

    figure.update_layout(
        height=720,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        paper_bgcolor=background["paper"],
        plot_bgcolor=background["paper"],
        font={"color": background["font"]},
        showlegend=show_bounds,
        uirevision=f"pcd-view-{camera_nonce}",
        scene={
            "uirevision": f"pcd-view-{camera_nonce}",
            "aspectmode": "data",
            "bgcolor": background["scene"],
            "xaxis": {
                "title": "X",
                "gridcolor": background["grid"],
                "zerolinecolor": background["grid"],
                "backgroundcolor": background["scene"],
            },
            "yaxis": {
                "title": "Y",
                "gridcolor": background["grid"],
                "zerolinecolor": background["grid"],
                "backgroundcolor": background["scene"],
            },
            "zaxis": {
                "title": "Z",
                "gridcolor": background["grid"],
                "zerolinecolor": background["grid"],
                "backgroundcolor": background["scene"],
            },
        },
    )
    return figure


def _build_bbox_trace(bounds_min: np.ndarray, bounds_max: np.ndarray, color: str) -> go.Scatter3d:
    xmin, ymin, zmin = bounds_min.tolist()
    xmax, ymax, zmax = bounds_max.tolist()
    vertices = [
        (xmin, ymin, zmin),
        (xmax, ymin, zmin),
        (xmax, ymax, zmin),
        (xmin, ymax, zmin),
        (xmin, ymin, zmax),
        (xmax, ymin, zmax),
        (xmax, ymax, zmax),
        (xmin, ymax, zmax),
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for start, end in edges:
        for idx in (start, end):
            x, y, z = vertices[idx]
            xs.append(x)
            ys.append(y)
            zs.append(z)
        xs.append(None)
        ys.append(None)
        zs.append(None)

    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        name="Bounds",
        line={"color": color, "width": 4},
        hoverinfo="skip",
    )


def _render_segmentation_class_controls(
    *,
    train_labels: dict[int, str],
    class_counts: dict[int, int],
    num_points: int,
    class_colors: dict[int, list[int]],
    class_visibility: dict[int, bool],
) -> tuple[dict[int, list[int]], dict[int, bool]]:
    visible_classes = sorted(train_labels.keys())

    for index, tid in enumerate(visible_classes):
        count = class_counts.get(tid, 0)
        pct = (count / num_points * 100) if num_points else 0.0
        color_hex = _rgb_to_hex(class_colors.get(tid, [160, 160, 160]))
        is_active = class_visibility.get(tid, True)
        row_class = "seg-class-row" if is_active else "seg-class-row inactive"
        display_name = _format_class_display_name(train_labels[tid])
        info_col, toggle_col, picker_col = st.columns(
            [2.12, 0.34, 0.40],
            gap="small",
            vertical_alignment="center",
        )

        info_col.markdown(
            f"""
            <div class="{row_class}">
              <div class="seg-class-title">
                <span class="seg-class-dot" style="background:{color_hex};"></span>
                <span>{display_name}</span>
              </div>
              <div class="seg-class-meta">{count:,} puntos · {pct:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        toggle_col.toggle(
            f"Mostrar {display_name}",
            key=f"seg_class_visible_{tid}",
            label_visibility="collapsed",
        )
        picker_col.color_picker(
            f"Color {display_name}",
            key=f"seg_class_color_{tid}",
            label_visibility="collapsed",
        )
        if index < len(visible_classes) - 1:
            st.markdown('<div class="seg-class-divider"></div>', unsafe_allow_html=True)

    return (
        _get_segmentation_class_colors_from_state(train_labels),
        _get_segmentation_class_visibility_from_state(train_labels),
    )


def _get_segmentation_class_colors(config: SegmentationConfig) -> dict[int, list[int]]:
    default_colors = config.train_id_colors
    signature = tuple(
        (
            tid,
            config.train_id_labels[tid],
            tuple(default_colors.get(tid, [160, 160, 160])),
        )
        for tid in sorted(config.train_id_labels.keys())
    )

    if st.session_state.get("seg_class_color_signature") != signature:
        for tid in sorted(config.train_id_labels.keys()):
            st.session_state[f"seg_class_color_{tid}"] = _rgb_to_hex(
                default_colors.get(tid, [160, 160, 160])
            )
        st.session_state["seg_class_color_signature"] = signature

    for tid in sorted(config.train_id_labels.keys()):
        key = f"seg_class_color_{tid}"
        st.session_state.setdefault(
            key,
            _rgb_to_hex(default_colors.get(tid, [160, 160, 160])),
        )
    return _get_segmentation_class_colors_from_state(config.train_id_labels)


def _get_segmentation_class_visibility(config: SegmentationConfig) -> dict[int, bool]:
    signature = tuple(
        (tid, config.train_id_labels[tid])
        for tid in sorted(config.train_id_labels.keys())
    )

    if st.session_state.get("seg_class_visibility_signature") != signature:
        for tid in sorted(config.train_id_labels.keys()):
            st.session_state[f"seg_class_visible_{tid}"] = True
        st.session_state["seg_class_visibility_signature"] = signature

    for tid in sorted(config.train_id_labels.keys()):
        key = f"seg_class_visible_{tid}"
        st.session_state.setdefault(key, True)
    return _get_segmentation_class_visibility_from_state(config.train_id_labels)


def _get_segmentation_class_colors_from_state(train_labels: dict[int, str]) -> dict[int, list[int]]:
    resolved_colors: dict[int, list[int]] = {}
    for tid in sorted(train_labels.keys()):
        resolved_colors[tid] = _hex_to_rgb(
            st.session_state.get(f"seg_class_color_{tid}", "#a0a0a0")
        )
    return resolved_colors


def _get_segmentation_class_visibility_from_state(train_labels: dict[int, str]) -> dict[int, bool]:
    resolved_visibility: dict[int, bool] = {}
    for tid in sorted(train_labels.keys()):
        resolved_visibility[tid] = bool(
            st.session_state.get(f"seg_class_visible_{tid}", True)
        )
    return resolved_visibility


def _format_class_display_name(name: str) -> str:
    return str(name).replace("_", " ").title()


def _rgb_to_hex(rgb: list[int]) -> str:
    red, green, blue = (max(0, min(int(value), 255)) for value in rgb[:3])
    return f"#{red:02x}{green:02x}{blue:02x}"


def _hex_to_rgb(color: str) -> list[int]:
    cleaned = str(color).strip().lstrip("#")
    if len(cleaned) != 6:
        return [160, 160, 160]
    try:
        return [int(cleaned[index:index + 2], 16) for index in (0, 2, 4)]
    except ValueError:
        return [160, 160, 160]


def _resolve_pcd_colors(preview: PCDPreviewData, *, color_mode: str) -> tuple[np.ndarray, str]:
    if color_mode == "Intensidad" and preview.intensity is not None:
        return preview.intensity, "Intensidad"
    return preview.xyz[:, 2], "Z"


def _resolve_background_palette(background_mode: str) -> dict[str, str]:
    if background_mode == "Claro":
        return {
            "paper": "#f8fafc",
            "scene": "#ffffff",
            "font": "#0f172a",
            "grid": "#cbd5e1",
            "accent": "#0f766e",
        }
    return {
        "paper": "#0f172a",
        "scene": "#111827",
        "font": "#e5e7eb",
        "grid": "#334155",
        "accent": "#f59e0b",
    }


def _render_metadata_page(output_state: OutputBrowserState) -> None:
    st.markdown("#### :material/description: Metadata y Logs")
    if not output_state.root_raw:
        st.info("Defina un directorio de output en el panel lateral para inspeccionar metadata.")
        return

    if not output_state.export_dirs:
        st.info("No se encontraron exportaciones con archivos .pcd.")
        return

    selected_export_dir = _render_export_directory_selector(
        output_state,
        widget_key="metadata_export_selector",
        label="Exportacion",
    )
    details = _load_export_details(selected_export_dir)

    if details.metadata_error:
        st.error(details.metadata_error)
        return

    if details.metadata is None:
        st.warning("La exportacion seleccionada no contiene metadata.json.")
        return

    metadata = details.metadata
    error_rows = _build_error_rows(metadata.get("errors", []))
    real_error_count = sum(1 for row in error_rows if row["categoria"] == "Error")
    omitted_count = sum(1 for row in error_rows if row["categoria"] == "Omitido")

    cards = [
        {
            "label": "Mensajes",
            "value": str(metadata.get("total_messages", "-")),
            "detail": "Reportados en metadata.json",
            "tone": "active",
        },
        {
            "label": "PCD generados",
            "value": str(metadata.get("pcd_generated", "-")),
            "detail": "Exportados en esa corrida",
            "tone": "ready",
        },
        {
            "label": "Omitidos",
            "value": str(metadata.get("skipped_empty_frames", omitted_count)),
            "detail": "Frames vacios o descartados",
            "tone": "warn" if omitted_count else "pending",
        },
        {
            "label": "Errores",
            "value": str(real_error_count),
            "detail": "Errores reales registrados",
            "tone": "warn" if real_error_count else "ready",
        },
    ]
    _render_stat_cards(cards)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Acciones ──
    action_col1, action_col2, _ = st.columns([1, 1, 2])
    if details.metadata_path is not None:
        action_col1.download_button(
            ":material/download: metadata.json",
            data=details.metadata_path.read_bytes(),
            file_name=details.metadata_path.name,
            mime="application/json",
            use_container_width=True,
        )
    if action_col2.button(
        ":material/verified: Validar",
        use_container_width=True,
        key="validate_export_button",
    ):
        _run_validation_for_export(details)

    _render_validation_panel(details)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Detalles en dos columnas ──
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        with st.expander(":material/source: Resumen de origen", expanded=True):
            st.json(
                {
                    "source": metadata.get("source"),
                    "topic": metadata.get("topic"),
                    "detected_fields": metadata.get("detected_fields"),
                    "had_intensity": metadata.get("had_intensity"),
                }
            )

    with info_col2:
        with st.expander(":material/schedule: Timestamps", expanded=True):
            st.json(
                {
                    "timestamp_initial": metadata.get("timestamp_initial"),
                    "timestamp_final": metadata.get("timestamp_final"),
                    "header_timestamp_initial": metadata.get("header_timestamp_initial"),
                    "header_timestamp_final": metadata.get("header_timestamp_final"),
                }
            )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Logs por frame ──
    st.markdown("##### :material/list_alt: Logs por frame")
    if error_rows:
        show_only_real_errors = st.checkbox(
            "Solo errores reales",
            key="metadata_show_only_real_errors",
            help="Oculta frames omitidos y muestra solo errores de parsing/escritura.",
        )
        rows_to_show = (
            [row for row in error_rows if row["categoria"] == "Error"]
            if show_only_real_errors
            else error_rows
        )
        st.dataframe(
            rows_to_show,
            use_container_width=True,
            height=min(420, 42 * (len(rows_to_show) + 1)),
        )
    else:
        st.success("No hay errores registrados en metadata.json.")

    with st.expander(":material/data_object: metadata.json completo"):
        st.json(metadata)


def _render_validation_panel(details: ExportDirectoryDetails) -> None:
    validation = st.session_state["validation_cache"].get(str(details.export_dir.resolve()))
    if not validation:
        st.caption("Todavia no se ha ejecutado una validacion rapida de esta exportacion en la sesion actual.")
        return

    if validation.get("valid"):
        st.success("Validacion rapida completada.")
    else:
        st.warning(f"Validacion incompleta: {validation.get('error', 'sin detalle')}")
    st.json(validation)


def _run_validation_for_export(details: ExportDirectoryDetails) -> None:
    try:
        result = validate_export_directory(
            details.export_dir,
            sample_size=min(max(len(details.pcd_files), 1), 3),
        )
    except Exception as exc:
        result = {"valid": False, "error": str(exc)}
    st.session_state["validation_cache"][str(details.export_dir.resolve())] = result


def _build_error_rows(errors: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    frame_pattern = re.compile(r"^Frame (\d+): (.*)$")

    for entry in errors:
        match = frame_pattern.match(entry)
        if match:
            frame = int(match.group(1))
            message = match.group(2)
            category = "Omitido" if "omitido" in message.lower() else "Error"
            rows.append(
                {
                    "frame": frame,
                    "categoria": category,
                    "mensaje": message,
                }
            )
            continue

        rows.append(
            {
                "frame": None,
                "categoria": "Resumen",
                "mensaje": entry,
            }
        )
    return rows


def _format_datetime(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def _format_xyz(vector: np.ndarray) -> str:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    return ", ".join(f"{value:.3f}" for value in values[:3])


def _render_stat_cards(cards: list[dict[str, str]]) -> None:
    if not cards:
        return

    columns = st.columns(len(cards))
    for column, card in zip(columns, cards):
        tone = card.get("tone", "pending")
        label = card.get("label", "")
        value = card.get("value", "-")
        detail = card.get("detail", "")
        with column:
            st.markdown(
                f'<div class="stat-card tone-{tone}">'
                f'<div class="label">{label}</div>'
                f'<div class="value">{value}</div>'
                f'<div class="detail">{detail}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
