# ForestSeg

Aplicacion Streamlit para segmentacion semantica de nubes de puntos LiDAR.

La app permite:

- segmentar archivos `.pcd` usando un modelo ONNX
- convertir mensajes `sensor_msgs/PointCloud2` desde rosbag1 o rosbag2 a `.pcd` como paso previo
- explorar nubes exportadas y revisar metadata de ejecucion


## Requisitos

- Python 3.10 a 3.12 recomendado
- `pip`
- Un entorno virtual Python
- Modelos en `app/models/` si vas a usar la vista `Segmentar`
- Un rosbag de entrada solo si vas a usar la vista `Convertir`

Dependencias Python:

- `streamlit`
- `rosbags`
- `numpy`
- `open3d`
- `plotly`
- `pyyaml`
- `onnxruntime`

Todas estan declaradas en `app/requirements.txt`.

## Modelos Necesarios

La funcionalidad principal de segmentacion espera que `app/models/` contenga al menos:

- `arch_cfg.yaml`
- `data_cfg.yaml`
- un archivo `.onnx`
- si el modelo usa pesos externos, su archivo `.onnx.data` asociado


## Instalacion

Desde la raiz del proyecto:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r app/requirements.txt
```

## Iniciar La App

Con el entorno virtual activo:

```bash
streamlit run app/app.py
```

Streamlit imprimira una URL local en terminal, normalmente:

```text
http://localhost:8501
```

## Uso Rapido

1. Abre la app con `streamlit run app/app.py`.
2. En el sidebar define el directorio de output.
3. En `Segmentar`, selecciona un `.pcd` y un modelo ONNX disponible en `app/models/`.
4. Si partes desde un rosbag, usa `Convertir` para generar `.pcd`.
5. Usa `Explorar PCD` y `Metadata` para inspeccionar resultados y exportaciones.

## Salidas Y Carpetas

- `app/models/` se usa para configuraciones y modelos de segmentacion
- `app/output/` se usa para exportaciones `.pcd` y `metadata.json`

