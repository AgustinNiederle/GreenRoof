import streamlit as st
import rasterio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image
from google.cloud import storage
import tempfile
import os
import requests
import cv2
import numpy as np
from matplotlib.patches import Rectangle
from ultralytics import YOLO

# Configuración de Google Cloud Storage
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/Agustin/Desktop/2024/GreenRoof/utopian-honor-438417-u7-5b7f84fcfd25.json"
bucket_name = "greenroofdatalake"
folder_path = "MapasTIFF/1991_2020/"
csv_path = "ciudades/NewYork.csv"
folder_huerta_path = "huerta"
csv_huerta_path = "tabla_completa_climas_cultivos_koppen.csv"

# Configuración de la API de Google Maps
API_KEY = "AIzaSyDDt1fiH2cTopWMX_qfg50nm0taKg4egV4"
EARTH_CIRCUMFERENCE = 40075017  # Circunferencia de la Tierra en metros

model = YOLO('https://hub.ultralytics.com/models/49SdOOhcyS5TI2IdsBql') 

# Diccionario de clasificación Köppen-Geiger
koppen_descriptions = {
    "Af": "Tropical, rainforest",
    "Am": "Tropical, monsoon",
    "Aw": "Tropical, savannah",
    "BWh": "Arid, desert, hot",
    "BWk": "Arid, desert, cold",
    "BSh": "Arid, steppe, hot",
    "BSk": "Arid, steppe, cold",
    "Csa": "Temperate, dry summer, hot summer",
    "Csb": "Temperate, dry summer, warm summer",
    "Csc": "Temperate, dry summer, cold summer",
    "Cwa": "Temperate, dry winter, hot summer",
    "Cwb": "Temperate, dry winter, warm summer",
    "Cwc": "Temperate, dry winter, cold summer",
    "Cfa": "Temperate, no dry season, hot summer",
    "Cfb": "Temperate, no dry season, warm summer",
    "Cfc": "Temperate, no dry season, cold summer",
    "Dsa": "Cold, dry summer, hot summer",
    "Dsb": "Cold, dry summer, warm summer",
    "Dsc": "Cold, dry summer, cold summer",
    "Dsd": "Cold, dry summer, very cold winter",
    "Dwa": "Cold, dry winter, hot summer",
    "Dwb": "Cold, dry winter, warm summer",
    "Dwc": "Cold, dry winter, cold summer",
    "Dwd": "Cold, dry winter, very cold winter",
    "Dfa": "Cold, no dry season, hot summer",
    "Dfb": "Cold, no dry season, warm summer",
    "Dfc": "Cold, no dry season, cold summer",
    "Dfd": "Cold, no dry season, very cold winter",
    "ET": "Polar, tundra",
    "EF": "Polar, frost"
}

koppen_mapping = {
    1: "Af",
    2: "Am",
    3: "Aw",
    4: "BWh",
    5: "BWk",
    6: "BSh",
    7: "BSk",
    8: "Csa",
    9: "Csb",
    10: "Csc",
    11: "Cwa",
    12: "Cwb",
    13: "Cwc",
    14: "Cfa",
    15: "Cfb",
    16: "Cfc",
    17: "Dsa",
    18: "Dsb",
    19: "Dsc",
    20: "Dsd",
    21: "Dwa",
    22: "Dwb",
    23: "Dwc",
    24: "Dwd",
    25: "Dfa",
    26: "Dfb",
    27: "Dfc",
    28: "Dfd",
    29: "ET",
    30: "EF"
}

def clean_temp_files(temp_dir="/tmp"):
    """Elimina archivos temporales en el directorio especificado."""
    for file_name in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                print(f"Deleted: {file_name}")
        except Exception as e:
            print(f"Error deleting {file_name}: {e}")

def list_files_in_bucket(bucket_name, folder_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder_path)
    return [blob.name for blob in blobs if blob.name.endswith(".tif")]

def download_file_from_bucket(bucket_name, blob_name):
    """Descarga un archivo de Google Cloud Storage a un directorio temporal."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    temp_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
    blob.download_to_filename(temp_file_path)
    return temp_file_path

def get_koppen_classification(lat, lon):
    """Obtiene la clasificación Köppen-Geiger para una ubicación específica."""
    tiff_files = list_files_in_bucket(bucket_name, folder_path)
    for tiff_file in tiff_files:
        temp_file = download_file_from_bucket(bucket_name, tiff_file)
        with rasterio.open(temp_file) as dataset:
            try:
                row, col = dataset.index(lon, lat)
                koppen_value = dataset.read(1)[row, col]
                koppen_key = koppen_mapping.get(koppen_value)
                if koppen_key:
                    return koppen_key, temp_file
            except IndexError:
                continue
    return None, None

def plot_tiff_with_point(tiff_path, lat, lon, pixel_window_size=100):
    """Genera un gráfico que incluye un punto específico sobre un archivo TIFF."""
    with rasterio.open(tiff_path) as dataset:
        row, col = dataset.index(lon, lat)
        window = rasterio.windows.Window(
            col - pixel_window_size // 2, 
            row - pixel_window_size // 2, 
            pixel_window_size, 
            pixel_window_size
        )
        data = dataset.read(1, window=window)
        window_transform = dataset.window_transform(window)

        extent = (
            window_transform.c,
            window_transform.c + window_transform.a * data.shape[1],
            window_transform.f + window_transform.e * data.shape[0],
            window_transform.f
        )

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(data, cmap="terrain", extent=extent)
        ax.scatter([lon], [lat], color="red", label="Ubicación")
        ax.legend()
        plt.xlabel("Longitud")
        plt.ylabel("Latitud")
        plt.title("Mapa Köppen-Geiger")
        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        plt.savefig(temp_img, dpi=300)
        plt.close(fig)
        return Image.open(temp_img)

def read_csv_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        df["Anio"] = pd.to_datetime(df["Anio"], format="%Y", errors="coerce")
        return df
    except Exception as e:
        st.error(f"Error al leer el archivo CSV: {e}")
        return None

def download_csv_from_bucket(bucket_name, csv_path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=csv_path)
    for blob in blobs:
        if blob.name.endswith(".csv"):
            temp_csv_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
            blob.download_to_filename(temp_csv_path)
            return temp_csv_path
    return None

def plot_climatic_data_with_plotly(df, lat, lon):
    
    ultimo_anio = df["Anio"].dt.year.max()
    df_ultimo_anio = df[df["Anio"].dt.year == ultimo_anio]

    if df_ultimo_anio.empty:
        st.warning(f"No hay datos disponibles para el último año registrado: {ultimo_anio}")
        return

    st.subheader(f"Gráficos Interactivos de Variables Climáticas para el año {ultimo_anio}")

    fig_humedad = px.line(df_ultimo_anio, x="Dia", y="Humedad", 
                          title="Humedad (%)", markers=True)
    st.plotly_chart(fig_humedad, use_container_width=True)

    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=df_ultimo_anio["Dia"], y=df_ultimo_anio["Temperatura_2M_MAX"], 
                                  mode='lines+markers', name='Temp. Máxima (°C)', line=dict(color='red')))
    fig_temp.add_trace(go.Scatter(x=df_ultimo_anio["Dia"], y=df_ultimo_anio["Temperatura_2M_MIN"], 
                                  mode='lines+markers', name='Temp. Mínima (°C)', line=dict(color='blue')))
    fig_temp.update_layout(title="Temperatura Máxima y Mínima (°C)", xaxis_title="Día", yaxis_title="Temperatura (°C)")
    st.plotly_chart(fig_temp, use_container_width=True)

    fig_radiacion = px.bar(df_ultimo_anio, x="Dia", y="Irradiacion_de_onda_corta", 
                           title="Radiación Solar (kWh/m²)")
    st.plotly_chart(fig_radiacion, use_container_width=True)

    fig_precipitacion = px.bar(df_ultimo_anio, x="Dia", y="Precipitaciones_promedio", 
                               title="Precipitación (mm)")
    st.plotly_chart(fig_precipitacion, use_container_width=True)

def get_cultivo_recomendaciones(bucket_name, folder_path="huerta", file_name="tabla_completa_climas_cultivos_koppen.csv"):
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob_path = f"{folder_path}/{file_name}"
        blob = bucket.blob(blob_path)

        if not blob.exists():
            st.error(f"El archivo {file_name} no existe en el bucket.")
            return None
        
        temp_csv_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
        blob.download_to_filename(temp_csv_path)
        df_cultivos = pd.read_csv(temp_csv_path)
        return df_cultivos
    except Exception as e:
        st.error(f"Error al descargar o leer el archivo de cultivos: {e}")
        return None

def obtener_recomendaciones_por_koppen(df_cultivos, koppen_key):
    try:
        recomendaciones = df_cultivos[df_cultivos["Koppen"] == koppen_key]

        if recomendaciones.empty:
            return {"Frutas": "Sin datos", "Verduras": "Sin datos", "Consejos": "Sin datos"}
        
        frutas = recomendaciones["Frutas"].iloc[0]
        verduras = recomendaciones["Verduras"].iloc[0]
        consejos = recomendaciones["Consejos"].iloc[0]
        
        return {"Frutas": frutas, "Verduras": verduras, "Consejos": consejos}
    except Exception as e:
        st.error(f"Error al obtener recomendaciones: {e}")
        return {"Frutas": "Error", "Verduras": "Error", "Consejos":  "Error"}

# Función para calcular la escala (metros por píxel)
def calculate_scale(zoom, image_width):
    return EARTH_CIRCUMFERENCE / (2**zoom * image_width)

# Función para descargar imágenes de Google Maps
def download_google_maps_image(lat, lon, zoom=17, size="640x640"):
    base_url = "https://maps.googleapis.com/maps/api/staticmap?"
    params = {
        "center": f"{lat},{lon}",
        "zoom": zoom,
        "size": size,
        "key": API_KEY,
        "maptype": "satellite"
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    else:
        st.error("No se pudo descargar la imagen de Google Maps.")
        return None

# Función para procesar la imagen y calcular áreas (solo techos)
def analyze_image_with_yolo(image, confidence_threshold, scale):
    results = model(image, conf=confidence_threshold)
    boxes = results[0].boxes  # Coordenadas de las detecciones
    class_names = model.names  # Diccionario de clases: {0: "car", 1: "tree", 2: "roof"}

    if boxes is not None:
        detections = boxes.xyxy.numpy()
        confidences = boxes.conf.numpy()
        classes = boxes.cls.numpy()

        areas = []
        labels = []
        coords = []

        annotated_image = image.copy()
        for i, (box, conf, cls_id) in enumerate(zip(detections, confidences, classes)):
            class_name = class_names[int(cls_id)]

            if class_name == "roof":
                width = (box[2] - box[0]) * scale
                height = (box[3] - box[1]) * scale
                area = width * height
                areas.append(area)
                coords.append(box)
                label = f"Techo {len(labels) + 1}"
                labels.append(label)

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 0, 255), 2, cv2.LINE_AA)

        if labels:
            df_results = pd.DataFrame({
                "label": labels,
                "x1": [c[0] for c in coords],
                "y1": [c[1] for c in coords],
                "x2": [c[2] for c in coords],
                "y2": [c[3] for c in coords],
                "area_m2": areas
            })
            return annotated_image, df_results, image
    return image, None, None

# Función para calcular distribución de macetas optimizada
def calculate_optimized_distribution(width, height, pattern="L"):
    pot_width, pot_height = 2, 1  # Tamaño de macetas
    rows = int(height // pot_height)
    cols = int(width // pot_width)
    total_pots = 0

    if pattern == "L":
        # Varias L en paralelo
        pots_per_L = (cols + rows - 1)  # Una sola L
        total_pots = pots_per_L * (rows // 3)  # Colocamos L cada 3 filas
    elif pattern == "M":
        # Intercalar filas de macetas con pasillos
        total_pots = (rows // 2) * cols  # Cada 2 filas hay un pasillo

    coverage_area = total_pots * (pot_width * pot_height)
    return total_pots, coverage_area

# Función para graficar distribución optimizada
def plot_optimized_pattern(width, height, pattern="L"):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.set_title(f"Distribución de Macetas en Patrón {pattern}")
    ax.set_xlabel("Metros")
    ax.set_ylabel("Metros")

    pot_width, pot_height = 2, 1  # Tamaño de macetas

    if pattern == "L":
        # Varias L en paralelo
        for k in range(0, int(height // pot_height), 3):  # Cada 3 filas una L
            for i in range(0, int(height // pot_height)):
                ax.add_patch(Rectangle((0, i * pot_height), pot_width, pot_height, facecolor="green"))
            for j in range(1, int(width // pot_width)):
                ax.add_patch(Rectangle((j * pot_width, k * pot_height), pot_width, pot_height, facecolor="green"))

    elif pattern == "M":
        # Intercalar filas de macetas con pasillos
        for i in range(0, int(height // pot_height), 2):  # Espacios cada 2 filas
            for j in range(0, int(width // pot_width)):
                ax.add_patch(Rectangle((j * pot_width, i * pot_height), pot_width, pot_height, facecolor="green"))

    return fig


# Streamlit App Principal
logo_greenroof = Image.open('C:/Users/Agustin/Desktop/2024/GreenRoof/logos/Green_Roof.png')
with st.sidebar:
    st.image(logo_greenroof)

st.title("Bienvenido a GreenRoof")
st.subheader("Elija con qué quiere comenzar")

st.sidebar.subheader("Sus Coordenadas")
lat = st.sidebar.number_input("Latitud:", format="%.6f", value=40.673429)
lon = st.sidebar.number_input("Longitud:", format="%.6f", value=-73.862674)

if st.sidebar.button("Su clasificación Koppen Geigen y recomendaciones según clima"):
    koppen_key, tiff_path = get_koppen_classification(lat, lon)
    if koppen_key:
        st.write(f"**Clasificación Köppen-Geiger:** {koppen_key}")
        st.write(f"**Descripción:** {koppen_descriptions.get(koppen_key, 'Descripción no disponible')}")

        map_image = plot_tiff_with_point(tiff_path, lat, lon)
        st.image(map_image, caption="Mapa Köppen-Geiger", use_container_width=True)
        os.remove(tiff_path)

        csv_path = download_csv_from_bucket(bucket_name, csv_path)
        if csv_path:
            df_climatic = read_csv_data(csv_path)
            if df_climatic is not None:
                plot_climatic_data_with_plotly(df_climatic, lat, lon)
            os.remove(csv_path)

        csv_huerta_path = get_cultivo_recomendaciones(bucket_name, folder_huerta_path)
       
        if csv_huerta_path is not None:
            recomendaciones = obtener_recomendaciones_por_koppen(csv_huerta_path, koppen_key)
            st.subheader("Recomendaciones de Cultivos")
            st.write(f"**Frutas recomendadas:** {recomendaciones['Frutas']}")
            st.write(f"**Verduras recomendadas:** {recomendaciones['Verduras']}")
            st.write(f"**Consejos:** {recomendaciones['Consejos']}")
    else:
        st.error("No se encontró una clasificación para las coordenadas proporcionadas.")

# Entrada única de coordenadas
st.sidebar.subheader("Coordenadas para planear su huerta en un techo")
lat = st.sidebar.number_input("Latitud:", format="%.6f", value=40.673429, key="lat_input")
lon = st.sidebar.number_input("Longitud:", format="%.6f", value=-73.862674, key="lon_input")
zoom = st.sidebar.slider("Zoom del mapa (17-20):", min_value=17, max_value=20, value=19)
confidence_threshold = st.sidebar.slider("Umbral de Confianza:", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

# Inicializar variables en el estado de la sesión
if "yolo_results" not in st.session_state:
    st.session_state["yolo_results"] = None
if "selected_label" not in st.session_state:
    st.session_state["selected_label"] = None
if "scale" not in st.session_state:
    st.session_state["scale"] = None

if st.sidebar.button("Ver techos disponibles y elegir el suyo"):
    # Clasificación Köppen-Geiger
    koppen_key, tiff_path = get_koppen_classification(lat, lon)
    if koppen_key:
        st.subheader("Clasificación Köppen-Geiger")
        st.write(f"**Clasificación:** {koppen_key}")
        st.write(f"**Descripción:** {koppen_descriptions.get(koppen_key, 'No disponible')}")

        map_image = plot_tiff_with_point(tiff_path, lat, lon)
        st.image(map_image, caption="Mapa Köppen-Geiger", use_container_width=True)
        os.remove(tiff_path)

    # Análisis de techos y huertas
    st.subheader("Optimización de Techos y Huertas")
    map_image = download_google_maps_image(lat, lon, zoom)
    scale = calculate_scale(zoom, 640)
    st.session_state["scale"] = scale  # Guardar el valor de escala en el estado
    annotated_image, yolo_results, _ = analyze_image_with_yolo(map_image, confidence_threshold, scale)

    if yolo_results is not None:
        st.session_state["yolo_results"] = yolo_results  # Guardar resultados en el estado
        st.image(annotated_image, caption="Detecciones de Techos")
        st.dataframe(yolo_results)
    else:
        st.warning("No se detectaron techos en la imagen.")
        st.session_state["yolo_results"] = None

# Mostrar resultados basados en la selección del techo
if st.session_state["yolo_results"] is not None:
    st.subheader("Análisis de Techos Seleccionados")
    selected_label = st.selectbox(
        "Seleccione un techo:", 
        st.session_state["yolo_results"]["label"].tolist(), 
        index=st.session_state["yolo_results"]["label"].tolist().index(st.session_state["selected_label"]) 
        if st.session_state["selected_label"] in st.session_state["yolo_results"]["label"].tolist() else 0
    )
    st.session_state["selected_label"] = selected_label  # Guardar selección en el estado

    selected_row = st.session_state["yolo_results"][
        st.session_state["yolo_results"]["label"] == selected_label
    ]

    if not selected_row.empty:
        # Usar la escala guardada en el estado
        scale = st.session_state["scale"]
        width = (selected_row["x2"].values[0] - selected_row["x1"].values[0]) * scale
        height = (selected_row["y2"].values[0] - selected_row["y1"].values[0]) * scale
        area_m2 = selected_row["area_m2"].values[0]

        st.write(f"Área del techo seleccionado: {area_m2:.2f} m²")
        pattern = st.radio("Seleccione el patrón optimizado:", ("L", "M"))
        total_pots, coverage_area = calculate_optimized_distribution(width, height, pattern)
        coverage_percentage = (coverage_area / area_m2) * 100
        st.write(f"- Macetas Totales: {total_pots}")
        st.write(f"- Área Cubierta: {coverage_area:.2f} m² ({coverage_percentage:.2f}%)")
        fig = plot_optimized_pattern(width, height, pattern)
        st.pyplot(fig)
    else:
        st.warning("No se encontraron datos para el techo seleccionado.")
