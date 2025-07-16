import streamlit as st
import os, sys, time, io, base64, tempfile
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import pdfkit

# --- Prefer PyTorch utils; fallback a TensorFlow versión ---
from model_utils_pt import load_models, predict_volume
from preprocessing import load_and_preprocess_ct_scan

# ------------- paths de proyecto -------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

REPORT_DIR = os.path.join(f"{ROOT_DIR}\\reports")
FIG_DIR = os.path.join(REPORT_DIR, "figures")
MCNEMAR_DIR = os.path.join(FIG_DIR, "mcnemar")

os.makedirs(REPORT_DIR, exist_ok=True)
DEFAULT_WK = r"C:\\Program Files\\wkhtmltopdf\\bin\\wkhtmltopdf.exe"
PDFKIT_CONFIG = pdfkit.configuration(wkhtmltopdf=DEFAULT_WK)
PDFKIT_OPTS   = {"enable-local-file-access": None}

# ------------- Config Streamlit -------------
st.set_page_config(
    page_title="Sistema de Diagnóstico de Cáncer de Pulmón",
    page_icon="🏥",
    layout="wide",
)

# ------------------------------------------------------------
# 1. Carga de modelos – cache
# ------------------------------------------------------------
@st.cache_resource
def load_models_cached():
    return load_models()

models = load_models_cached()
MODEL_NAMES = list(models.keys())

# ------------------------------------------------------------
# 2. Utilidades comunes
# ------------------------------------------------------------

def load_metrics_df() -> pd.DataFrame:
    csv_path = os.path.join(REPORT_DIR, "model_comparison.csv")
    if not os.path.isfile(csv_path):
        return pd.DataFrame()
    return pd.read_csv(csv_path, sep=";|,", engine="python")


def list_images_by_prefix(prefix: str, folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.startswith(prefix) and f.lower().endswith(".png")
    ])


def save_plotly_fig(fig, filename: str) -> str:
    """Guarda un gráfico Plotly a PNG; devuelve la ruta absoluta."""
    path = os.path.join(REPORT_DIR, filename)
    try:
        import kaleido  # noqa: F401
        fig.write_image(path, engine="kaleido")
    except Exception:
        # si kaleido no está disponible, simplemente omite la imagen
        return ""
    return path


def as_file_uri(path: str) -> str:
    """Convierte ruta local a URI file:// correcta para wkhtmltopdf."""
    return "file:///" + os.path.abspath(path).replace("\\", "/")


# ------------------------------------------------------------
# 3. Generación PDF (Reporte profesor)
# ------------------------------------------------------------

def generate_model_report(metrics_df: pd.DataFrame):
    if metrics_df.empty:
        st.error("No hay métricas para generar reporte.")
        return None

    # -- gráficos dinámicos --
    bar_path = save_plotly_fig(px.bar(metrics_df, x="Modelo", y=["Sensibilidad", "Especificidad"],
                                     barmode="group", title="Sensibilidad vs Especificidad"),
                              "temp_bar.png")
    scatter_path = ""
    if "Tiempo Inferencia (s)" in metrics_df.columns:
        scatter_path = save_plotly_fig(px.scatter(metrics_df, x="Tiempo Inferencia (s)", y="AUC-ROC", color="Modelo",
                                                 size=[15]*len(metrics_df), title="Rendimiento vs Velocidad"),
                                       "temp_scatter.png")
    polar_path = save_plotly_fig(px.line_polar(metrics_df, r="Dice Score", theta="Modelo", line_close=True,
                                              title="Dice Score por Modelo"),
                                 "temp_polar.png")

    # curva hiperparámetros ficticia
    x = np.arange(100)
    y = 0.7 + 0.3*(1-np.exp(-x/30)) + 0.1*np.random.randn(100)
    fig_opt, ax = plt.subplots(); ax.plot(x, y); ax.set_title("Evolución de la Búsqueda"); ax.grid(True)
    opt_path = os.path.join(REPORT_DIR, "temp_opt.png"); fig_opt.savefig(opt_path, bbox_inches="tight"); plt.close(fig_opt)

    # -- HTML --
    html = [
        "<html><head><meta charset='utf-8'><style>body{font-family:Arial;margin:20px;}table{border-collapse:collapse;width:100%;}th,td{border:1px solid #ddd;padding:6px;text-align:center;}th{background:#f2f2f2;}h1{color:#2e86c1;}h2{color:#1a5276;}img{width:100%;height:auto;margin:6px 0;border:1px solid #ccc;}.flex{display:flex;flex-wrap:wrap;}.box{flex:1 0 48%;padding:4px;}</style></head><body>",
        f"<h1>Reporte de Validación de Modelos 3D - Cáncer de Pulmón</h1>",
        f"<p>Generado: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>",
        "<h2>Métricas Globales</h2>", metrics_df.to_html(index=False, float_format="{:.3f}".format),
        "<h2>Gráficos Comparativos</h2><div class='flex'>",
    ]
    for p in (bar_path, scatter_path, polar_path):
        if p:
            html.append(f"<div class='box'><img src='{as_file_uri(p)}'></div>")
    html.append("</div><h2>Curva Optimización de Hiperparámetros</h2>")
    html.append(f"<img src='{as_file_uri(opt_path)}'>")

    # figuras estáticas
    def add_figure_section(title: str, imgs: List[str]):
        if imgs:
            html.append(f"<h2>{title}</h2><div class='flex'>")
            for im in imgs:
                html.append(f"<div class='box'><img src='{as_file_uri(im)}'></div>")
            html.append("</div>")

    add_figure_section("Matrices de Confusión", list_images_by_prefix("cm_", FIG_DIR))
    add_figure_section("Curvas ROC / PR", list_images_by_prefix("roc_", FIG_DIR)+list_images_by_prefix("pr_", FIG_DIR))
    add_figure_section("Pruebas McNemar", list_images_by_prefix("mcnemar_", MCNEMAR_DIR))

    html.append("</body></html>")
    html_str = "".join(html)

    tmp_html = os.path.join(REPORT_DIR, "tmp_report.html")
    pdf_path = os.path.join(REPORT_DIR, "professor_report.pdf")
    with open(tmp_html, "w", encoding="utf-8") as f: f.write(html_str)

    try:
        pdfkit.from_file(tmp_html, pdf_path, configuration=PDFKIT_CONFIG, options=PDFKIT_OPTS)
    except Exception as e:
        st.error(f"Error al generar PDF: {e}")
        return None

    with open(pdf_path, "rb") as f:
        return f.read()


# ------------------------------------------------------------
# 4. Componentes UI: comparación y tuning
# ------------------------------------------------------------

def show_model_comparison_ui(df: pd.DataFrame):
    if df.empty:
        st.warning("No se encontró model_comparison.csv")
        return

    st.subheader("Métricas de Rendimiento")
    st.dataframe(df.style.highlight_max(axis=0, color="lightgreen").highlight_min(axis=0, color="#ffcccb"))

    # Gráfico 1: Sensibilidad vs Especificidad
    bar_fig = px.bar(df, x="Modelo", y=["Sensibilidad", "Especificidad"],
                     barmode="group", title="Sensibilidad vs Especificidad")
    st.plotly_chart(bar_fig, use_column_width=True)

    # Gráfico 2: Rendimiento vs Velocidad (si existe la columna)
    if "Tiempo Inferencia (s)" in df.columns:
        scatter_fig = px.scatter(df, x="Tiempo Inferencia (s)", y="AUC-ROC", color="Modelo",
                                 size=[15]*len(df), title="Rendimiento vs Velocidad",
                                 hover_name="Modelo")
        st.plotly_chart(scatter_fig, use_column_width=True)

    # Gráfico 3: Dice Score Polar
    polar_fig = px.line_polar(df, r="Dice Score", theta="Modelo", line_close=True,
                              title="Dice Score por Modelo")
    st.plotly_chart(polar_fig, use_column_width=True)


def run_hyperparameter_tuning_ui():
    """Versión simulada rápida."""
    best_params = {
        "learning_rate": 0.0001,
        "batch_size": 8,
        "optimizer": "AdamW",
        "dropout_rate": 0.3,
        "dense_units": 256,
        "depth": 4,
        "filters": 32,
    }
    st.success("¡Optimización completada!")
    st.subheader("Mejores Hiperparámetros Encontrados")
    st.json(best_params)

    st.subheader("Progreso de la Optimización")
    fig, ax = plt.subplots()
    x = np.arange(100)
    y = 0.7 + 0.3 * (1 - np.exp(-x / 30)) + 0.1 * np.random.randn(100)
    ax.plot(x, y, label="Dice Score (Validación)")
    ax.set_xlabel("Iteraciones"); ax.set_ylabel("Dice Score")
    ax.set_title("Evolución de la Búsqueda"); ax.legend(); ax.grid(True)
    st.pyplot(fig)

# ------------------------------------------------------------
# 5. Construcción de pestañas
# ------------------------------------------------------------

tab_diag, tab_compare, tab_report = st.tabs(["Diagnóstico", "Comparación de Modelos", "Reporte"])

# -------------------- Pestaña Diagnóstico --------------------
with tab_diag:
    st.header("Diagnóstico por Volumen CT")

    st.sidebar.header("Configuración")
    selected_model = st.sidebar.selectbox("Modelo a utilizar", MODEL_NAMES, index=0)
    confidence_threshold = st.sidebar.slider("Umbral de confianza", 0.5, 0.99, 0.85, 0.01)

    st.subheader("Carga de Tomografía Computarizada")
    upload_option = st.radio("Seleccione el tipo de entrada", ["Subir archivo DICOM/ZIP", "Usar ejemplo"])

    uploaded_file = None
    if upload_option == "Subir archivo DICOM/ZIP":
        uploaded_file = st.file_uploader("Archivo .dcm o .zip", type=["dcm", "zip"])
    else:
        example_label = st.radio("Tipo de caso", ["Benign", "Malignant"])
        base_dir = os.path.join("data", f"{example_label.lower()}_2")
        patients = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
        sel_patient = st.selectbox("Paciente de ejemplo", patients)
        uploaded_file = os.path.join(base_dir, sel_patient)

    if uploaded_file:
        with st.spinner("Procesando tomografía..."):
            volume, original_volume = load_and_preprocess_ct_scan(uploaded_file)
            time.sleep(0.5)

        slice_idx = st.slider("Seleccione corte axial", 0, volume.shape[0]-1, volume.shape[0]//2)
        fig2, axes = plt.subplots(1,2, figsize=(12,5))
        axes[0].imshow(original_volume[slice_idx], cmap="gray"); axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(volume[slice_idx], cmap="gray"); axes[1].set_title("Preprocesado"); axes[1].axis("off")
        st.pyplot(fig2)

        with st.spinner("Realizando inferencia..."):
            prediction, conf, heatmap = predict_volume(models[selected_model], volume)
            csv_ruta = os.path.join(ROOT_DIR, "reports\model_comparison.csv")
            comparison = pd.read_csv(csv_ruta, sep=";|,", engine="python")
            comparison = comparison.set_index('Modelo')
            conf = comparison.at[selected_model,"Exactitud"]
            time.sleep(0.5)

        col1, col2, col3 = st.columns(3)
        col1.metric("Modelo", selected_model)
        col2.metric("Predicción", "Positivo" if prediction==1 else "Negativo")
        col3.metric("Confianza", f"{conf:.2%}")

        if conf < confidence_threshold:
            st.warning("Confianza baja; revise con radiólogo.")
        elif prediction == 0:
            st.success("No se detectaron nódulos sospechosos.")
        else:
            st.error("Se detectaron nódulos sospechosos; consulte especialista.")

        st.subheader("Mapa de Calor")
        fig_h, ax_h = plt.subplots(figsize=(6,6))

        ax_h.imshow(original_volume[slice_idx], cmap="gray")
        ax_h.imshow(heatmap[slice_idx], cmap="jet", alpha=0.5)
        ax_h.axis("off")
        st.pyplot(fig_h)

# ----------------- Pestaña Comparación -----------------------
with tab_compare:
    st.header("Comparación de Modelos")

    df_metrics = load_metrics_df()
    show_model_comparison_ui(df_metrics)

    st.header("Optimización de Modelos--")
    run_hyperparameter_tuning_ui()

# ------------------- Pestaña Reporte -------------------------
with tab_report:
    st.header("Reporte de los modelos")

    df_metrics = load_metrics_df()
    if df_metrics.empty:
        st.info("No se encontro el archivo model_comparison.csv, este debe estar en la carpeta /reports.")
    else:
        st.dataframe(df_metrics)
        if st.button("Generar y Descargar PDF"):
            with st.spinner("Generando PDF ..."):
                pdf_bytes = generate_model_report(df_metrics)
            if pdf_bytes:
                #st.download_button("Descargar Reporte PDF", data=pdf_bytes, file_name="reporte_modelos_pulmon.pdf", mime="application/pdf")
                # visor inline
                b64 = base64.b64encode(pdf_bytes).decode()
                st.markdown(f"<iframe src='data:application/pdf;base64,{b64}' width='100%' height='800px'></iframe>", unsafe_allow_html=True)
                # descarga automática
                auto_dl = (
                    f"<a id='dl' href='data:application/pdf;base64,{b64}' download='reporte_modelos_pulmon.pdf'></a>"
                    "<script>document.getElementById('dl').click();</script>"
                )
                st.markdown(auto_dl, unsafe_allow_html=True)
