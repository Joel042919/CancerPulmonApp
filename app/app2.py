import os, sys
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Hacer que Python vea la carpeta padre para importar ml/
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Inferencia 3D
try:
    from model_utils_pt import load_models, predict_volume, generate_saliency_map
except ImportError:
    from model_utils_tf import load_models, predict_volume, generate_saliency_map
from preprocessing import load_and_preprocess_ct_scan

# Evaluación y optimización real usando ml/
from ml.evaluate import evaluate_models
from ml.hyperparameter_tuning import tune_hyperparameters

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Diagnóstico de Cáncer de Pulmón",
    page_icon="🏥",
    layout="wide"
)

# Título
st.title("Sistema Inteligente de Diagnóstico de Cáncer de Pulmón")
st.markdown("""
Este sistema utiliza redes neuronales 3D para analizar tomografías computarizadas
y detectar nódulos pulmonares sospechosos.
""")

# Cargar modelos en caché
def loaded_models():
    return load_models()
models = st.cache_resource(loaded_models)()
model_names = list(models.keys())

# Sidebar: selección de modelo y umbral
st.sidebar.header("Configuración")
selected_model = st.sidebar.selectbox("Modelo a utilizar", model_names)
confidence_threshold = st.sidebar.slider(
    "Umbral de confianza", 0.5, 0.99, 0.85, 0.01
)

# Carga de un caso individual
st.header("Carga de Tomografía Computarizada")
upload_option = st.radio(
    "Seleccione el modo de entrada",
    ["Subir archivo DICOM", "Usar ejemplo"]
)

uploaded_file = None
if upload_option == "Subir archivo DICOM":
    uploaded_file = st.file_uploader(
        "Suba un .zip o carpeta DICOM", type=["zip", "dcm"]
    )
else:
    # Ruta al dataset separado (data/benign, data/malignant)
    DATA_ROOT = st.text_input(
        "Ruta al directorio de ejemplo (contiene benign/ y malignant/):",
        value=os.path.join(ROOT_DIR, 'data')
    )
    if not os.path.isdir(DATA_ROOT):
        st.warning(f"No encontré el directorio: {DATA_ROOT}")

# Procesamiento de un único caso
if uploaded_file:
    with st.spinner("Procesando tomografía..."):
        volume, original = load_and_preprocess_ct_scan(uploaded_file)
    slice_idx = st.slider("Corte axial", 0, volume.shape[0]-1, volume.shape[0]//2)
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    ax[0].imshow(original[slice_idx], cmap='gray'); ax[0].axis('off'); ax[0].set_title('Original')
    ax[1].imshow(volume[slice_idx], cmap='gray'); ax[1].axis('off'); ax[1].set_title('Preprocesado')
    st.pyplot(fig)

    st.header("Resultados de Diagnóstico")
    model = models[selected_model]
    with st.spinner("Analizando..."):
        pred, conf, heatmap = predict_volume(model, volume)
    col1, col2, col3 = st.columns(3)
    col1.metric("Modelo", selected_model)
    col2.metric("Predicción", "Positivo" if pred==1 else "Negativo")
    col3.metric("Confianza", f"{conf:.2%}")

    if conf < confidence_threshold:
        st.warning("Confianza baja. Recomendado revisar con radiólogo.")
    else:
        st.success("Nódulos no sospechosos.") if pred==0 else st.error("Nódulos sospechosos: consultar especialista.")

    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(original[slice_idx], cmap='gray'); ax.imshow(heatmap[slice_idx], alpha=0.5, cmap='jet'); ax.axis('off')
    st.pyplot(fig)

# Función de comparación de modelos usando datos reales

def show_model_comparison(data_root):
    st.subheader("Comparación de Modelos (evaluación real)")
    with st.spinner("Evaluando modelos en dataset de prueba..."):
        df = evaluate_models(data_root)
    # Mostrar tabla
    st.dataframe(df.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='#ffcccb'))
    # Gráficos
    fig1 = px.bar(df, x='Modelo', y=['Sensibilidad','Especificidad'], barmode='group')
    st.plotly_chart(fig1, use_column_width=True)
    fig2 = px.scatter(df, x='AP', y='AUC-ROC', color='Modelo', size='Precisión', title='AP vs AUC-ROC')
    st.plotly_chart(fig2, use_column_width=True)
    fig3 = px.line_polar(df, r='Dice Score', theta='Modelo', line_close=True, title='Dice Score')
    st.plotly_chart(fig3, use_column_width=True)

# Botón para comparar
if 'DATA_ROOT' in globals() and os.path.isdir(DATA_ROOT):
    if st.button("Mostrar Métricas Comparativas"):
        show_model_comparison(DATA_ROOT)

# Botón para optimización de hiperparámetros
if 'DATA_ROOT' in globals() and os.path.isdir(DATA_ROOT):
    if st.button("Ejecutar Búsqueda de Hiperparámetros"):
        st.subheader("Optimización de Hiperparámetros")
        with st.spinner("Buscando mejores parámetros..."):
            best_hps, _ = tune_hyperparameters(DATA_ROOT)
        st.json(best_hps.values)

if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT_DIR,'models'), exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR,'data','examples'), exist_ok=True)
