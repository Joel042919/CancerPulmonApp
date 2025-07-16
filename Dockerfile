#################################################################
#  Dockerfile para  detectar-cancer-pulmon  (Streamlit + Torch) #
#################################################################

# ---------- Imagen base ----------
FROM python:3.10-slim           # pequeñita, compatible con SimpleITK

# ---------- Ajustes de entorno ----------
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# ---------- Paquetes del sistema ----------
# wkhtmltopdf + Git LFS + dependencias mínimas de Qt y OpenGL
RUN apt-get update && apt-get install -y --no-install-recommends \
      wkhtmltopdf \
      git git-lfs \
      libglib2.0-0 libgl1 libxext6 libxrender1 \
   && rm -rf /var/lib/apt/lists/*

RUN git lfs install --system

# ---------- Workspace ----------
WORKDIR /app
COPY requirements.txt ./

# ---------- Python deps ----------
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# ---------- Copia del proyecto ----------
COPY . .

# ---------- Descarga de pesos gestionados por LFS ----------
RUN git lfs pull

# ---------- Exponer puerto ----------
EXPOSE 8501      # Render sobrescribe, pero es buena práctica

# ---------- Arranque ----------
#  Usamos la forma “shell” para que $PORT se expanda dentro del contenedor
CMD streamlit run app/main.py \
      --server.headless true \
      --server.port $PORT \
      --server.address 0.0.0.0
