# --- Base ---------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    APP_HOME=/LUNG_CANCER_DETECTION_APP

# --- System deps --------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget curl \
        libgl1 libglib2.0-0 libstdc++6 \
        libxext6 libxrender1 libx11-6 \
        xfonts-base xfonts-75dpi \
        wkhtmltopdf \
    && rm -rf /var/lib/apt/lists/*

# --- Python deps --------------------------------------------------------
WORKDIR ${APP_HOME}
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --timeout 100 -r requirements.txt

# --- App code -----------------------------------------------------------
COPY . .
# Crear enlace /reports → /LUNG_CANCER_DETECTION_APP/reports
RUN ln -s ${APP_HOME}/reports /reports

EXPOSE 8501
CMD ["streamlit", "run", "main.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
