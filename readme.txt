# 🩺 **Lung‑AI‑3D**
### _Evaluación comparativa de redes neuronales tridimensionales para la detección automática de cáncer pulmonar_

![Python 3.11.9](https://img.shields.io/badge/python-3.11.9-blue?logo=python)
![Streamlit 1.34+](https://img.shields.io/badge/streamlit-1.34%2B-red?logo=streamlit)
![Status beta](https://img.shields.io/badge/status-beta-yellow)

> **TL;DR**  
> Este repositorio contiene **todo el pipeline** — pre‑procesamiento, entrenamiento, evaluación y _dashboard_ web — para comparar **ResNet50 3D**, **DenseNet121 3D** y una **CNN 3D personalizada** en la clasificación de nódulos pulmonares (benignos vs malignos) usando tomografías computarizadas.

---

## 🗺️ Índice
1. [Motivación](#motivación)  
2. [Estructura del proyecto](#estructura-del-proyecto)  
3. [Datasets](#datasets)  
4. [Entorno de desarrollo](#entorno-de-desarrollo)  
5. [Instalación paso a paso](#instalación-paso-a-paso)  
6. [Uso rápido](#uso-rápido)  
7. [Comandos avanzados](#comandos-avanzados)  
8. [Resultados principales](#resultados-principales)  
9. [Créditos y licencias](#créditos-y-licencias)

---

## Motivación
- **Impacto clínico**  El cáncer de pulmón es la primera causa de mortalidad oncológica mundial. Diagnosticar nódulos pequeños requiere que un radiólogo revise ∼120 cortes DICOM por paciente: tarea tediosa y propensa a error.  
- **IA 3D**  Las CNN 3D analizan volúmenes completos, preservando el contexto espacial. ¿Qué arquitectura logra el mejor _trade‑off_ sensibilidad / especificidad usando solo CPU?  
- **Transparencia**  Se ofrece código reproducible, métricas, figuras y un dashboard **Streamlit** listo para demo.

---

## Estructura del proyecto
```text
LUNG_CANCER_DETECTION_APP/
├── newApp.py               # app Streamlit (Diagnóstico · Comparación · Reporte)
├── requirements.txt
├── reports/
│   ├── model_comparison.csv
│   └── figures/
│       ├── cm_*            # matrices de confusión
│       ├── roc_* , pr_*    # curvas ROC & PR
│       └── mcnemar/
├── models/                 # pesos .pth (se generan tras entrenamiento)
├── ml/
│   ├── train.py
│   ├── evaluate.py
│   └── hyperparameter_tuning.py
├── preprocessing.py
├── model_utils_pt.py
└── data/
    ├── benign/             # 80 % train benign
    ├── malignant/
    ├── benign_test/        # 20 % test benign
    ├── malignant_test/
    └── examples/           # volúmenes demo para la app
```

---

## Datasets
| Fuente | Descripción | Enlace |
|--------|-------------|--------|
| **LIDC‑IDRI** | 1 012 estudios TC (157 con veredicto radiólogo) | <https://www.cancerimagingarchive.net/collection/lidc-idri/> |
| **Etiquetas benigno/maligno** | CSV de Kaggle con diagnóstico | <https://www.kaggle.com/datasets/wissmeddeb/lidc-idri> |

> Solo necesitas **LIDC‑IDRI** + el CSV de etiquetas para reproducir los experimentos.

---

## Entorno de desarrollo
| Recurso | Detalle |
|---------|---------|
| **IDE** | Visual Studio Code 1.90 |
| **SO** | Windows 11 Pro x64 (24H2) |
| **Python** | 3.11.9 |
| **Hardware** | Intel i7‑1255U · 12 GB RAM · GPU integrada |
| **Librerías** | PyTorch 2.3 · MONAI 1.3 · Streamlit 1.34 · pdfkit 1.0 |

---

## Instalación paso a paso
```bash
# 1) Clonar
git clone https://github.com/tuUsuario/LUNG_CANCER_DETECTION_APP.git
cd LUNG_CANCER_DETECTION_APP

# 2) Crear entorno virtual (opcional)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

# 3) Instalar dependencias
pip install -r requirements.txt

# 4) Descargar LIDC‑IDRI + CSV de etiquetas
#    Descomprimir según la estructura /data mostrada arriba

# 5) (Opcional) Separar train/test automáticamente
python separaDatos.py
```

---

## Uso rápido

| Acción | Comando |
|--------|---------|
| **Dashboard Streamlit** | `streamlit run newApp.py` |
| **Entrenar modelos** | `python ml/train.py` |
| **Evaluar modelos** | `python ml/evaluate.py` |
| **Hiper‑tuning** | `python ml/hyperparameter_tuning.py` |

El **dashboard** permite:  
1. Diagnosticar un volumen TC y ver mapa de calor.  
2. Explorar métricas comparativas.  
3. Generar un PDF interactivo con un solo clic.

---

## Resultados principales
| Métrica | ResNet50 3D | DenseNet121 3D | CNN 3D custom |
|---------|------------:|---------------:|--------------:|
| Accuracy | **0.65** | 0.56 | 0.60 |
| Sensitivity | 0.26 | **0.53** | 0.06 |
| Specificity | **0.91** | 0.58 | 0.96 |
| AUC‑ROC | 0.59 | **0.60** | 0.54 |

**DenseNet121** es la más sensible; **ResNet50** la más específica.  
_¡Ninguna alcanza los umbrales clínicos recomendados todavía!_

---

## Créditos y licencias
- **Datasets** © TCIA & Kaggle, licencias CC‑BY.  
- **Código** © 2025 _Joelito AI Team_, licenciado bajo [MIT](LICENSE).

> ¿Preguntas o sugerencias? Abre un **Issue** o crea un _Pull Request_. ¡Contribuye! :sparkles:
