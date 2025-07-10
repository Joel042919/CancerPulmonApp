# 🩺 **Lung-AI-3D**
### _“Evaluación comparativa de redes neuronales tridimensionales para la detección automática de cáncer pulmonar”_

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11.9-blue?logo=python">
  <img src="https://img.shields.io/badge/streamlit-1.34%2B-red?logo=streamlit">
  <img src="https://img.shields.io/badge/status-beta-yellow">
</p>

> **TL;DR** Este repo contiene todo el pipeline — pre‑proceso, entrenamiento, evaluación y app web — para comparar **ResNet50 3D**, **DenseNet121 3D** y una **CNN 3D personalizada** en la clasificación de nódulos pulmonares (benignos vs malignos) a partir de tomografías computarizadas (TC).

---

## 🗺️ Índice
1. Motivación  
2. Estructura del proyecto  
3. Datasets  
4. Entorno de desarrollo  
5. Instalación paso a paso  
6. Uso rápido  
7. Comandos avanzados  
8. Resultados principales  
9. Créditos y licencias  

---

## Motivación
- **Impacto clínico** El cáncer de pulmón es la primera causa de mortalidad oncológica mundial. Diagnosticar nódulos < 1 cm supone revisar ± 120 cortes DICOM por paciente: tedioso y propenso a error.
- **IA 3D** Las CNN 3D analizan volúmenes completos, preservando contexto espacial. ¿Cuál arquitectura logra el mejor *trade‑off* sensibilidad / especificidad solo en CPU?
- **Transparencia** El repo ofrece código reproducible, métricas, figuras y un dashboard Streamlit listo para demo.

---

## Estructura del proyecto
```text
LUNG_CANCER_DETECTION_APP/
│
├── newApp.py              # app Streamlit (Diagnóstico, Comparación, Reporte)
├── requirements.txt
├── reports/               # métricas, PDFs y figuras
│   ├── model_comparison.csv
│   └── figures/
│       ├── cm_*           # matrices de confusión
│       ├── roc_* / pr_*   # ROC & PR
│       └── mcnemar/
├── models/                # pesos .pth (post‑training)
├── ml/
│   ├── train.py
│   ├── evaluate.py
│   └── hyperparameter_tuning.py
├── preprocessing.py
├── model_utils_pt.py
└── data/
    ├── benign/            # 80 % train benign
    ├── malignant/
    ├── benign_test/       # 20 % test benign
    ├── malignant_test/
    └── examples/          # volúmenes demo
```

---

## Datasets
| Fuente | Tipo | Link |
|--------|------|------|
| **LIDC‑IDRI** | 1 012 TC; 157 con dictamen radiólogo | <https://www.cancerimagingarchive.net/collection/lidc-idri/> |
| **Etiquetas Kaggle** | benigno / maligno CSV | <https://www.kaggle.com/datasets/wissmeddeb/lidc-idri> |

---

## Entorno de desarrollo
|  |  |
|--|--|
| **IDE** | VS Code 1.90 |
| **SO**  | Windows 11 Pro 64 (24H2) |
| **Python** | 3.11.9 |
| **HW** | Intel i7‑1255U · 12 GB RAM · GPU integrada |
| **Libs** | PyTorch 2.3, MONAI 1.3, Streamlit 1.34, pdfkit 1.0 |

---

## Instalación paso a paso
```bash
git clone https://github.com/tuUsuario/LUNG_CANCER_DETECTION_APP.git
cd LUNG_CANCER_DETECTION_APP
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt

# ↓ descarga LIDC‑IDRI y el CSV de etiquetas
#   descomprime en data/ según estructura mostrada
python separaDatos.py   # opcional: reparte train / test
```

---

## Uso rápido

| Acción | Comando |
|--------|---------|
| Dashboard Streamlit | `streamlit run newApp.py` |
| Entrenar modelos | `python ml/train.py` |
| Evaluar modelos | `python ml/evaluate.py` |
| Hyper‑tuning (demo) | `python ml/hyperparameter_tuning.py` |

---

## Resultados principales
| Métrica | ResNet50 3D | DenseNet121 3D | CNN 3D custom |
|---------|-------------|---------------|---------------|
| Accuracy | **0.65** | 0.56 | 0.60 |
| Sensitivity | 0.26 | **0.53** | 0.06 |
| Specificity | **0.91** | 0.58 | 0.96 |
| AUC‑ROC | 0.59 | **0.60** | 0.54 |

---

## Créditos y licencias
- Datasets TCIA & Kaggle — CC‑BY.  
- Código © 2025 — Equipo *Joelito AI* — MIT License.

¡Contribuciones y detección de *issues* son bienvenidos! 🩻✨
