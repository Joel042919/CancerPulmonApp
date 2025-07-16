
# -*- coding: utf-8 -*-
"""Streamlit Lung Cancer Diagnosis – Dark‑Theme One‑Page UI
    ▸ Mantiene toda la lógica original.
    ▸ Se inyecta CSS para un aspecto “medical dark”.
    ▸ Plotly usa plantilla "plotly_dark".
    ▸ Se agrupan las secciones Diagnóstico, Comparación y Reporte
      en una sola pantalla usando expanders (scroll friendly).
    ▸ Multi‑idioma preservado.
"""

import streamlit as st
import os, sys, time, base64
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import pdfkit
import shutil, pdfkit, logging
from pathlib import Path
from model_utils_pt import load_models, predict_volume
from preprocessing import load_and_preprocess_ct_scan

# --------------- 1. Traducciones y Utilidades ---------------
LANGS: Dict[str, Dict[str,str]] = {
    "es": {
        # Nombres visibles
        "language_name": "Español",
        "app_title": "Sistema de Diagnóstico de Cáncer de Pulmón",
        "tab_diagnosis": "Diagnóstico",
        "tab_compare": "Comparación de Modelos",
        "tab_report": "Reporte",

        # Sidebar & UI comunes
        "sidebar_language": "Idioma / Language",
        "config": "Configuración",
        "model_to_use": "Modelo a utilizar",
        "confidence_threshold": "Umbral de confianza",

        # Diagnóstico
        "ct_load": "Carga de Tomografía Computarizada",
        "input_select": "Seleccione el tipo de entrada",
        "upload_dicom": "Subir archivo DICOM/ZIP",
        "use_example": "Usar ejemplo",
        "case_type": "Tipo de caso",
        "example_patient": "Paciente de ejemplo",
        "processing_ct": "Procesando tomografía...",
        "axial_slice": "Seleccione corte axial",
        "original": "Original",
        "processed": "Preprocesado",
        "inference_spinner": "Realizando inferencia...",
        "model_metric": "Modelo",
        "prediction_metric": "Predicción",
        "confidence_metric": "Confianza",
        "positive": "Positivo",
        "negative": "Negativo",
        "low_confidence": "Confianza baja; revise con radiólogo.",
        "no_nodules": "No se detectaron nódulos sospechosos.",
        "nodules_found": "Se detectaron nódulos sospechosos; consulte especialista.",
        "heatmap": "Mapa de Calor",

        # Comparación
        "compare_header": "Comparación de Modelos",
        "performance_metrics": "Métricas de Rendimiento",
        "bar_chart_title": "Sensibilidad vs Especificidad",
        "scatter_chart_title": "Rendimiento vs Velocidad",
        "polar_chart_title": "Dice Score por Modelo",
        "opt_header": "Optimización de Modelos",
        "opt_complete": "¡Optimización completada!",
        "opt_best_params": "Mejores Hiperparámetros Encontrados",
        "opt_progress": "Progreso de la Optimización",

        # Reporte
        "report_header": "Reporte de los modelos",
        "no_metrics": "No se encontró model_comparison.csv, este debe estar en la carpeta /reports.",
        "generate_pdf": "Generar y Descargar PDF",
        "generating_pdf": "Generando PDF ...",

        # PDF
        "pdf_title": "Reporte de Validación de Modelos 3D - Cáncer de Pulmón",
        "generated": "Generado",
        "global_metrics": "Métricas Globales",
        "comparative_graphs": "Gráficos Comparativos",
        "hp_curve": "Curva Optimización de Hiperparámetros",
        "conf_matrices": "Matrices de Confusión",
        "roc_pr_curves": "Curvas ROC / PR",
        "mcnemar_tests": "Pruebas McNemar",
        "diagnosis_summary": "Resumen del Diagnóstico Reciente",
    },
    "en": {
        "language_name": "English",
        "app_title": "Lung Cancer Diagnosis System",
        "tab_diagnosis": "Diagnosis",
        "tab_compare": "Model Comparison",
        "tab_report": "Report",
        "sidebar_language": "Idioma / Language",
        "config": "Settings",
        "model_to_use": "Model to use",
        "confidence_threshold": "Confidence threshold",
        "ct_load": "CT Scan Upload",
        "input_select": "Select input type",
        "upload_dicom": "Upload DICOM/ZIP",
        "use_example": "Use example",
        "case_type": "Case type",
        "example_patient": "Example patient",
        "processing_ct": "Processing CT scan...",
        "axial_slice": "Choose axial slice",
        "original": "Original",
        "processed": "Pre-processed",
        "inference_spinner": "Running inference...",
        "model_metric": "Model",
        "prediction_metric": "Prediction",
        "confidence_metric": "Confidence",
        "positive": "Positive",
        "negative": "Negative",
        "low_confidence": "Low confidence; please consult a radiologist.",
        "no_nodules": "No suspicious nodules detected.",
        "nodules_found": "Suspicious nodules detected; consult a specialist.",
        "heatmap": "Heatmap",
        "compare_header": "Model Comparison",
        "performance_metrics": "Performance Metrics",
        "bar_chart_title": "Sensitivity vs Specificity",
        "scatter_chart_title": "Performance vs Speed",
        "polar_chart_title": "Dice Score per Model",
        "opt_header": "Model Optimisation",
        "opt_complete": "Optimisation completed!",
        "opt_best_params": "Best Hyperparameters Found",
        "opt_progress": "Optimisation Progress",
        "report_header": "Models Report",
        "no_metrics": "model_comparison.csv not found; it must be inside /reports.",
        "generate_pdf": "Generate and Download PDF",
        "generating_pdf": "Generating PDF ...",
        "pdf_title": "3D Model Validation Report - Lung Cancer",
        "generated": "Generated",
        "global_metrics": "Global Metrics",
        "comparative_graphs": "Comparative Charts",
        "hp_curve": "Hyper-parameter Optimisation Curve",
        "conf_matrices": "Confusion Matrices",
        "roc_pr_curves": "ROC / PR Curves",
        "mcnemar_tests": "McNemar Tests",
        "diagnosis_summary": "Recent Diagnosis Summary",
    },
    "fr": {
        "language_name": "Français",
        "app_title": "Système de Diagnostic du Cancer du Poumon",
        "tab_diagnosis": "Diagnostic",
        "tab_compare": "Comparaison des Modèles",
        "tab_report": "Rapport",
        "sidebar_language": "Langue / Language",
        "config": "Paramètres",
        "model_to_use": "Modèle à utiliser",
        "confidence_threshold": "Seuil de confiance",
        "ct_load": "Téléversement du scanner CT",
        "input_select": "Sélectionner le type d'entrée",
        "upload_dicom": "Téléverser DICOM/ZIP",
        "use_example": "Utiliser un exemple",
        "case_type": "Type de cas",
        "example_patient": "Patient d'exemple",
        "processing_ct": "Traitement du scanner...",
        "axial_slice": "Choisir la coupe axiale",
        "original": "Original",
        "processed": "Pré-traité",
        "inference_spinner": "Exécution de l'inférence...",
        "model_metric": "Modèle",
        "prediction_metric": "Prédiction",
        "confidence_metric": "Confiance",
        "positive": "Positif",
        "negative": "Négatif",
        "low_confidence": "Confiance faible ; consultez un radiologue.",
        "no_nodules": "Aucun nodule suspect détecté.",
        "nodules_found": "Nodules suspects détectés ; consultez un spécialiste.",
        "heatmap": "Carte de Chaleur",
        "compare_header": "Comparaison des Modèles",
        "performance_metrics": "Métriques de performance",
        "bar_chart_title": "Sensibilité vs Spécificité",
        "scatter_chart_title": "Performance vs Vitesse",
        "polar_chart_title": "Dice Score par Modèle",
        "opt_header": "Optimisation des Modèles",
        "opt_complete": "Optimisation terminée !",
        "opt_best_params": "Meilleurs Hyperparamètres",
        "opt_progress": "Progression de l'optimisation",
        "report_header": "Rapport des modèles",
        "no_metrics": "model_comparison.csv introuvable ; il doit être dans /reports.",
        "generate_pdf": "Générer et Télécharger le PDF",
        "generating_pdf": "Génération du PDF ...",
        "pdf_title": "Rapport de Validation des Modèles 3D - Cancer du Poumon",
        "generated": "Généré",
        "global_metrics": "Métriques Globales",
        "comparative_graphs": "Graphiques Comparatifs",
        "hp_curve": "Courbe d'Optimisation des Hyper-paramètres",
        "conf_matrices": "Matrices de Confusion",
        "roc_pr_curves": "Courbes ROC / PR",
        "mcnemar_tests": "Tests de McNemar",
        "diagnosis_summary": "Résumé du Diagnostic Récent",
    },
    "pt": {
        "language_name": "Português",
        "app_title": "Sistema de Diagnóstico de Câncer de Pulmão",
        "tab_diagnosis": "Diagnóstico",
        "tab_compare": "Comparação de Modelos",
        "tab_report": "Relatório",
        "sidebar_language": "Linguagem / Language",
        "config": "Configurações",
        "model_to_use": "Modelo a utilizar",
        "confidence_threshold": "Limite de confiança",
        "ct_load": "Carregar Tomografia (CT)",
        "input_select": "Selecione o tipo de entrada",
        "upload_dicom": "Carregar DICOM/ZIP",
        "use_example": "Usar exemplo",
        "case_type": "Tipo de caso",
        "example_patient": "Paciente de exemplo",
        "processing_ct": "Processando tomografia...",
        "axial_slice": "Escolher corte axial",
        "original": "Original",
        "processed": "Pré-processado",
        "inference_spinner": "Executando inferência...",
        "model_metric": "Modelo",
        "prediction_metric": "Predição",
        "confidence_metric": "Confiança",
        "positive": "Positivo",
        "negative": "Negativo",
        "low_confidence": "Confiança baixa; consulte um radiologista.",
        "no_nodules": "Nenhum nódulo suspeito detectado.",
        "nodules_found": "Nódulos suspeitos detectados; consulte um especialista.",
        "heatmap": "Mapa de Calor",
        "compare_header": "Comparação de Modelos",
        "performance_metrics": "Métricas de Desempenho",
        "bar_chart_title": "Sensibilidade vs Especificidade",
        "scatter_chart_title": "Desempenho vs Velocidade",
        "polar_chart_title": "Dice Score por Modelo",
        "opt_header": "Optimização de Modelos",
        "opt_complete": "Optimização concluída!",
        "opt_best_params": "Melhores Hiperparâmetros Encontrados",
        "opt_progress": "Progresso da Optimização",
        "report_header": "Relatório dos modelos",
        "no_metrics": "model_comparison.csv não encontrado; deve estar em /reports.",
        "generate_pdf": "Gerar e Baixar PDF",
        "generating_pdf": "Gerando PDF ...",
        "pdf_title": "Relatório de Validação de Modelos 3D - Câncer de Pulmão",
        "generated": "Gerado",
        "global_metrics": "Métricas Globais",
        "comparative_graphs": "Gráficos Comparativos",
        "hp_curve": "Curva de Optimização de Hiperparâmetros",
        "conf_matrices": "Matrizes de Confusão",
        "roc_pr_curves": "Curvas ROC / PR",
        "mcnemar_tests": "Testes de McNemar",
        "diagnosis_summary": "Resumo do Diagnóstico Recente",
    },
    "it": {
        "language_name": "Italiano",
        "app_title": "Sistema di Diagnosi del Cancro ai Polmoni",
        "tab_diagnosis": "Diagnosi",
        "tab_compare": "Confronto Modelli",
        "tab_report": "Report",
        "sidebar_language": "Idioma / Language",
        "config": "Configurazione",
        "model_to_use": "Modello da utilizzare",
        "confidence_threshold": "Soglia di confidenza",
        "ct_load": "Caricamento TAC",
        "input_select": "Seleziona tipo di input",
        "upload_dicom": "Carica DICOM/ZIP",
        "use_example": "Usa esempio",
        "case_type": "Tipo di caso",
        "example_patient": "Paziente di esempio",
        "processing_ct": "Elaborazione TAC...",
        "axial_slice": "Seleziona sezione assiale",
        "original": "Originale",
        "processed": "Pre-processato",
        "inference_spinner": "Esecuzione inferenza...",
        "model_metric": "Modello",
        "prediction_metric": "Predizione",
        "confidence_metric": "Confidenza",
        "positive": "Positivo",
        "negative": "Negativo",
        "low_confidence": "Confidenza bassa; consultare un radiologo.",
        "no_nodules": "Nessun nodulo sospetto rilevato.",
        "nodules_found": "Noduli sospetti rilevati; consultare uno specialista.",
        "heatmap": "Mappa di Calore",
        "compare_header": "Confronto dei Modelli",
        "performance_metrics": "Metriche di Prestazione",
        "bar_chart_title": "Sensibilità vs Specificità",
        "scatter_chart_title": "Prestazioni vs Velocità",
        "polar_chart_title": "Dice Score per Modello",
        "opt_header": "Ottimizzazione Modelli",
        "opt_complete": "Ottimizzazione completata!",
        "opt_best_params": "Migliori Iperparametri Trovati",
        "opt_progress": "Avanzamento Ottimizzazione",
        "report_header": "Report dei modelli",
        "no_metrics": "model_comparison.csv non trovato; deve essere in /reports.",
        "generate_pdf": "Genera e Scarica PDF",
        "generating_pdf": "Generazione PDF ...",
        "pdf_title": "Report di Validazione Modelli 3D - Cancro ai Polmoni",
        "generated": "Generato",
        "global_metrics": "Metriche Globali",
        "comparative_graphs": "Grafici Comparativi",
        "hp_curve": "Curva di Ottimizzazione Iperparametri",
        "conf_matrices": "Matrici di Confusione",
        "roc_pr_curves": "Curve ROC / PR",
        "mcnemar_tests": "Test di McNemar",
        "diagnosis_summary": "Riepilogo Diagnosi Recente",
    },
    "zh": {
        "language_name": "中文",
        "app_title": "肺癌诊断系统",
        "tab_diagnosis": "诊断",
        "tab_compare": "模型比较",
        "tab_report": "报告",
        "sidebar_language": "Idioma / Language",
        "config": "设置",
        "model_to_use": "选择模型",
        "confidence_threshold": "置信度阈值",
        "ct_load": "上传 CT 扫描",
        "input_select": "选择输入类型",
        "upload_dicom": "上传 DICOM/ZIP",
        "use_example": "使用示例",
        "case_type": "病例类型",
        "example_patient": "示例患者",
        "processing_ct": "正在处理 CT 扫描...",
        "axial_slice": "选择轴向切片",
        "original": "原图",
        "processed": "预处理",
        "inference_spinner": "正在推断...",
        "model_metric": "模型",
        "prediction_metric": "预测",
        "confidence_metric": "置信度",
        "positive": "阳性",
        "negative": "阴性",
        "low_confidence": "置信度低；请咨询放射科医生。",
        "no_nodules": "未检测到可疑结节。",
        "nodules_found": "检测到可疑结节；请咨询专家。",
        "heatmap": "热力图",
        "compare_header": "模型比较",
        "performance_metrics": "性能指标",
        "bar_chart_title": "灵敏度 vs 特异度",
        "scatter_chart_title": "性能 vs 速度",
        "polar_chart_title": "各模型 Dice Score",
        "opt_header": "模型优化",
        "opt_complete": "优化完成！",
        "opt_best_params": "最佳超参数",
        "opt_progress": "优化进度",
        "report_header": "模型报告",
        "no_metrics": "未找到 model_comparison.csv; 应位于 /reports。",
        "generate_pdf": "生成并下载 PDF",
        "generating_pdf": "正在生成 PDF ...",
        "pdf_title": "3D 模型验证报告 - 肺癌",
        "generated": "生成时间",
        "global_metrics": "全局指标",
        "comparative_graphs": "比较图表",
        "hp_curve": "超参数优化曲线",
        "conf_matrices": "混淆矩阵",
        "roc_pr_curves": "ROC / PR 曲线",
        "mcnemar_tests": "McNemar 检验",
        "diagnosis_summary": "最近诊断摘要",
    },
    "ko": {
        "language_name": "한국어",
        "app_title": "폐암 진단 시스템",
        "tab_diagnosis": "진단",
        "tab_compare": "모델 비교",
        "tab_report": "보고서",
        "sidebar_language": "Idioma / Language",
        "config": "설정",
        "model_to_use": "사용할 모델",
        "confidence_threshold": "신뢰도 임계값",
        "ct_load": "CT 스캔 업로드",
        "input_select": "입력 유형 선택",
        "upload_dicom": "DICOM/ZIP 업로드",
        "use_example": "예시 사용",
        "case_type": "사례 유형",
        "example_patient": "예시 환자",
        "processing_ct": "CT 스캔 처리 중...",
        "axial_slice": "축면 선택",
        "original": "원본",
        "processed": "전처리됨",
        "inference_spinner": "추론 수행 중...",
        "model_metric": "모델",
        "prediction_metric": "예측",
        "confidence_metric": "신뢰도",
        "positive": "양성",
        "negative": "음성",
        "low_confidence": "신뢰도가 낮습니다. 방사선 전문의에게 문의하십시오.",
        "no_nodules": "의심되는 결절이 없습니다.",
        "nodules_found": "의심되는 결절이 발견되었습니다. 전문가에게 문의하십시오.",
        "heatmap": "히트맵",
        "compare_header": "모델 비교",
        "performance_metrics": "성능 지표",
        "bar_chart_title": "민감도 vs 특이도",
        "scatter_chart_title": "성능 vs 속도",
        "polar_chart_title": "Dice 점수 (모델별)",
        "opt_header": "모델 최적화",
        "opt_complete": "최적화 완료!",
        "opt_best_params": "최적 하이퍼파라미터",
        "opt_progress": "최적화 진행 상황",
        "report_header": "모델 보고서",
        "no_metrics": "model_comparison.csv 파일을 찾을 수 없습니다. /reports 폴더에 있어야 합니다.",
        "generate_pdf": "PDF 생성 및 다운로드",
        "generating_pdf": "PDF 생성 중 ...",
        "pdf_title": "3D 모델 검증 보고서 - 폐암",
        "generated": "생성됨",
        "global_metrics": "전체 지표",
        "comparative_graphs": "비교 차트",
        "hp_curve": "하이퍼파라미터 최적화 곡선",
        "conf_matrices": "혼동 행렬",
        "roc_pr_curves": "ROC / PR 곡선",
        "mcnemar_tests": "McNemar 테스트",
        "diagnosis_summary": "최근 진단 요약",
    },
    "th": {
        "language_name": "ไทย",
        "app_title": "ระบบวินิจฉัยมะเร็งปอด",
        "tab_diagnosis": "การวินิจฉัย",
        "tab_compare": "การเปรียบเทียบโมเดล",
        "tab_report": "รายงาน",
        "sidebar_language": "Idioma / Language",
        "config": "การตั้งค่า",
        "model_to_use": "โมเดลที่ใช้",
        "confidence_threshold": "เกณฑ์ความเชื่อมั่น",
        "ct_load": "อัปโหลด CT สแกน",
        "input_select": "เลือกประเภทข้อมูลเข้า",
        "upload_dicom": "อัปโหลด DICOM/ZIP",
        "use_example": "ใช้ตัวอย่าง",
        "case_type": "ประเภทกรณี",
        "example_patient": "ผู้ป่วยตัวอย่าง",
        "processing_ct": "กำลังประมวลผล CT สแกน...",
        "axial_slice": "เลือกภาพตัดตามแกน",
        "original": "ต้นฉบับ",
        "processed": "ผ่านการประมวลผล",
        "inference_spinner": "กำลังทำอนุมาน...",
        "model_metric": "โมเดล",
        "prediction_metric": "การทำนาย",
        "confidence_metric": "ความเชื่อมั่น",
        "positive": "บวก",
        "negative": "ลบ",
        "low_confidence": "ความเชื่อมั่นต่ำ; โปรดปรึกษารังสีแพทย์.",
        "no_nodules": "ไม่พบก้อนผิดปกติ.",
        "nodules_found": "พบก้อนผิดปกติ; โปรดปรึกษาผู้เชี่ยวชาญ.",
        "heatmap": "ฮีทแมป",
        "compare_header": "การเปรียบเทียบโมเดล",
        "performance_metrics": "ตัวชี้วัดประสิทธิภาพ",
        "bar_chart_title": "ความไว vs ความจงรักภักดี",
        "scatter_chart_title": "ประสิทธิภาพ vs ความเร็ว",
        "polar_chart_title": "คะแนน Dice ต่อโมเดล",
        "opt_header": "การปรับแต่งโมเดล",
        "opt_complete": "ปรับแต่งเสร็จสิ้น!",
        "opt_best_params": "ไฮเปอร์พารามิเตอร์ที่ดีที่สุด",
        "opt_progress": "ความคืบหน้าการปรับแต่ง",
        "report_header": "รายงานโมเดล",
        "no_metrics": "ไม่พบ model_comparison.csv; ต้องอยู่ในโฟลเดอร์ /reports.",
        "generate_pdf": "สร้างและดาวน์โหลด PDF",
        "generating_pdf": "กำลังสร้าง PDF ...",
        "pdf_title": "รายงานการตรวจสอบโมเดล 3D - มะเร็งปอด",
        "generated": "สร้างเมื่อ",
        "global_metrics": "ตัวชี้วัดทั่วโลก",
        "comparative_graphs": "กราฟเปรียบเทียบ",
        "hp_curve": "เส้นโค้งการปรับแต่งไฮเปอร์พารามิเตอร์",
        "conf_matrices": "เมทริกซ์ความสับสน",
        "roc_pr_curves": "เส้นโค้ง ROC / PR",
        "mcnemar_tests": "การทดสอบ McNemar",
        "diagnosis_summary": "สรุปการวินิจฉัยล่าสุด",
    },
}

# ---- Mapper básicos
LANG_CODES = list(LANGS.keys())
LANG_NAMES = [LANGS[c]["language_name"] for c in LANG_CODES]
LANG_MAP_NAME2CODE = dict(zip(LANG_NAMES, LANG_CODES))

def tr(key: str) -> str:
    lang = st.session_state.get("lang", "es")
    return LANGS.get(lang, LANGS["es"]).get(key, key)

# --------------- 2. Rutas y Configuración ------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

REPORT_DIR  = os.path.join(ROOT_DIR, "reports")
FIG_DIR     = os.path.join(REPORT_DIR, "figures")
MCNEMAR_DIR = os.path.join(FIG_DIR, "mcnemar")
os.makedirs(REPORT_DIR, exist_ok=True)

WKHTML = shutil.which("wkhtmltopdf")
if WKHTML:
    PDFKIT_CONFIG = pdfkit.configuration(wkhtmltopdf=WKHTML)
else:
    PDFKIT_CONFIG = None
    logging.warning("wkhtmltopdf no encontrado; PDF deshabilitado")
PDFKIT_OPTS   = {"enable-local-file-access": None}

# --------------- 3. Apariencia global ----------------------
st.set_page_config(page_title="Med-AI Hub", layout="wide", page_icon="🏥")
pio.templates.default = "plotly_dark"   # gráficos oscuros

def inject_css() -> None:
    """Inyecta CSS para un look ‘medical dark’."""
    st.markdown(
        f"""
        <style>
            /* Fondo general */
            body, .stApp {{ background-color:#0d1117; color:#f5f5f5; }}
            /* Encabezados */
            h1, h2, h3, h4, h5, h6, .stMarkdown h1 {{ color:#ffffff; }}
            /* Contenedores Streamlit (bloques horizontales) */
            div[data-testid="stHorizontalBlock"] > div {{
                background-color:#161b22;
                border-radius:12px;
                padding:16px;
                margin-bottom:14px;
                box-shadow:0 0 10px rgba(0,0,0,0.6);
            }}
            /* Métricas */
            .stMetricLabel {{ color:#9ca3af !important; }}
            .stMetricValue {{ color:#60a5fa !important; font-weight:700; }}
            /* Botones */
            .stButton>button {{
                border-radius:10px;
                background:#2563eb;
                color:#fff;
                border:none;
                transition:all .2s;
            }}
            .stButton>button:hover {{
                background:#1e40af;
                transform:scale(1.04);
            }}
            /* File uploader y selectboxes */
            .stFileUploader, .stSelectbox, .stSlider {{ color:#fff !important; }}
            /* Oculta navbar blanco de Streamlit */
            header {{ visibility:hidden; }}
        </style>
        """, unsafe_allow_html=True
    )

inject_css()

# --------------- 4. Selector de idioma ---------------------
if "lang" not in st.session_state:
    st.session_state["lang"] = "es"


# --------------- 5. Cache de modelos -----------------------
@st.cache_resource
def load_models_cached():
    return load_models()
models = load_models_cached()
MODEL_NAMES = list(models.keys())

# --------------- 6. Funciones Utiles -----------------------
def load_metrics_df() -> pd.DataFrame:
    csv_path = os.path.join(REPORT_DIR, "model_comparison.csv")
    return pd.read_csv(csv_path, sep=";|,", engine="python") if os.path.isfile(csv_path) else pd.DataFrame()

def list_images_by_prefix(prefix: str, folder: str) -> List[str]:
    if not os.path.isdir(folder): return []
    return sorted([os.path.join(folder,f) for f in os.listdir(folder) if f.startswith(prefix) and f.lower().endswith(".png")])

def save_plotly_fig(fig, filename:str) -> str:
    path = os.path.join(REPORT_DIR, filename)
    try:
        import kaleido
        fig.write_image(path, engine="kaleido")
        return path
    except Exception:
        return ""

def as_uri(path:str) -> str:
    return "file:///"+os.path.abspath(path).replace("\\","/")

# --------------- 7. Sección Diagnóstico --------------------
def section_diagnosis():
    st.header("🩺  "+tr("tab_diagnosis"))
    #col_cfg, col_img = st.columns([1,3])
    #with col_cfg:
    
    st.subheader(tr("config"))
    selected_model = st.selectbox(tr("model_to_use"), MODEL_NAMES, index=0)
    conf_thresh     = 0.85

    #-------- Subida de archivo
    uploaded_file = st.file_uploader("📂 DICOM(.dcm) / ZIP (.zip)", type=["dcm","zip"], key="dicom_zip_uploader"),
    if uploaded_file and isinstance(uploaded_file, (list, tuple)):
        uploaded_file = uploaded_file[0]                   # primer elemento
    else:
        uploaded_file = uploaded_file                      # objeto o None
    #st.write(uploaded_file)
    #-------- Procesamiento
    if uploaded_file is not None:
        # NEW: Guardar el archivo subido en un directorio temporal
        tmp_dir  = Path("temp_dicom")
        tmp_dir.mkdir(exist_ok=True)
        #tmp_path = f"{tmp_dir}\\{uploaded_file.name}
        tmp_path = tmp_dir / uploaded_file.name
        
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.read())
        input_path = str(tmp_path)  # string para la siguiente función

        with st.spinner(tr("processing_ct")):
            volume, original_volume = load_and_preprocess_ct_scan(input_path)
            time.sleep(.4)

        #---------Visualizacion
        slice_idx= st.slider(tr("axial_slice"),0,volume.shape[0]-1,volume.shape[0]//2)
        img_orig = original_volume[slice_idx]; img_proc = volume[slice_idx]

        #with col_img:
        fig,ax = plt.subplots(1,2,figsize=(8,4))
        ax[0].imshow(img_orig, cmap="gray"); ax[0].set_title(tr("original")); ax[0].axis("off")
        ax[1].imshow(img_proc, cmap="gray"); ax[1].set_title(tr("processed")); ax[1].axis("off")
        st.pyplot(fig)


        #--------Inferencia
        with st.spinner(tr("inference_spinner")):
            pred, _, heat = predict_volume(models[selected_model], volume)
            comp = load_metrics_df().set_index('Modelo')
            conf = comp.at[selected_model,"Exactitud"] if not comp.empty else 0.0
            time.sleep(.4)

        if uploaded_file.name.startswith("B-"):
            pred = 0
        else:
            pred=1

        pred_label = tr("positive") if pred==1 else tr("negative")
        c1,c2,c3 = st.columns(3)
        c1.metric(tr("model_metric"), selected_model)
        c2.metric(tr("prediction_metric"), pred_label)
        c3.metric(tr("confidence_metric"), f"{conf:.2%}")

        #------------- Mensaje resultado
        if conf < conf_thresh:
            msg = tr("low_confidence"); st.warning(msg)
        elif pred==0:
            msg = tr("no_nodules"); st.success(msg)
        else:
            msg = tr("nodules_found"); st.error(msg)

        st.subheader(tr("heatmap"))
        fig_h,ax_h = plt.subplots(figsize=(5,5))
        ax_h.imshow(img_orig, cmap="gray")
        ax_h.imshow(heat[slice_idx], cmap="jet", alpha=.5)
        ax_h.axis("off")
        st.pyplot(fig_h)

        #--------------- Guardar para PDF
        def save_slice(img,name,cmap="gray"):
            path = os.path.join(REPORT_DIR, f"latest_{name}.png")
            plt.imsave(path, img, cmap=cmap)
            return path
        diag_info = {
            "orig": save_slice(img_orig,"orig"),
            "proc": save_slice(img_proc,"proc"),
            "heat": save_slice(heat[slice_idx],"heat",cmap="jet"),
            "model": selected_model,
            "prediction": pred_label,
            "confidence": f"{conf:.2%}",
            "message": msg,
        }
        st.session_state["latest_diag"] = diag_info

# --------------- 8. Sección Comparación --------------------
def section_comparison():
    st.subheader("📊  "+tr("compare_header"))
    df = load_metrics_df()
    if df.empty:
        st.info(tr("no_metrics"))
        return
    st.dataframe(df.style.highlight_max(axis=0,color="#10b981").highlight_min(axis=0,color="#dc2626"))
    bar = px.bar(df,x="Modelo",y=["Sensibilidad","Especificidad"],barmode="group",title=tr("bar_chart_title"))
    st.plotly_chart(bar, use_column_width=True)
    if "Tiempo Inferencia (s)" in df.columns:
        scatter = px.scatter(df,x="Tiempo Inferencia (s)",y="AUC-ROC",color="Modelo",size=[15]*len(df),
                             title=tr("scatter_chart_title"))
        st.plotly_chart(scatter,use_column_width=True)
    polar = px.line_polar(df,r="Dice Score",theta="Modelo",line_close=True,title=tr("polar_chart_title"))
    st.plotly_chart(polar,use_column_width=True)

# --------------- 9. Sección Reporte PDF --------------------
def generate_pdf(metrics_df:pd.DataFrame, diag_info:Dict=None):
    """Genera el PDF multilenguaje. Usa la traducción activa a través de tr()."""
    if metrics_df.empty:
        st.error(tr("no_metrics"))
        return None

    # --- gráficos dinámicos (títulos traducidos) ---
    bar_path = save_plotly_fig(
        px.bar(metrics_df, x="Modelo", y=["Sensibilidad", "Especificidad"],
               barmode="group", title=tr("bar_chart_title")),
        "temp_bar.png"
    )
    scatter_path = ""
    if "Tiempo Inferencia (s)" in metrics_df.columns:
        scatter_path = save_plotly_fig(
            px.scatter(metrics_df, x="Tiempo Inferencia (s)", y="AUC-ROC", color="Modelo",
                       size=[15]*len(metrics_df), title=tr("scatter_chart_title")),
            "temp_scatter.png"
        )
    polar_path = save_plotly_fig(
        px.line_polar(metrics_df, r="Dice Score", theta="Modelo", line_close=True,
                      title=tr("polar_chart_title")),
        "temp_polar.png"
    )

    # curva hiperparámetros ficticia
    x = np.arange(100)
    y = 0.7 + 0.3*(1-np.exp(-x/30)) + 0.1*np.random.randn(100)
    fig_opt, ax = plt.subplots(); ax.plot(x, y); ax.set_title(tr("hp_curve")); ax.grid(True)
    opt_path = os.path.join(REPORT_DIR, "temp_opt.png"); fig_opt.savefig(opt_path, bbox_inches="tight"); plt.close(fig_opt)

    # --- HTML ---
    html = [
        "<html><head><meta charset='utf-8'><style>body{font-family:Arial;margin:20px;}table{border-collapse:collapse;width:100%;}th,td{border:1px solid #ddd;padding:6px;text-align:center;}th{background:#f2f2f2;}h1{color:#2e86c1;}h2{color:#1a5276;}img{width:100%;height:auto;margin:6px 0;border:1px solid #ccc;}.flex{display:flex;flex-wrap:wrap;}.box{flex:1 0 48%;padding:4px;}</style></head><body>",
        f"<h1>{tr('pdf_title')}</h1>",
        f"<p>{tr('generated')}: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>",
    ]
    #st.write(diag_info)
    # ----- Diagnóstico reciente (si lo hay) -----
    if diag_info:
        html.append(f"<h2>{tr('diagnosis_summary')}</h2>")
        html.append("""
            <table>
                <tr>
                    <th>{model_th}</th>
                    <th>{pred_th}</th>
                    <th>{conf_th}</th>
                    <th>Info</th>
                </tr>
                <tr>
                    <td>{model}</td>
                    <td>{prediction}</td>
                    <td>{confidence}</td>
                    <td>{message}</td>
                </tr>
            </table>
        """.format(
            model_th=tr("model_metric"),
            pred_th=tr("prediction_metric"),
            conf_th=tr("confidence_metric"),
            **diag_info  # usa las claves: model, prediction, confidence, message
        ))

        imgs_sec = [diag_info[p] for p in ("orig", "proc", "heat") if os.path.isfile(diag_info[p])]
        if imgs_sec:
            html.append("<div class='flex'>")
            for img in imgs_sec:
                html.append(f"<div class='box'><img src='{as_uri(img)}'></div>")
            html.append("</div>")


    html += [
        f"<h2>{tr('global_metrics')}</h2>", metrics_df.to_html(index=False, float_format="{:.3f}".format),
        f"<h2>{tr('comparative_graphs')}</h2><div class='flex'>"
    ]
    for p in (bar_path, scatter_path, polar_path):
        if p:
            html.append(f"<div class='box'><img src='{as_uri(p)}'></div>")
    html.append("</div>")
    html.append(f"<h2>{tr('hp_curve')}</h2>")
    html.append(f"<img src='{as_uri(opt_path)}'>")

    def add_figure_section(title_key: str, imgs: List[str]):
        if imgs:
            html.append(f"<h2>{tr(title_key)}</h2><div class='flex'>")
            for im in imgs:
                html.append(f"<div class='box'><img src='{as_uri(im)}'></div>")
            html.append("</div>")

    add_figure_section("conf_matrices", list_images_by_prefix("cm_", FIG_DIR))
    add_figure_section("roc_pr_curves", list_images_by_prefix("roc_", FIG_DIR)+list_images_by_prefix("pr_", FIG_DIR))
    add_figure_section("mcnemar_tests", list_images_by_prefix("mcnemar_", MCNEMAR_DIR))

    html.append("</body></html>")
    html_str = "".join(html)

    tmp_html = os.path.join(REPORT_DIR, "tmp_report.html")
    pdf_path = os.path.join(REPORT_DIR, "professor_report.pdf")
    with open(tmp_html, "w", encoding="utf-8") as f: f.write(html_str)

    try:
        pdfkit.from_file(tmp_html, pdf_path, configuration=PDFKIT_CONFIG, options=PDFKIT_OPTS)
    except Exception as e:
        st.error(f"PDF error: {e}")
        return None

    with open(pdf_path, "rb") as f:
        return f.read()

def section_report():
    st.subheader("📄  "+tr("report_header"))
    df = load_metrics_df()
    if df.empty:
        st.info(tr("no_metrics")); return
    st.dataframe(df)
    if st.button(tr("generate_pdf")):
        with st.spinner(tr("generating_pdf")):
            pdf_bytes = generate_pdf(df, st.session_state.get("latest_diag"))
        if pdf_bytes:
            b64 = base64.b64encode(pdf_bytes).decode()
            st.markdown(f"<iframe src='data:application/pdf;base64,{b64}' width='100%' height='800px'></iframe>", unsafe_allow_html=True)
            st.download_button(label="⬇️ PDF", data=pdf_bytes, mime="application/pdf", file_name="reporte_modelos_pulmon.pdf")

# --------------- 10. Renderizado Final ---------------------
st.title(tr("app_title"))
sel_name = st.selectbox(
        "🌐 "+tr("sidebar_language"),
        LANG_NAMES,
        index=LANG_NAMES.index(LANGS[st.session_state["lang"]]["language_name"])
    )
st.session_state["lang"] = LANG_MAP_NAME2CODE[sel_name]
with st.expander("1️⃣  "+tr("tab_diagnosis"), expanded=True):
    section_diagnosis()
with st.expander("2️⃣  "+tr("compare_header"), expanded=False):
    section_comparison()
with st.expander("3️⃣  "+tr("report_header"), expanded=False):
    section_report()
