# save_preds.py
import numpy as np
import os, sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from app.model_utils_pt import load_models, predict_volume
from app.preprocessing import load_and_preprocess_ct_scan

from tqdm import tqdm

# 1) Carga etiquetas y paths
test_cases = []
true_labels = []
for p in os.listdir('data/benign')[:50]:
    test_cases.append(os.path.join('data/benign', p)); true_labels.append(0)
for p in os.listdir('data/malignant')[:50]:
    test_cases.append(os.path.join('data/malignant', p)); true_labels.append(1)
true_labels = np.array(true_labels)

# 2) Inferencia y guardado
models = load_models()
all_preds = {'true': true_labels}

for name, model in models.items():
    probs = []
    for case in tqdm(test_cases, desc=f'{name}'):
        vol, _ = load_and_preprocess_ct_scan(case)
        _, prob, _ = predict_volume(model, vol)
        probs.append(prob)
    all_preds[name] = np.array(probs)

# 3) Volcado a disco
np.savez('predictions.npz', **all_preds)
print("Predicciones guardadas en predictions.npz")
