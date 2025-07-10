# stats_tests.py
import numpy as np
import itertools
from sklearn.metrics import matthews_corrcoef
from statsmodels.stats.contingency_tables import mcnemar

# 1) Carga
data = np.load('predictions.npz')
true = data['true']  # shape (N,)

# Lista de nombres de modelos (nombres de arrays dentro del .npz, salvo 'true')
models = [k for k in data.files if k != 'true']

# 2) MCC para cada modelo
print("=== MCC por modelo ===")
for name in models:
    preds = (data[name] > 0.5).astype(int)
    mcc = matthews_corrcoef(true, preds)
    print(f"{name:25s}: {mcc:.3f}")

# 3) McNemar entre pares de modelos
print("\n=== McNemar entre pares ===")
for m1, m2 in itertools.combinations(models, 2):
    p1 = (data[m1] > 0.5).astype(int)
    p2 = (data[m2] > 0.5).astype(int)

    # Tabla de contingencia b y c  
    # b = casos donde m1 acierta y m2 falla  
    # c = casos donde m1 falla y m2 acierta
    b = np.sum((p1 == true) & (p2 != true))
    c = np.sum((p1 != true) & (p2 == true))
    table = [[0, b],
             [c, 0]]
    result = mcnemar(table, exact=True)

    print(f"{m1:15s} vs {m2:15s} →  b={b:3d}, c={c:3d}, "
          f"χ²={result.statistic:.2f}, p={result.pvalue:.4f}")
