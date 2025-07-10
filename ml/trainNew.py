import os
import sys
from datetime import datetime

# ------------------ CPU OPTIMIZATIONS ------------------
import numpy as np
# Ajuste de hilos para BLAS/OpenMP
num_threads = os.cpu_count() or 1
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["MKL_NUM_THREADS"] = str(num_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)

# PyTorch configuración de hilos
import torch
torch.set_num_threads(num_threads)
torch.set_num_interop_threads(1)

# ------------------ PATHS ------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ------------------ Imports de librerías ------------------
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Tus módulos 3D
from app.model_utils_pt import load_models          # carga dict{name: nn.Module}
from app.preprocessing import load_and_preprocess_ct_scan

# ------------------ Carga de datos ------------------
def load_train_val(data_dir, val_size=0.2, random_state=42, max_cases=None):
    """
    Lee benign/ y malignant/ (80%) y separa en train/val.
    """
    benign_dir = os.path.join(data_dir, 'benign')
    malignant_dir = os.path.join(data_dir, 'malignant')

    benign = [os.path.join(benign_dir, f) for f in os.listdir(benign_dir)]
    malignant = [os.path.join(malignant_dir, f) for f in os.listdir(malignant_dir)]

    all_cases = benign + malignant
    all_labels = [0]*len(benign) + [1]*len(malignant)

    if max_cases is not None:
        rng = np.random.RandomState(random_state)
        idx = rng.permutation(len(all_cases))[:max_cases]
        all_cases = [all_cases[i] for i in idx]
        all_labels = [all_labels[i] for i in idx]

    train_c, val_c, train_l, val_l = train_test_split(
        all_cases, all_labels,
        test_size=val_size,
        stratify=all_labels,
        random_state=random_state
    )
    return train_c, val_c, train_l, val_l


def load_test(data_dir):
    """
    Lee benign_test/ y malignant_test/ como test final.
    """
    bt = os.path.join(data_dir, 'benign_test')
    mt = os.path.join(data_dir, 'malignant_test')

    benign_test = [os.path.join(bt, f) for f in os.listdir(bt)]
    malignant_test = [os.path.join(mt, f) for f in os.listdir(mt)]

    cases = benign_test + malignant_test
    labels = [0]*len(benign_test) + [1]*len(malignant_test)
    return cases, labels

# ------------------ Dataset PyTorch 3D ------------------
class CTScanDataset(Dataset):
    def __init__(self, cases, labels, target_size=(64,64,64)):
        self.cases = cases
        self.labels = labels
        self.target_size = target_size

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        vol, _ = load_and_preprocess_ct_scan(self.cases[idx], self.target_size)
        vol = np.expand_dims(vol, 0)               # (1, D, H, W)
        x = torch.from_numpy(vol).float()
        y = torch.tensor(self.labels[idx]).float()
        return x, y

# ------------------ Entrenamiento PyTorch 3D ------------------
def train_model_pt(model, train_loader, val_loader,
                   epochs=20, model_name='lung_cancer_model', lr=1e-4):
    """
    Entrena un modelo 3D en PyTorch. Guarda pesos en carpeta models/.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience = 7
    no_improve = 0

    os.makedirs('models', exist_ok=True)

    for ep in range(1, epochs+1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        # Validación
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device).unsqueeze(1)
                logits = model(x)
                val_loss += criterion(logits, y).item()
                preds = (torch.sigmoid(logits) > 0.5).int()
                correct += (preds == y.int()).sum().item()
                total += y.size(0)
        avg_val = val_loss / len(val_loader)
        acc_val = correct / total
        print(f"[Epoch {ep}/{epochs}] val_loss={avg_val:.4f} val_acc={acc_val:.2%}")

        # Chequeo de EarlyStopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            no_improve = 0
            torch.save(model.state_dict(), f"models/{model_name}.pth")
            print(f"✔ Guardado mejor modelo: models/{model_name}.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"⚠️ Stop early por {patience} epochs sin mejora.")
                break

# ------------------ Main ------------------
if __name__ == "__main__":
    data_dir = 'data'  # Debe tener benign/, malignant/, benign_test/, malignant_test/

    # Split interno train/val
    train_cases, val_cases, train_labels, val_labels = load_train_val(
        data_dir, val_size=0.2, random_state=42, max_cases=None
    )
    # Test final
    test_cases, test_labels = load_test(data_dir)

    # DataLoaders CPU-optimizado
    train_ds = CTScanDataset(train_cases, train_labels)
    val_ds   = CTScanDataset(val_cases,   val_labels)
    train_loader = DataLoader(
        train_ds, batch_size=2, shuffle=True,
        num_workers=num_threads, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=2, shuffle=False,
        num_workers=num_threads, pin_memory=False
    )

    # Carga y entrenamiento de modelos 3D
    models = load_models()
    for name, model in models.items():
        print(f"\n=== Entrenando modelo: {name} ===")
        train_model_pt(
            model,
            train_loader,
            val_loader,
            epochs=15,
            model_name=name.lower().replace(' ', '_')
        )

    print("\n🎉 Entrenamiento completado.")
