#import tensorflow as tf
from keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)
import os, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.model_utils_pt import load_models
from app.preprocessing import load_and_preprocess_ct_scan
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score


def load_dataset(data_dir, test_size=0.2, random_state=42, max_cases=None):
    """Carga y divide el dataset en entrenamiento y validación"""
    # Obtener listas de casos benignos y malignos
    benign_cases = [os.path.join(data_dir, 'benign_2', f) for f in os.listdir(os.path.join(data_dir, 'benign_2'))]
    malignant_cases = [os.path.join(data_dir, 'malignant_2', f) for f in os.listdir(os.path.join(data_dir, 'malignant_2'))]
    
    # Crear etiquetas
    benign_labels = [0] * len(benign_cases)
    malignant_labels = [1] * len(malignant_cases)
    
    # Combinar y dividir
    all_cases = benign_cases + malignant_cases
    all_labels = benign_labels + malignant_labels
    
    if max_cases:                      # ← NUEVO
        rng = np.random.RandomState(random_state)
        idx = rng.permutation(len(all_cases))[:max_cases]
        all_cases  = [all_cases[i]  for i in idx]
        all_labels = [all_labels[i] for i in idx]

    # Dividir en train y test
    train_cases, val_cases, train_labels, val_labels = train_test_split(
        all_cases, all_labels, test_size=test_size, random_state=random_state, stratify=all_labels
    )
    
    return train_cases, val_cases, train_labels, val_labels



class CTScanDataset(Dataset):                      #128,128,128
    def __init__(self, cases, labels, target_size=(128,128,128)):
        self.cases = cases
        self.labels = labels
        self.target_size = target_size
    def __len__(self):
        return len(self.cases)
    def __getitem__(self, idx):
        vol, _ = load_and_preprocess_ct_scan(self.cases[idx], self.target_size)
        vol = np.expand_dims(vol, 0)               # (1,D,H,W)
        return torch.from_numpy(vol).float(), torch.tensor(self.labels[idx]).float()



def train_model(model, train_generator, val_generator, epochs=100, model_name='lung_cancer_model'):
    """Entrena un modelo con los generadores de datos"""
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            f'models/{model_name}.h5',
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir=f'logs/{model_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            histogram_freq=1
        )
    ]
    
    # Entrenamiento
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Guardar historial
    pd.DataFrame(history.history).to_csv(f'models/{model_name}_history.csv', index=False)
    
    return history



def train_model_pt(model, train_loader, val_loader, epochs=20,
                   model_name='lung_cancer_model', lr=1e-4):
    device = next(model.parameters()).device
    # en train_model_pt
    pos = sum(train_labels); neg = len(train_labels)-pos
    pos_weight = torch.tensor([neg/pos]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_auc = 0
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)  # (B,1)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        # --- validación rápida ---
        model.eval(); val_loss, gt, pred = 0, [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).unsqueeze(1)
                logits = model(x); val_loss += criterion(logits, y).item()
                preds = (torch.sigmoid(logits) > 0.5).int()
                correct += (preds == y.int()).sum().item(); total += y.size(0)
                val_loss += criterion(logits, y).item()
                pred.extend(torch.sigmoid(logits).cpu().numpy().ravel())
                gt.extend(y.cpu().numpy())
        val_loss /= len(val_loader)
        val_auc   = roc_auc_score(gt, pred)
        print(f"Epoch {epoch+1}/{epochs}  "
              f"val_loss={val_loss/len(val_loader):.4f}  "
              f"val_acc={correct/total:.3f}")
        if val_auc > best_auc:                # ⇦ usa AUC
            best_auc = val_auc
            torch.save(model.state_dict(),
                       f"models/{model_name}_best.pth")
            print(f"  🔖  Nuevo mejor AUC ({best_auc:.3f}) → modelo guardado")

    # Guarda pesos
    #torch.save(model.state_dict(), f"models/{model_name}.pth")



if __name__ == "__main__":
    # Configuración
    data_dir = 'data'  # Directorio con dataset LIDC-IDRI
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Cargar dataset
    train_cases, val_cases, train_labels, val_labels = load_dataset(data_dir, max_cases=100)
    
    # Crear generadores
    #train_gen = CTScanGenerator(train_cases, train_labels, batch_size=4)
    #val_gen = CTScanGenerator(val_cases, val_labels, batch_size=4)
    train_ds = CTScanDataset(train_cases, train_labels,target_size=(96,96,96))
    val_ds   = CTScanDataset(val_cases, val_labels,target_size=(96,96,96))
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=os.cpu_count()//2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=2, shuffle=False, num_workers=os.cpu_count()//2, pin_memory=True)

    # Cargar y entrenar modelos
    models = load_models()
    #for name, model in models.items():
    #    print(f"\nEntrenando modelo: {name}")
    #    history = train_model_pt(
    #        model,
    #        train_ds,
    #        val_ds,
    #        epochs=20,
    #        model_name=f"{name.lower().replace(' ', '_')}_lung_cancer"
    #    )
    torch.set_num_threads(min(32, os.cpu_count()))
    for name, model in models.items():
        print(f"\nEntrenando modelo: {name}")
        train_model_pt(
            model,
            train_loader,
            val_loader,
            epochs=15,
            model_name=f"{name.lower().replace(' ', '_')}_lung_cancer"
        )
