# ──────────────────────────────────────────────────────────────────────
# 1. IMPORTS & CONFIG
# ──────────────────────────────────────────────────────────────────────
import os, sys, random, logging, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import numpy as np, torch, pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from app.augmentations import get_train_transforms

# Ajusta aquí:
DATA_DIR    = "data"
TARGET_SHAPE = (96, 96, 96)
BATCH_SIZE   = 2
EPOCHS_MAX   = 50
LR_INIT      = 3e-4
EARLY_STOP   = 6            # épocas sin mejora
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS  = 0            # 0 = CPU-friendly en Windows

# ──────────────────────────────────────────────────────────────────────
# 2. DATASET
# ──────────────────────────────────────────────────────────────────────
from app.preprocessing import load_and_preprocess_ct_scan

class CTScanDataset(Dataset):
    def __init__(self, cases, labels, target_size=TARGET_SHAPE,transforms=None):
        self.cases=cases 
        self.labels = labels
        self.target_size=target_size
        self.transforms = transforms

    def __len__(self): return len(self.cases)

    def __getitem__(self, idx):
        vol, _ = load_and_preprocess_ct_scan(
            self.cases[idx], self.target_size
        )

        #if self.augment:
        #    if random.random() < .5: vol = np.flip(vol, axis=0)
        #    if random.random() < .5: vol = np.rot90(vol, 1, (1, 2))

        vol = np.expand_dims(vol, 0)             # (1, D, H, W)
        #vol = np.ascontiguousarray(vol)          # evita strides negativos
        if self.transforms:
            vol = self.transforms({"image": vol})["image"]

        if isinstance(vol, torch.Tensor):          # ya es Tensor / MetaTensor
            vol_tensor = vol.float()
        else:                                      # es ndarray
            vol_tensor = torch.from_numpy(
                np.ascontiguousarray(vol)
            ).float()

        label_tensor = torch.tensor(self.labels[idx]).float()
        return vol_tensor, label_tensor
    
        #return torch.from_numpy(vol).float(), torch.tensor(self.labels[idx]).float()

def load_dataset(data_dir, test_size=.2, random_state=42):
    b_dir = os.path.join(data_dir, "benign_2")
    m_dir = os.path.join(data_dir, "malignant_2")
    benign    = [os.path.join(b_dir, f) for f in os.listdir(b_dir)]
    malignant = [os.path.join(m_dir, f) for f in os.listdir(m_dir)]
    labels = [0]*len(benign) + [1]*len(malignant)
    cases  = benign + malignant
    return train_test_split(cases, labels, test_size=test_size,
                            stratify=labels, random_state=random_state)

# ──────────────────────────────────────────────────────────────────────
# 3. TRAINER
# ──────────────────────────────────────────────────────────────────────
def train_model_pt(model, train_loader, val_loader,
                   epochs=EPOCHS_MAX, model_name="model"):
    model.to(DEVICE)
    # ─── Loss con pos_weight
    y_train = np.array(train_loader.dataset.labels)
    pos_w   = (len(y_train)-y_train.sum()) / y_train.sum()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w]).to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.3, patience=3, min_lr=1e-6)

    best_auc, epochs_no_imp, log = 0.0, 0, []

    for epoch in range(1, epochs+1):
        # ——— TRAIN ———
        model.train(); train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Ep {epoch}/{epochs}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ——— VALIDATE ———
        model.eval(); val_loss, preds, gts = 0.0, [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
                logits = model(x)
                val_loss += criterion(logits, y).item()
                prob = torch.sigmoid(logits).cpu().numpy().ravel()
                preds.extend(prob)
                gts.extend(y.cpu().numpy().ravel())

        val_loss /= len(val_loader)
        val_auc   = roc_auc_score(gts, preds)
        val_acc   = ((np.array(preds) > 0.5) == np.array(gts)).mean()
        scheduler.step(val_auc)

        print(f"Epoch {epoch:02d}  train_loss {train_loss/len(train_loader):.4f}  "
              f"val_loss {val_loss:.4f}  val_auc {val_auc:.3f}  val_acc {val_acc:.3f}")

        log.append([epoch, train_loss/len(train_loader), val_loss, val_auc, val_acc])

        # ——— Checkpoint / Early-Stop ———
        if val_auc > best_auc + 1e-4:          # mejora mínima
            best_auc = val_auc; epochs_no_imp = 0
            torch.save(model.state_dict(), f"models/{model_name}_best.pth")
            print(f"  🔖  Nuevo mejor AUC = {best_auc:.3f} → modelo guardado")
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= EARLY_STOP:
                print("⏹ Early-Stopping: sin mejora en", EARLY_STOP, "épocas")
                break

    pd.DataFrame(log, columns=["epoch","train_loss","val_loss","val_auc","val_acc"])\
      .to_csv(f"models/{model_name}_history.csv", index=False)

# ──────────────────────────────────────────────────────────────────────
# 4. MAIN
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    tr_c, val_c, y_tr, y_val = load_dataset(DATA_DIR)
    train_ds = CTScanDataset(
                    tr_c, 
                    y_tr, 
                    transforms=get_train_transforms()   # ← AUGMENTACIONES
                )
    val_ds   = CTScanDataset(val_c, y_val, transforms=None)

    # sampler balanceado
    class_counts = np.bincount(y_tr)
    weights = 1.0 / class_counts[y_tr]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    from app.model_utils_pt import load_models
    torch.set_num_threads(min(32, os.cpu_count()))
    for name, net in load_models().items():
        print(f"\n==== Entrenando {name} ====")
        train_model_pt(net, train_dl, val_dl,
                       epochs=EPOCHS_MAX,
                       model_name=name.lower().replace(" ", "_"))
