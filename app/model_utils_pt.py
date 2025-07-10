import os
import numpy as np
import torch
import torch.nn as nn
from monai.networks.nets import resnet, densenet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_models():
    """Carga modelos 3D con MONAI (PyTorch)."""
    models = {}

    # 1) ResNet50 3D
    resnet3d = resnet.resnet50(
        spatial_dims=3,
        n_input_channels=1,
        num_classes=1,
        pretrained=False
    ).to(device)
    models['3D ResNet50'] = resnet3d

    # 2) DenseNet121 3D
    densenet3d = densenet.densenet121(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        pretrained=False
    ).to(device)
    models['3D DenseNet121'] = densenet3d

    # 3) CNN 3D Personalizada
    class CustomCNN3D(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv3d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(2),
                nn.BatchNorm3d(32),

                nn.Conv3d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(2),
                nn.BatchNorm3d(64),

                nn.Conv3d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(2),
                nn.BatchNorm3d(128),

                nn.AdaptiveAvgPool3d((1,1,1))
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 1)
            )

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x)

    custom_cnn3d = CustomCNN3D().to(device)
    models['CNN 3D Personalizada'] = custom_cnn3d

    # Cargar pesos (.pth) si existen
    for name, model in models.items():
        path = f'models/{name.lower().replace(" ", "_")}_lung_cancer.pth'
        if os.path.isfile(path):
            model.load_state_dict(torch.load(path, map_location=device))
        else:
            print(f"No se encontraron pesos preentrenados para {name}")

    return models

def predict_volume(model, volume):
    #Predicción + probabilidad + mapa de saliencia (PyTorch).
    model.eval()
    # volume: ndarray (D,H,W) o (D,H,W,1)

    vol = np.asarray(volume)  # (D,H,W) o (D,H,W,1)

    # 1) Asegurarnos de que volume es un ndarray de 3D (D,H,W) o 4D (D,H,W,1)
    vol = np.asarray(volume)
    if vol.ndim == 4 and vol.shape[-1] == 1:
        vol = vol[..., 0]       # de (D,H,W,1) a (D,H,W)

    # 2) Añadimos batch y canal: (1,1,D,H,W)
    vol = vol[None, None, ...]

    # 3) Convertimos a tensor
    input_tensor = torch.from_numpy(vol).float().to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.sigmoid(logits).cpu().numpy().ravel()[0]
        diagnosis = int(prob > 0.5)

    heatmap = generate_saliency_map(model, input_tensor)
    return diagnosis, float(prob), heatmap

def generate_saliency_map(model, input_tensor):
    """Grad-CAM simplificado por gradientes absolutos."""
    model.eval()
    inp = input_tensor.requires_grad_(True)
    logits = model(inp)
    prob = torch.sigmoid(logits)
    prob.backward()
    grads = inp.grad.abs().cpu().numpy()[0,0]  # (D,H,W)
    sal = (grads - grads.min()) / (grads.max() - grads.min() + 1e-8)
    return sal
