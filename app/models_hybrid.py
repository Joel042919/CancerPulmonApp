# app/models_hybrid.py
import torch, torch.nn as nn
from monai.networks.nets import resnet

class FlexibleSE(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.gap3d = nn.AdaptiveAvgPool3d(1)   # para 5-D
        self.gap2d = nn.AdaptiveAvgPool2d(1)   # para 4-D
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch // r, ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() == 5:                       # (B,C,D,H,W)
            z = self.gap3d(x).flatten(1)       # (B,C)
            w = self.fc(z).view(x.size(0), -1, 1, 1, 1)
            return x * w                       # (B,C,D,H,W)
        elif x.dim() == 4:                     # (B,C,H,W)
            z = self.gap2d(x).flatten(1)       # (B,C)
            w = self.fc(z).view(x.size(0), -1, 1, 1)
            return x * w                       # (B,C,H,W)
        else:                                  # (B,C)
            return x * self.fc(x)              # (B,C)

class ResNet50SE(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet.resnet50(
            spatial_dims=3, n_input_channels=1, num_classes=0
        )
        self.backbone.fc = nn.Identity() 
        self.se   = FlexibleSE(2048)
        self.cls  = nn.Linear(2048, 1)
        self._trace = []

    def forward(self, x):
        self._trace.clear()
        for name, layer in self.backbone._modules.items():
            x = layer(x)
            self._trace.append((name, x.detach()))
        x = self.se(x);       self._trace.append(("se_out", x.detach()))
        x = x.mean(dim=(2,3,4))                  # GAP 3-D  → (B,2048)
        out = self.cls(x)
        return out               # (B,1) logits

#b) DenseNet121 3D + Autoencoder de características

# -- al final de models_hybrid.py ------------------------------------
from monai.networks.nets import densenet

class DenseNet121AE(nn.Module):
    """
    DenseNet121 3-D como encoder + pequeño decoder up-sampling.
    Entrada  : (B,1,D,H,W)
    Salida   : logits (B,1)
    """
    def __init__(self):
        super().__init__()

        # 1) Encoder
        self.encoder = densenet.densenet121(
            spatial_dims=3, in_channels=1, out_channels=1  # out=1 → fc pero lo anulamos
        )
        self.encoder.class_layers = nn.Identity()         # quita la FC → deja mapa (B,1024,d,h,w)

        # 2) Decoder: 3× up-conv
        self.dec = nn.Sequential(
            nn.ConvTranspose3d(1024, 512, 2, stride=2), nn.BatchNorm3d(512), nn.ReLU(inplace=True),
            nn.ConvTranspose3d(512, 256, 2, stride=2),  nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose3d(256, 128, 2, stride=2),  nn.BatchNorm3d(128), nn.ReLU(inplace=True),
        )

        # 3) Fusion + clasificación
        self.gap = nn.AdaptiveAvgPool3d(1)      # GAP para encoder y decoder
        self.cls = nn.Linear(1024 + 128, 1)     # 1152→1 o usa 2048 si cambias canales

    def forward(self, x):
        f_enc = self.encoder.features(x)        # (B,1024,6,6,6)
        f_dec = self.dec(f_enc)                 # (B,128,24,24,24)

        vec = torch.cat(
            [self.gap(f_enc).flatten(1), self.gap(f_dec).flatten(1)],
            dim=1
        )                                       # (B,1152)
        return self.cls(vec)

