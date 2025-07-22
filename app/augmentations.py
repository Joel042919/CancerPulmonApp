import numpy as np
from monai.transforms import (
    Compose, RandAffined, Rand3DElasticd, RandFlipd, RandGaussianNoised
)
from monai.config import KeysCollection

def get_train_transforms(keys: KeysCollection = ("image",)):
    return Compose([
        # --- Aumentos geométricos solicitados ---
        RandAffined(
            keys, prob=0.9,
            rotate_range=(0, 0, np.pi/12),        # ±15°
            scale_range=(0.15, 0.15, 0.15),       # zoom ±15 %
            translate_range=(10, 10, 10)          # pequeños desplazamientos
        ),
        # Opcional : ruido y flips
        Rand3DElasticd(keys, prob=0.3, sigma_range=(5, 7), magnitude_range=(50, 100)),
        RandFlipd(keys, prob=0.5, spatial_axis=2),
        RandGaussianNoised(keys, prob=0.2, mean=0., std=0.01),
    ])
