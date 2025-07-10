import numpy as np
import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)
import pydicom
import os
import zipfile
from skimage.transform import resize
from skimage.exposure import equalize_adapthist


def load_dicom_series(directory):
    """
    Recorre recursivamente `directory`, agrupa los ficheros DICOM por
    SeriesInstanceUID y carga la serie que tenga más cortes.
    Devuelve (volume, spacing) con ejes (X,Y,Z).
    """
    import pydicom

    # 1. Recoger todos los .dcm del árbol
    dicom_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(directory)
        for f in files
        if f.lower().endswith('.dcm')
    ]
    if not dicom_files:
        raise RuntimeError(f"No se encontraron DICOM en {directory}")

    # 2. Agrupar por SeriesInstanceUID
    series_dict = {}
    for fp in dicom_files:
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True,
                                 specific_tags=['SeriesInstanceUID', 'InstanceNumber'])
            uid = ds.SeriesInstanceUID
            inst = ds.get('InstanceNumber', 0)
            series_dict.setdefault(uid, []).append((inst, fp))
        except Exception:
            # archivo corrupto o no DICOM: lo ignoramos
            continue

    if not series_dict:
        raise RuntimeError(f"No se encontró ninguna serie válida en {directory}")

    # 3. Elegir la serie más “grande”
    best_uid = max(series_dict, key=lambda u: len(series_dict[u]))
    best_files = [fp for _, fp in sorted(series_dict[best_uid],
                                         key=lambda t: t[0])]   # orden por InstanceNumber

    # 4. Cargar con SimpleITK
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(best_files)
    image = reader.Execute()

    volume = sitk.GetArrayFromImage(image).transpose((2, 1, 0))  # (X,Y,Z)
    spacing = image.GetSpacing()
    return volume, spacing


def load_dicom_series_REPARANDO(directory):
    """Carga una serie DICOM desde un directorio, incluso si hay subcarpetas."""
    reader = sitk.ImageSeriesReader()
    # obtiene todas las SeriesUIDs en el directorio
    series_IDs = reader.GetGDCMSeriesIDs(directory)
    if not series_IDs:
        raise RuntimeError(f"No se encontró ninguna serie DICOM en {directory}")
    # escoge la primera serie
    series_files = reader.GetGDCMSeriesFileNames(directory, series_IDs[0])
    reader.SetFileNames(series_files)
    image = reader.Execute()

    volume = sitk.GetArrayFromImage(image)
    # SimpleITK usa (z,y,x); tu pipeline espera (x,y,z)
    volume = volume.transpose((2,1,0))
    spacing = image.GetSpacing()
    return volume, spacing
                                            #target_size=(128, 128, 128)
def load_and_preprocess_ct_scan(input_path, target_size=(64,64,64)):
    
    #Carga y preprocesa un volumen de TC
    #Args:
    #    input_path: Ruta al archivo DICOM, directorio o zip
    #    target_size: Tamaño objetivo para el volumen
    #Returns:
    #    volume_preprocessed: Volumen preprocesado
    #    original_volume: Volumen original (redimensionado)
    
    # Cargar datos según el tipo de entrada
    if isinstance(input_path, str) and input_path.endswith('.zip'):
        # Extraer archivo zip
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            temp_dir = 'temp_dicom'
            zip_ref.extractall(temp_dir)
            input_path = temp_dir
    
    if os.path.isdir(input_path):
        # BUSCAR .mhd RECURSIVAMENTE
        mhd_file = None
        for root, dirs, files in os.walk(input_path):
            for f in files:
                if f.lower().endswith('.mhd'):
                    mhd_file = os.path.join(root, f)
                    break
            if mhd_file:
                break

        if mhd_file:
            # Leemos el par .mhd/.raw
            image = sitk.ReadImage(mhd_file)
            volume = sitk.GetArrayFromImage(image)      # (Z,Y,X)
            spacing = image.GetSpacing()
        else:
            # Si no hay .mhd, caemos a DICOM
            volume, spacing = load_dicom_series(input_path)
    else:
        # Cargar archivo DICOM individual (no recomendado)
        ds = pydicom.dcmread(input_path)
        volume = ds.pixel_array
        spacing = (1.0, 1.0, 1.0)  # Asumir si no hay información
    
    # Guardar volumen original (para visualización)
    original_volume = resize(volume, target_size, mode='reflect', anti_aliasing=True)
    
    # Preprocesamiento
    volume_preprocessed = preprocess_volume(volume, spacing, target_size)
    
    return volume_preprocessed, original_volume

def preprocess_volume(volume, spacing, target_size):
    """Aplica preprocesamiento a un volumen CT"""
    # 1. Normalizar intensidades (ventana pulmonar)
    volume = apply_lung_window(volume)
    
    # 2. Resample para tamaño de voxel isotrópico
    volume = resample_volume(volume, spacing)
    
    # 3. Redimensionar al tamaño objetivo
    volume = resize(volume, target_size, mode='reflect', anti_aliasing=True)
    
    # 4. Ecualización adaptativa de histograma (slice por slice)
    for i in range(volume.shape[0]):
        volume[i] = equalize_adapthist(volume[i])
    
    # 5. Normalización [0,1]
    volume = (volume - volume.min()) / (volume.max() - volume.min())
    
    return volume

def apply_lung_window(volume, level=-600, width=1500):
    """Aplica ventana pulmonar a un volumen CT"""
    window_min = level - width / 2
    window_max = level + width / 2
    
    volume = np.clip(volume, window_min, window_max)
    volume = (volume - window_min) / (window_max - window_min)
    
    return volume

def resample_volume(volume, original_spacing, target_spacing=1.0):
    """
    Re-muestrea el volumen a un voxel isotrópico.  
    Si alguna dimensión resulta 0, devuelve el volumen original.
    """
    # Reemplaza espaciados inválidos (0 o None) por 1 mm
    safe_spacing = tuple(s if (s and s > 0) else 1.0 for s in original_spacing)

    resize_factor = [s * target_spacing for s in safe_spacing]

    new_shape = (
        int(max(1, volume.shape[0] * resize_factor[0])),
        int(max(1, volume.shape[1] * resize_factor[1])),
        int(max(1, volume.shape[2] * resize_factor[2])),
    )

    # Si alguna dimensión quedó en 0, aborta el remuestreo
    if 0 in new_shape:
        return volume

    return resize(
        volume,
        new_shape,
        mode="reflect",
        anti_aliasing=True,
    )

