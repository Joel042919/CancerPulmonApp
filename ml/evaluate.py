import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import matthews_corrcoef
from statsmodels.stats.contingency_tables import mcnemar
import itertools
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
# ahora usamos predict_volume
import os, sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from app.model_utils_pt import load_models, predict_volume  
from app.preprocessing import load_and_preprocess_ct_scan
from tqdm import tqdm

def load_test_dataset(test_dir):
    """Carga el conjunto de prueba desde el directorio especificado"""
    test_cases = []
    test_labels = []
    
    # Cargar casos benignos
    benign_dir = os.path.join(test_dir, 'benign_test')
    for case in tqdm(os.listdir(benign_dir), desc='Loading benign cases'):
        test_cases.append(os.path.join(benign_dir, case))
        test_labels.append(0)
    
    # Cargar casos malignos
    malignant_dir = os.path.join(test_dir, 'malignant_test')
    for case in tqdm(os.listdir(malignant_dir), desc='Loading malignant cases'):
        test_cases.append(os.path.join(malignant_dir, case))
        test_labels.append(1)
    
    return test_cases, test_labels

def save_mcnemar_plots(all_pred, true_labels, output_dir):
    os.makedirs(os.path.join(output_dir, 'figures', 'mcnemar'), exist_ok=True)
    for (name_i, pred_i), (name_j, pred_j) in itertools.combinations(all_pred.items(), 2):
        # construir tabla de contingencia
        table = [
            [int(sum((pred_i==1)&(pred_j==1)&(true_labels==1))),
             int(sum((pred_i==1)&(pred_j==0)&(true_labels==true_labels)))],
            [int(sum((pred_i==0)&(pred_j==1)&(true_labels==true_labels))),
             int(sum((pred_i==0)&(pred_j==0)&(true_labels==true_labels)))]
        ]
        # mejor usar solo b y c para McNemar (desacuerdos)
        tb = [[int(sum((pred_i==true_labels)&(pred_j==true_labels))),
               int(sum((pred_i==true_labels)&(pred_j!=true_labels)))],
              [int(sum((pred_i!=true_labels)&(pred_j==true_labels))),
               int(sum((pred_i!=true_labels)&(pred_j!=true_labels)))]]
        result = mcnemar(tb, exact=False)
        # dibujar tabla y anotación
        fig, ax = plt.subplots(figsize=(5,4))
        ax.axis('off')
        tbl = ax.table(cellText=tb,
                       rowLabels=[f'{name_i} acierto', f'{name_i} error'],
                       colLabels=[f'{name_j} acierto', f'{name_j} error'],
                       loc='center')
        plt.title(f'McNemar: {name_i} vs {name_j}\nχ²={result.statistic:.2f}, p={result.pvalue:.3f}')
        fn = f'{output_dir}/figures/mcnemar/mcnemar_{name_i.lower().replace(" ","_")}_{name_j.lower().replace(" ","_")}.png'
        fig.savefig(fn, bbox_inches='tight')
        plt.close(fig)



def evaluate_models(test_dir, output_dir='reports'):
    """Evalúa todos los modelos en el conjunto de prueba"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    # Cargar datos de prueba
    test_cases, test_labels = load_test_dataset(test_dir)
    
    # Cargar modelos
    models = load_models()
    results = []
    
    all_pred_labels = {}  # name -> array de 0/1

    for name, model in models.items():
        print(f"\nEvaluando modelo: {name}")
        
        # Predecir
        predictions = []
        for case in tqdm(test_cases, desc=f'Predicting with {name}'):
            #volume = np.expand_dims(np.expand_dims(volume, axis=-1), axis=0)
            #pred = model.predict(volume)[0][0]
            #predictions.append(pred)
            # inferencia unificada
            
            #volume, _ = load_and_preprocess_ct_scan(case)
            #_, prob, _ = predict_volume(model, volume)
            #predictions.append(prob)
            try:
                volume, _ = load_and_preprocess_ct_scan(case,target_size=(96,96,96))
                _, prob, _ = predict_volume(model, volume)
                predictions.append(prob)
            except Exception as e:
                print(f"[{case}] omitido ({e})")
                continue
        
        predictions = np.array(predictions)
        predicted_labels = (predictions > 0.5).astype(int)

        # Calcular MCC
        mcc = matthews_corrcoef(test_labels, predicted_labels)

        # Métricas
        report = classification_report(
            test_labels,
            predicted_labels,
            target_names=['benign', 'malignant'],
            output_dict=True
        )
        
        # Matriz de confusión
        cm = confusion_matrix(test_labels, predicted_labels)
        
        # Curva ROC
        fpr, tpr, _ = roc_curve(test_labels, predictions)
        roc_auc = auc(fpr, tpr)
        
        # Curva Precision-Recall
        precision, recall, _ = precision_recall_curve(test_labels, predictions)
        ap_score = average_precision_score(test_labels, predictions)
        
        # Guardar resultados
        results.append({
            'Modelo': name,
            'Precisión': report['accuracy'],
            'Sensibilidad': report['malignant']['recall'],
            'Especificidad': report['benign']['recall'],
            'F1-Score': report['malignant']['f1-score'],
            'AUC-ROC': roc_auc,
            'AP': ap_score,
            'Dice Score': calculate_dice_score(test_labels, predicted_labels),
            'MCC': mcc
        })
        all_pred_labels[name]=predicted_labels
        
        # Guardar gráficos
        save_plots(name, cm, fpr, tpr, roc_auc, precision, recall, ap_score, output_dir)
    
    # DataFrame final
    df = pd.DataFrame(results)
    df.to_csv(f'{output_dir}/model_comparison.csv', index=False)

    # Tests de McNemar entre cada par de modelos
    save_mcnemar_plots(all_pred_labels, np.array(test_labels), output_dir)

    return df

def calculate_dice_score(true_labels, pred_labels):
    """Calcula el Dice Score para evaluación de segmentación"""
    intersection = np.sum(true_labels * pred_labels)
    return (2. * intersection) / (np.sum(true_labels) + np.sum(pred_labels))

def save_plots(model_name, cm, fpr, tpr, roc_auc, precision, recall, ap_score, output_dir):
    """Guarda gráficos de evaluación"""
    # Matriz de confusión
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benigno', 'Maligno'],
                yticklabels=['Benigno', 'Maligno'])
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.savefig(f'{output_dir}/figures/cm_{model_name.lower().replace(" ", "_")}.png', bbox_inches='tight')
    plt.close()
    
    # Curva ROC
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Curva ROC - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'{output_dir}/figures/roc_{model_name.lower().replace(" ", "_")}.png', bbox_inches='tight')
    plt.close()
    
    # Curva Precision-Recall
    plt.figure(figsize=(8,6))
    plt.plot(recall, precision, label=f'PR curve (AP = {ap_score:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Curva Precision-Recall - {model_name}')
    plt.legend(loc="upper right")
    plt.savefig(f'{output_dir}/figures/pr_{model_name.lower().replace(" ", "_")}.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    test_dir = 'data'  # Directorio con datos de prueba
    output_dir = 'reports'
    
    # Evaluar modelos
    results_df = evaluate_models(test_dir, output_dir)
    
    # Guardar resultados
    results_df.to_csv(f'{output_dir}/model_comparison.csv', index=False)
    
    # Mostrar resultados
    print("\nResultados de Evaluación:")
    print(results_df.to_markdown(index=False))
