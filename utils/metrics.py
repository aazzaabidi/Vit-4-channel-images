# metrics.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score
)
import pandas as pd
import os

def calculate_metrics(y_true, y_pred, model_name, save_path="../results/"):
    """
    Calcule et sauvegarde toutes les métriques d'évaluation.
    
    Args:
        y_true (array): Étiquettes réelles
        y_pred (array): Prédictions du modèle
        model_name (str): Nom du modèle (ex: "ViT")
        save_path (str): Chemin de sauvegarde des résultats
    """
    # Création des dossiers si inexistants
    os.makedirs(f"{save_path}confusion_matrices/", exist_ok=True)
    os.makedirs(f"{save_path}per_class_metrics/", exist_ok=True)
    
    # 1. Métriques globales
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro"), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro"), 4)
    }
    
    # 2. Matrice de confusion
    plot_confusion_matrix(y_true, y_pred, model_name, save_path)
    
    # 3. Rapport par classe
    save_classification_report(y_true, y_pred, model_name, save_path)
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """Génère et sauvegarde la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(f"{save_path}confusion_matrices/{model_name}.png", bbox_inches="tight")
    plt.close()

def save_classification_report(y_true, y_pred, model_name, save_path):
    """Génère un rapport détaillé par classe."""
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(f"{save_path}per_class_metrics/{model_name}_report.csv", index=True)

def print_metrics_summary(metrics):
    """Affiche un résumé des métriques principales."""
    print(f"\n{' Metric ':=^40}")
    print(f"Accuracy: {metrics['accuracy']}%")
    print(f"F1-Score (macro): {metrics['f1_macro']}")
    print(f"Precision (macro): {metrics['precision_macro']}")
    print(f"Recall (macro): {metrics['recall_macro']}")
    print("=" * 40)
