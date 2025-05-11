# visualization.py
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os

def plot_umap_embeddings(features, labels, model_name, save_path="../results/umap_visualizations/", show_plot=False):
    """
    Generate and save UMAP visualization of feature embeddings.
    
    Args:
        features (np.array): Feature vectors (2D array: n_samples x n_features)
        labels (np.array): Ground truth labels
        model_name (str): Name of the model (for title and filename)
        save_path (str): Directory to save the plot
        show_plot (bool): Whether to display the plot interactively
    """
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Apply UMAP
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    umap_embeddings = umap_model.fit_transform(features)
    
    # Plot settings
    plt.figure(figsize=(12, 10))
    scatter = sns.scatterplot(
        x=umap_embeddings[:, 0],
        y=umap_embeddings[:, 1],
        hue=labels,
        palette='Set1',
        s=60,
        alpha=0.8
    )
    
    # Formatting
    plt.title(f'UMAP: {model_name} Feature Embeddings', fontsize=15)
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save and optionally show
    plt.savefig(f"{save_path}{model_name}_umap.png", bbox_inches='tight', dpi=300)
    if show_plot:
        plt.show()
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name, class_names=None, save_path="../results/confusion_matrices/", show_plot=False):
    """
    Generate and save a confusion matrix plot.
    
    Args:
        y_true (np.array): Ground truth labels
        y_pred (np.array): Predicted labels
        model_name (str): Name of the model (for title and filename)
        class_names (list): List of class names for labeling
        save_path (str): Directory to save the plot
        show_plot (bool): Whether to display the plot interactively
    """
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot settings
    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.2)
    
    # Create heatmap
    heatmap = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names if class_names else np.unique(y_true),
        yticklabels=class_names if class_names else np.unique(y_true)
    )
    
    # Formatting
    plt.title(f'Confusion Matrix: {model_name}', fontsize=15)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Save and optionally show
    plt.savefig(f"{save_path}{model_name}_cm.png", bbox_inches='tight', dpi=300)
    if show_plot:
        plt.show()
    plt.close()

def save_visualizations(model, x_data, y_true, y_pred, model_name, class_names=None):
    """
    Convenience function to save both UMAP and confusion matrix visualizations.
    
    Args:
        model: Trained model (to extract features)
        x_data: Input data
        y_true: Ground truth labels
        y_pred: Predicted labels
        model_name: Name of the model
        class_names: List of class names
    """
    # Get feature embeddings (second-to-last layer)
    feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    features = feature_model.predict(x_data)
    
    # Generate visualizations
    plot_umap_embeddings(features, y_true, model_name)
    plot_confusion_matrix(y_true, y_pred, model_name, class_names)
