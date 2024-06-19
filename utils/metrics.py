import numpy  as np
import pandas as pd
import sklearn.metrics
from matplotlib import pyplot as plt
from typing import Dict, Optional, List

def calc_metrics(true_labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    accuracy  = sklearn.metrics.accuracy_score(true_labels, preds)
    precision = sklearn.metrics.precision_score(true_labels, preds)
    recall    = sklearn.metrics.recall_score(true_labels, preds)
    micro_f1  = sklearn.metrics.f1_score(true_labels, preds, average='micro')
    met_f1    = sklearn.metrics.f1_score(true_labels, preds, average='binary')
    notmet_f1 = sklearn.metrics.f1_score(1-true_labels, 1-preds, average='binary')
    n2c2_f1   = (met_f1 + notmet_f1) / 2
    return {
        'accuracy' : accuracy,
        'precision' : precision,
        'recall' : recall,
        'micro_f1' : micro_f1,
        'n2c2-met-f1' : met_f1,
        'n2c2-notmet-f1' : notmet_f1,
        # NOTE: This is the metric reported in the N2C2 challenge
        # It is incorrect -- it is actually calculating the Macro F1 score
        # for the two binary classes. The Micro F1 score calculated above
        # is the correct "Micro-F1" metric to use.
        # `n2c2_f1 = sklearn.metrics.f1_score(df['true_label'], df['is_met'], average='macro')`
        'n2c2-micro-f1' : n2c2_f1,
    }

def plot_confusion_matrices(df: pd.DataFrame, file_name: Optional[str] = None):
    # Unique labels
    unique_labels = df['criterion'].unique()
    num_labels = len(unique_labels)

    # Creating subplots
    cols = 3
    rows = (num_labels + cols - 1) // cols  # Calculate rows needed
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()  # Flatten to 1D array for easy indexing

    for i, label in enumerate(unique_labels):
        # Create binary classification problem
        y_binary = df[df['criterion'] == label]['true_label']
        y_hat_binary = df[df['criterion'] == label]['is_met']

        # Compute confusion matrix
        cm = sklearn.metrics.confusion_matrix(y_binary, y_hat_binary)

        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
        axes[i].set_title(f'Confusion Matrix for Label: {label}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

    # Turn off any unused subplots
    for j in range(i + 1, rows * cols):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if file_name:
        plt.savefig(file_name)
    else:
        plt.show()