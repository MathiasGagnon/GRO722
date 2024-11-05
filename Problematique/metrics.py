# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
import torch
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as conf_matrix
# import sklearn.metrics as metrics

def edit_distance(a,b):
    # Calcul de la distance d'édition
    dp = np.arange(len(b) + 1)
    for i, char_a in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, char_b in enumerate(b, 1):
            insert, delete, replace = dp[j] + 1, dp[j - 1] + 1, prev + (char_a != char_b)
            prev, dp[j] = dp[j], min(insert, delete, replace)
    return dp[-1]
    pass

def confusion_matrix(true, pred, ignore=[]):
    # Calcul de la matrice de confusion
    # Convert tensors to numpy arrays if needed
    true_np = true.numpy() if isinstance(true, torch.Tensor) else true
    pred_np = pred.numpy() if isinstance(pred, torch.Tensor) else pred

    # Generate the confusion matrix
    cm = conf_matrix(true_np, pred_np)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(true_np), 
                yticklabels=np.unique(true_np))
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
