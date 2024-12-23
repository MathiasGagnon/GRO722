# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2021
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def edit_distance(a, b):
    dp = np.arange(len(b) + 1)
    for i, char_a in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, char_b in enumerate(b, 1):
            insert, delete, replace = (
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + (char_a != char_b),
            )
            prev, dp[j] = dp[j], min(insert, delete, replace)
    return dp[-1]
    pass


def confusion_matrix(true, pred, labels):
    true_np = true.numpy() if isinstance(true, torch.Tensor) else np.array(true)
    pred_np = pred.numpy() if isinstance(pred, torch.Tensor) else np.array(pred)
    
    num_classes = len(labels)
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for t_word, p_word in zip(true_np, pred_np):
        for t_char, p_char in zip(t_word, p_word):
            cm[t_char, p_char] += 1
    
    sorted_labels = [labels[i] for i in range(num_classes)]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sorted_labels, yticklabels=sorted_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
