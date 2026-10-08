import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from data import get_dataLoader
from inference import DEVICE, Predictor


def main():
    _, _, test_loader, classes = get_dataLoader()
    predictor = Predictor()
    threshold = predictor.threshold

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = predictor.model(images.to(DEVICE))
            all_probs.extend(torch.sigmoid(outputs).squeeze(1).cpu().tolist())
            all_labels.extend(labels.tolist())

    all_probs, all_labels = np.array(all_probs), np.array(all_labels)
    all_preds = (all_probs > threshold).astype(int)

    print(f"Threshold (chosen on validation): {threshold:.4f}")
    print(f"ROC AUC: {roc_auc_score(all_labels, all_probs):.4f}\n")
    print(classification_report(all_labels, all_preds, target_names=classes))
    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
