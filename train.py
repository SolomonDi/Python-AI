import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, roc_curve

from data import get_dataLoader, IMG_SIZE
from model import PneumoniaModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 40
PATIENCE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MODEL_PATH = "pneumonia_model.pth"


def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_probs, all_labels = [], []

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True).float().unsqueeze(1)

            with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * images.size(0)
            all_probs.extend(torch.sigmoid(outputs.float()).squeeze(1).tolist())
            all_labels.extend(labels.squeeze(1).tolist())

    return total_loss / len(loader.dataset), np.array(all_probs), np.array(all_labels)


def pick_threshold(probs, labels):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    best = np.argmax(tpr - fpr)
    return float(np.clip(thresholds[best], 0.0, 1.0))


def main():
    torch.manual_seed(42)
    train_loader, val_loader, _, classes = get_dataLoader()

    model = PneumoniaModel().to(DEVICE)

    targets = np.array(train_loader.dataset.dataset.targets)[train_loader.dataset.indices]
    n_pos, n_neg = (targets == 1).sum(), (targets == 0).sum()
    pos_weight = torch.tensor([n_neg / n_pos], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, epochs=EPOCHS, steps_per_epoch=1, pct_start=0.15
    )
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_probs, val_labels = run_epoch(model, val_loader, criterion)
        scheduler.step()

        val_accuracy = 100 * ((val_probs > 0.5) == val_labels).mean()
        val_auc = roc_auc_score(val_labels, val_probs)

        print(f"EPOCH {epoch+1}/{EPOCHS} | Train loss: {train_loss:.4f} | "
              f"Val loss: {val_loss:.4f} | Val accuracy: {val_accuracy:.2f}% | Val AUC: {val_auc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print("  -> new best result, model saved")
        else:
            epochs_without_improvement += 1
            print(f"  -> no improvement ({epochs_without_improvement}/{PATIENCE})")
            if epochs_without_improvement >= PATIENCE:
                print("Early stopping: val loss stopped improving.")
                break

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    _, val_probs, val_labels = run_epoch(model, val_loader, criterion)
    threshold = pick_threshold(val_probs, val_labels)

    torch.save({
        "state_dict": model.state_dict(),
        "classes": classes,
        "img_size": IMG_SIZE,
        "threshold": threshold,
    }, MODEL_PATH)

    print(f"Done. Best val loss: {best_val_loss:.4f} | Threshold (from val): {threshold:.4f}")


if __name__ == "__main__":
    main()
