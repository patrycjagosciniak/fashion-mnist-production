import pandas as pd
import torch


def train_model(
        model,
        train_loader,
        val_loader,
        criterion, optimizer, device, epochs=15):
    """Train a PyTorch model and evaluate it on validation data after each epoch.

    The function runs the full training loop for a given number of epochs.
    During each epoch, it updates model weights using the training data and then
    calculates validation loss and validation accuracy without updating weights.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model that should be trained.
    train_loader : torch.utils.data.DataLoader
        DataLoader containing training batches.
    val_loader : torch.utils.data.DataLoader
        DataLoader containing validation batches.
    criterion : torch.nn.Module
        Loss function used to calculate training and validation loss.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model weights.
    device : torch.device
        Device used for training, for example CPU or GPU.
    epochs : int, default=15
        Number of epochs used for training.

    Returns
    -------
    tuple
        Trained model and a pandas DataFrame with training history.
        The history contains train loss, train accuracy, validation loss
        and validation accuracy for each epoch.
    """
    history = []

    for epoch in range(epochs):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_acc:.4f}"
        )

    return model, pd.DataFrame(history)