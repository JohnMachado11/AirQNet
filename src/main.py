"""
Training script for AirQNet - Air Quality Classification Neural Network.
"""

from data_setup.dataset import train_loader, val_loader
from config import SAVE_MODEL, WEIGHTS_DIR, DEBUG
from neural_network import FFN

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
import os


# Fix the random seed for reproducibility
torch.manual_seed(123)

# Hyperparameters
NUM_INPUTS: int = 9
NUM_OUTPUTS: int = 4
NUM_EPOCHS: int = 500
# NUM_EPOCHS: int = 1
LEARNING_RATE: float = 0.0001
WEIGHT_DECAY: float = 0.001

# Instantiate model
model = FFN(num_inputs=NUM_INPUTS, num_outputs=NUM_OUTPUTS)

# AdamW Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Initialize lists to store metrics
train_losses: list = []
val_losses: list = []
train_accuracies: list = []
val_accuracies: list = []

# Early stopping parameters
patience: int = 10  # number of epochs to wait without improvement
best_val_loss: float = float('inf')
patience_counter: int = 0

# Storing the best model weights
best_model_weights = copy.deepcopy(model.state_dict())

# Initialize live plot
plt.ion()  # Interactive mode on
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))


for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_train_loss: float = 0.0
    correct_train: int = 0
    total_train: int = 0

    for batch_idx, (features, labels) in enumerate(train_loader):
        optimizer.zero_grad() # Reset loss gradients from previous iteration
        logits = model(features) # Forward pass (predict)
        loss = F.cross_entropy(logits, labels) # Compute loss

        loss.backward() # Backpropagation: compute gradients
        optimizer.step() # Update model parameters

        if DEBUG:
            print("\n", "-" * 50)
            print("Loss:", loss)
            print("Loss.item():", loss.item())
        
        epoch_train_loss += loss.item()

        # Calculate training accuracy
        predictions = torch.argmax(logits, dim=1)

        if DEBUG:
            print("Prediction:", predictions)
            print("Labels:", labels)
            print("Len predictions:", len(predictions))
            print("Len Labels:", len(labels))
            print("=" * 50)

        correct_train += (predictions == labels).sum().item()
        total_train += labels.size(0)

    # Calculate average training loss and accuracy
    if DEBUG:
        print("\nEpoch train loss:", epoch_train_loss)
        print("Train Loader Length:", len(train_loader))

    avg_train_loss = epoch_train_loss / len(train_loader)

    if DEBUG:
        print("Avg train loss:", avg_train_loss)

    train_accuracy = correct_train / total_train

    if DEBUG:
        print("Correct train:", correct_train)
        print("Total train:", total_train)
        print("Train Accuracy:", train_accuracy)

    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)

    # Validation loop
    model.eval()
    val_loss: float = 0.0
    correct_val: int = 0
    total_val: int = 0

    with torch.no_grad():
        for features, labels in val_loader:
            logits = model(features)
            loss = F.cross_entropy(logits, labels)
            val_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)
            correct_val += (predictions == labels).sum().item()
            total_val += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct_val / total_val
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS}, "
        f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
        f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}"
    )

    # ------------------------------
    # Early Stopping Check
    # ------------------------------
    if avg_val_loss < best_val_loss:
        # Found new best validation loss
        best_val_loss = avg_val_loss
        patience_counter: int = 0
        best_model_weights = copy.deepcopy(model.state_dict())
    else:
        # No improvement
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

    # Live plot update
    ax1.clear()
    ax2.clear()

    # Loss plot
    ax1.plot(train_losses, label="Training Loss", color="orange")
    ax1.plot(val_losses, label="Validation Loss", color="blue")
    ax1.set_title("Training and Validation Loss Over Epochs")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Accuracy plot
    ax2.plot(train_accuracies, label="Training Accuracy", color="orange")
    ax2.plot(val_accuracies, label="Validation Accuracy", color="blue")
    ax2.set_title("Training and Validation Accuracy Over Epochs")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.pause(0.1)

plt.ioff()  # Turn off interactive mode
plt.show()  # Show the final plot


# After training ends (or early stops), restore best weights
model.load_state_dict(best_model_weights)

if SAVE_MODEL:

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_count: int = count_parameters(model)
    model_name: str = f"AirQNet_{param_count // 1000}K" 

    # Path for saving
    weights_path: str = os.path.join(WEIGHTS_DIR, f"{model_name}.pth")

    # Save the model weights
    torch.save(model.state_dict(), weights_path)
    print(f"Model weights saved at: {weights_path}")
