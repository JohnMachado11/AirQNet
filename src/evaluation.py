"""
AirQNet - Evaluation Script

It includes functions to compute accuracy, make predictions on individual samples, and perform batch 
predictions on test datasets while saving the results to a CSV file.
"""

from neural_network import FFN
from data_setup.dataset import test_loader
from config import DEBUG

import torch
import pandas as pd
import joblib


scaler = joblib.load("models/scaler/scaler.pkl")


def compute_accuracy(model, dataloader):
    """
    Calculate the accuracy of the trained model.
    """

    model = model.eval()
    correct: float = 0.0
    total_examples: int = 0

    for idx, (features, labels) in enumerate(dataloader):

        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)

    (correct / total_examples).item()
    
    print("\n", "ACCURACY:", (correct / total_examples).item())


def predict_class(model):
    """
    Predict the class on a single sample.
    """

    input_data: list[list[float]] = [[27.1, 76.3, 3.9, 18.0, 28.7, 9.9, 2.07, 5.6, 558.0]] # expected label is 2

    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        input_tensor = torch.tensor(scaler.transform(input_data), dtype=torch.float32)
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)  # Convert logits to probabilities
        predicted_class = torch.argmax(probabilities, dim=1)  # Get the class with the highest probability
    
    print(f"Predicted Class: {predicted_class.item()}")
    print(f"Class Probabilities: {probabilities.squeeze().tolist()}")


def predict_classes_and_save(model, features_csv, labels_csv, output_csv, include_label_str: bool = False):
    """
    Predict on a dataset and save results (features + Actual Label + Predicted Class + Probabilities)
    to a CSV file. Optionally append a string label representation in the same columns
    as the numeric labels, e.g., '2 (Moderate)'.
    
    Args:
        model: Trained PyTorch model.
        features_csv (str): Path to the CSV of test features.
        labels_csv (str): Path to the CSV of test labels (integer-encoded).
        output_csv (str): Path to write the CSV of predictions.
        scaler: The fitted scaler used for preprocessing (from training).
        include_label_str (bool): Whether to append e.g. ' (Moderate)' to the numeric label.
    """

    # Map integer classes to string labels
    CLASS_MAPPING = {
        0: "Hazardous",
        1: "Poor",
        2: "Moderate",
        3: "Good",
    }

    # Load test features & labels
    test_features = pd.read_csv(features_csv)
    test_labels_df = pd.read_csv(labels_csv)

    # Make sure we get a Series of labels (not a DataFrame)
    # If your labels CSV has a single column named "Air Quality":
    test_labels = test_labels_df["Air Quality"]
    # or if it only has one unnamed column you can do:
    # test_labels = test_labels_df.iloc[:, 0]  # first/only column

    # Normalize the test features using existing scaler
    test_features_normalized = scaler.transform(test_features)

    # Directly convert to tensor
    test_features_tensor = torch.tensor(test_features_normalized, dtype=torch.float32)

    # Predict with the model
    model.eval()
    with torch.no_grad():
        logits = model(test_features_tensor) # normalized test features
        probabilities = torch.softmax(logits, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)  # integer classes

    all_predictions = predicted_classes.tolist()
    all_probabilities = probabilities.tolist()

    # Calculate accuracy
    correct_predictions = sum(1 for pred, true in zip(all_predictions, test_labels) if pred == true)
    total_predictions = len(test_labels)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    # Create a DataFrame with all results
    results_df = test_features.copy()
    
    # Convert integer labels to string: "2 (Moderate)"
    if include_label_str:
        actual_labels_str = [
            f"{true} ({CLASS_MAPPING[true]})" for true in test_labels
        ]
        predicted_classes_str = [
            f"{pred} ({CLASS_MAPPING[pred]})" for pred in all_predictions
        ]
        # Add these columns to results
        results_df["Actual Label"] = actual_labels_str
        results_df["Predicted Class"] = predicted_classes_str
    else:
        # Just store the integer classes
        results_df["Actual Label"] = test_labels
        results_df["Predicted Class"] = all_predictions

    # Also store the raw probabilities
    results_df["Class Probabilities"] = [
        list(map(float, probs)) for probs in all_probabilities
    ]

    # Add accuracy summary row
    summary_row = {
        "Actual Label": f"Accuracy: {accuracy:.4f}",
        "Predicted Class": f"{correct_predictions}/{total_predictions}"
    }
    # Add blank entries for any extra columns
    for col in results_df.columns:
        if col not in summary_row:
            summary_row[col] = None

    summary_df = pd.DataFrame([summary_row], columns=results_df.columns)
    results_df = pd.concat([results_df, summary_df], ignore_index=True)

    # Save to CSV
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    print(f"Accuracy: {accuracy:.4f}  ({correct_predictions}/{total_predictions})")


if __name__ == "__main__":
    print("Starting Evaluations")

    NUM_INPUTS: int = 9
    NUM_OUTPUTS: int = 4

    model = FFN(num_inputs=NUM_INPUTS, num_outputs=NUM_OUTPUTS)

    # Load the full model
    weights_path = "models/weights/AirQNet_10K.pth"
    model.load_state_dict(torch.load(weights_path))
    model.eval()  # Set to evaluation mode

    if DEBUG:
        print(model)

    print("Full model loaded successfully!")

    compute_accuracy(model, test_loader)
    predict_class(model)

    # File paths
    features_csv = "data_setup/data/processed/test_features.csv"
    labels_csv = "data_setup/data/processed/test_labels.csv"
    output_csv = "prediction_results.csv"

    predict_classes_and_save(model, features_csv, labels_csv, output_csv, include_label_str=True)