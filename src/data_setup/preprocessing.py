from config import SAVE_DATA, SAVE_SCALAR, SCALER_DIR

from sklearn.preprocessing import StandardScaler
from typing import Tuple
import pandas as pd
import joblib
import os


def dataframe_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the DataFrame for modeling.

    This function performs the following operations:
    1. Maps the 'Air Quality' column's string labels to integer classes.
    2. Drops the original 'Air Quality' column and renames the newly encoded column to 'Air Quality'.
    3. Converts 'Population_Density' column to float datatype.
    """
    
    # Convert Air Quality (y) classes to int
    mapping = {
        "Hazardous": 0, 
        "Poor": 1,
        "Moderate": 2,
        "Good": 3
        }

    df['Air Quality Encoded'] = df['Air Quality'].map(mapping)
    df.drop("Air Quality", axis=1, inplace=True)

    df.rename(columns={"Air Quality Encoded": "Air Quality"}, inplace=True)

    # Convert Population_Density to float
    df['Population_Density'] = df['Population_Density'].astype(float)

    return df


def normalize_data(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Normalize the data using StandardScaler.

    This function fits the scaler on the training set and
    applies the same scaling to the validation and test sets.
    It returns the scaled features and saves the fitted scaler.
    """
    
    # Initialize the scaler
    scaler = StandardScaler()

    # Fit on training data and transform
    X_train_normalized = scaler.fit_transform(X_train)
    X_val_normalized = scaler.transform(X_val)  # Use the same scaler for validation
    X_test_normalized = scaler.transform(X_test)  # Use the same scaler for testing

    # Convert back to DataFrame with original column names
    X_train_normalized = pd.DataFrame(X_train_normalized, columns=X_train.columns)
    X_val_normalized = pd.DataFrame(X_val_normalized, columns=X_val.columns)
    X_test_normalized = pd.DataFrame(X_test_normalized, columns=X_test.columns)

    if SAVE_SCALAR:

        # Save the fitted scaler for later use
        scaler_path = os.path.join(SCALER_DIR, "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        
        print(f"Scaler saved to: {scaler_path}")

    return X_train_normalized, X_val_normalized, X_test_normalized


# Training = 70% | Validation = 20% | Test = 10%
def random_split(df: pd.DataFrame, train_frac: float, val_frac: float) -> Tuple[
    Tuple[pd.DataFrame, pd.Series], 
    Tuple[pd.DataFrame, pd.Series],
    Tuple[pd.DataFrame, pd.Series]]:
    """
    Randomly split the DataFrame into training, validation, and testing sets.

    The function shuffles the DataFrame first, then splits it into
    training, validation, and test sets based on the provided fractions.
    It separates the features (X) and the target column 'Air Quality' (y)
    for each set.
    """

    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate the split indices
    train_end = int(len(df) * train_frac)
    val_end = train_end + int(len(df) * val_frac)

    # Split the DataFrame
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    # Separate features and labels
    X_train = train_df.drop(columns=["Air Quality"])
    y_train = train_df["Air Quality"]

    X_val = val_df.drop(columns=["Air Quality"])
    y_val = val_df["Air Quality"]

    X_test = test_df.drop(columns=["Air Quality"])
    y_test = test_df["Air Quality"]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


df = pd.read_csv("data_setup/data/raw/updated_pollution_dataset.csv")
cleaned_df = dataframe_cleanup(df)

# Perform the 70-20-10 split
(X_train, y_train), (X_val, y_val), (X_test, y_test) = random_split(cleaned_df, 0.7, 0.2)


if SAVE_DATA:
    # Save training features and labels
    X_train.to_csv("data_setup/data/processed/train_features.csv", index=False)
    y_train.to_csv("data_setup/data/processed/train_labels.csv", index=False)

    # Save validation features and labels
    X_val.to_csv("data_setup/data/processed/val_features.csv", index=False)
    y_val.to_csv("data_setup/data/processed/val_labels.csv", index=False)

    # Save testing features and labels
    X_test.to_csv("data_setup/data/processed/test_features.csv", index=False)
    y_test.to_csv("data_setup/data/processed/test_labels.csv", index=False)

    print("Datasets saved as CSV files:")
    print("train_features.csv, train_labels.csv")
    print("val_features.csv, val_labels.csv")
    print("test_features.csv, test_labels.csv")


print("Before:")
print(X_train.head())

X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)

print("After:")
print(X_train.head())

if SAVE_DATA:
    # Save training data
    X_train.to_csv("data_setup/data/processed/train_features_normalized.csv", index=False)
    # Save validation data
    X_val.to_csv("data_setup/data/processed/val_features_normalized.csv", index=False)
    # Save testing data
    X_test.to_csv("data_setup/data/processed/test_features_normalized.csv", index=False)



# ---------------- Useful Pandas Methods ----------------
# print(df.info())
# print(df.head())
# print(df.tail())
# print(df.shape)
# print(df.describe())

# Null check
# print(df.isnull().sum())

# Uniques
# print(df['Air Quality'].unique())

# for col in df.columns:
#     print(col, df[col].dtypes)
