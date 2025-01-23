import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to preprocess input data
def preprocess_data(data):
    """
    Preprocess the input data by handling missing values and normalizing.

    Args:
        data (pd.DataFrame): The input data.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    try:
        logging.info("Starting data preprocessing.")

        # Handle missing values by filling with the mean
        data = data.fillna(data.mean())
        logging.info("Missing values handled.")

        # Normalize the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        logging.info("Data normalized.")

        return pd.DataFrame(data_scaled, columns=data.columns)
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise

# Function to train the IsolationForest model
def train_isolation_forest(data, contamination=0.1):
    """
    Train an IsolationForest model.

    Args:
        data (pd.DataFrame): Preprocessed data.
        contamination (float): Proportion of anomalies in the data.

    Returns:
        IsolationForest: Trained model.
    """
    try:
        logging.info("Starting model training.")

        # Initialize and train the IsolationForest model
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(data)
        logging.info("Model training completed.")

        return model
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

# Function to evaluate the model
def evaluate_model(model, data):
    """
    Evaluate the IsolationForest model.

    Args:
        model (IsolationForest): Trained model.
        data (pd.DataFrame): Data to evaluate on.

    Returns:
        dict: Evaluation metrics.
    """
    try:
        logging.info("Starting model evaluation.")

        # Predict anomalies (-1 for anomalies, 1 for normal)
        predictions = model.predict(data)

        # Map predictions to binary values
        binary_predictions = np.where(predictions == -1, 1, 0)  # 1 for anomaly, 0 for normal

        # Placeholder for actual labels (for demonstration purposes)
        actual_labels = np.zeros(len(data))

        # Generate a classification report
        report = classification_report(actual_labels, binary_predictions, output_dict=True)
        logging.info("Model evaluation completed.")

        return report
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise

# Example usage of the trained model to predict anomalies
def predict_anomalies(model, new_data):
    """
    Use the trained model to predict anomalies in new data.

    Args:
        model (IsolationForest): Trained model.
        new_data (pd.DataFrame): New data to predict on.

    Returns:
        np.ndarray: Predictions (-1 for anomalies, 1 for normal).
    """
    try:
        logging.info("Predicting anomalies.")
        return model.predict(new_data)
    except Exception as e:
        logging.error(f"Error during anomaly prediction: {e}")
        raise

# Main execution block (example)
if __name__ == "__main__":
    try:
        # Example dataset
        data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })

        logging.info("Dataset created.")

        # Preprocess the data
        preprocessed_data = preprocess_data(data)

        # Split the data for demonstration (though not used for IsolationForest)
        train_data, test_data = train_test_split(preprocessed_data, test_size=0.2, random_state=42)

        # Train the IsolationForest model
        model = train_isolation_forest(train_data)

        # Evaluate the model
        evaluation_report = evaluate_model(model, test_data)
        logging.info(f"Evaluation Report: {evaluation_report}")

        # Predict anomalies on new data
        new_data = pd.DataFrame({
            'feature1': np.random.randn(5),
            'feature2': np.random.randn(5),
            'feature3': np.random.randn(5)
        })
        new_data_preprocessed = preprocess_data(new_data)
        predictions = predict_anomalies(model, new_data_preprocessed)

        logging.info(f"Predictions: {predictions}")
    except Exception as e:
        logging.critical(f"Critical error in main execution: {e}")
