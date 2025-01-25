import logging
from datetime import datetime
import pandas as pd
from predictive_model import train_predictive_model
from database.db_setup import get_engine
from sqlalchemy import create_engine

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='network_service.log',
                    filemode='a')

# Initialize the database engine
db_engine = get_engine()

class NetworkService:
    """
    A class to manage network monitoring services.
    This class uses a predictive model to analyze live network data, detect anomalies, log issues, and store performance logs in a database.
    """

    def __init__(self):
        logging.info("Initializing NetworkService...")

    def analyze_network_data(self, data):
        """
        Analyze live network data using the predictive model.

        Args:
            data (pd.DataFrame): The live network data to be analyzed.

        Returns:
            pd.DataFrame: Data with anomaly predictions.
        """
        try:
            logging.info("Starting network data analysis...")
            # Use the predictive model to analyze data
            analyzed_data = train_predictive_model(data)
            logging.info("Network data analysis completed.")
            return analyzed_data
        except Exception as e:
            logging.error(f"Error during network data analysis: {e}")
            raise

    def detect_anomalies(self, analyzed_data):
        """
        Detect anomalies in the analyzed network data.

        Args:
            analyzed_data (pd.DataFrame): The analyzed network data with anomaly predictions.

        Returns:
            pd.DataFrame: Data containing only the rows flagged as anomalies.
        """
        try:
            logging.info("Detecting anomalies in network data...")
            # Filter rows where anomalies are detected (e.g., 'Predicted_Failure' == -1)
            anomalies = analyzed_data[analyzed_data['Predicted_Failure'] == -1]
            logging.info(f"Detected {len(anomalies)} anomalies.")
            return anomalies
        except Exception as e:
            logging.error(f"Error during anomaly detection: {e}")
            raise

    def log_performance(self, analyzed_data):
        """
        Store network performance logs in the database.

        Args:
            analyzed_data (pd.DataFrame): The analyzed network data to be logged.
        """
        try:
            logging.info("Logging network performance data to the database...")
            # Write data to the database
            analyzed_data.to_sql('network_performance_logs', con=db_engine, if_exists='append', index=False)
            logging.info("Network performance data logged successfully.")
        except Exception as e:
            logging.error(f"Error during performance logging: {e}")
            raise

    def monitor_network(self, data):
        """
        Full workflow to monitor the network: analyze data, detect anomalies, and log performance.

        Args:
            data (pd.DataFrame): The live network data to be monitored.
        """
        try:
            logging.info("Starting network monitoring workflow...")
            # Step 1: Analyze network data
            analyzed_data = self.analyze_network_data(data)

            # Step 2: Detect anomalies
            anomalies = self.detect_anomalies(analyzed_data)

            # Step 3: Log performance data
            self.log_performance(analyzed_data)

            logging.info("Network monitoring workflow completed successfully.")

            # Optionally return the anomalies for further action
            return anomalies
        except Exception as e:
            logging.error(f"Error during network monitoring workflow: {e}")
            raise

if __name__ == "__main__":
    # Example usage of the NetworkService class

    # Simulated live network data
    simulated_data = pd.DataFrame({
        'Latency_ms': [10, 20, 300, 15, 50],
        'Packet_Loss': [0.1, 0.0, 5.0, 0.3, 0.2],
        'Bandwidth_Mbps': [100, 95, 10, 80, 70]
    })

    # Initialize the service
    service = NetworkService()

    # Run the network monitoring workflow
    try:
        anomalies = service.monitor_network(simulated_data)
        if not anomalies.empty:
            logging.info("Anomalies detected:")
            logging.info(anomalies)
    except Exception as e:
        logging.error(f"An error occurred in the main process: {e}")
