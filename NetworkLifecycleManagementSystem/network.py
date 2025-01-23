import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random

# Simulated Data Generation

def simulate_network_data(n=1000):
    """Simulate network performance data."""
    timestamps = pd.date_range(start=pd.Timestamp.now() - pd.Timedelta(days=30), periods=n, freq='H')
    latency = np.random.normal(loc=50, scale=10, size=n)
    packet_loss = np.clip(np.random.normal(loc=0.05, scale=0.02, size=n), 0, 1)
    bandwidth = np.random.uniform(100, 1000, size=n)
    failures = [1 if random.random() < 0.05 else 0 for _ in range(n)]  # Random 5% failure rate

    return pd.DataFrame({
        'Timestamp': timestamps,
        'Latency_ms': latency,
        'Packet_Loss': packet_loss,
        'Bandwidth_Mbps': bandwidth,
        'Failure': failures
    })

network_data = simulate_network_data()

# Predictive Maintenance

def predictive_maintenance_model(data):
    """Train a simple anomaly detection model to predict potential failures."""
    features = data[['Latency_ms', 'Packet_Loss', 'Bandwidth_Mbps']]
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(features)
    data['Anomaly_Score'] = model.decision_function(features)
    data['Predicted_Failure'] = model.predict(features)

    return data

network_data = predictive_maintenance_model(network_data)
print("Predictive Maintenance Sample:")
print(network_data.head())

# Resource Allocation for Procurement

def simulate_procurement_data():
    """Simulate procurement dataset for equipment pricing and quality."""
    data = pd.DataFrame({
        'Vendor': np.random.choice(['Vendor A', 'Vendor B', 'Vendor C'], size=100),
        'Price': np.random.uniform(100, 1000, size=100),
        'Quality_Score': np.random.uniform(0.5, 1.0, size=100)
    })
    return data

def optimize_procurement(data):
    """Use a simple linear regression to model price vs. quality for optimal selection."""
    model = LinearRegression()
    X = data[['Quality_Score']]
    y = data['Price']
    model.fit(X, y)
    data['Cost_Effectiveness'] = model.predict(X) / data['Quality_Score']

    # Select top recommendations
    recommended = data.sort_values(by='Cost_Effectiveness').head(5)
    return recommended

procurement_data = simulate_procurement_data()
recommended_procurement = optimize_procurement(procurement_data)
print("Recommended Procurement Options:")
print(recommended_procurement)

# Energy Management Simulation

def simulate_energy_data(n=100):
    """Simulate energy consumption and renewable production data."""
    solar_output = np.random.uniform(0, 100, size=n)  # Solar energy in kWh
    network_load = np.random.uniform(50, 150, size=n)
    battery_level = np.clip(np.cumsum(solar_output - network_load), 0, 1000)  # Battery level
    return pd.DataFrame({'Solar_Output_kWh': solar_output, 'Network_Load_kWh': network_load, 'Battery_Level': battery_level})

energy_data = simulate_energy_data()

# Bandwidth Management

def simulate_user_activity_data(n=100):
    """Simulate bandwidth usage by different activities."""
    activity_types = ['Educational', 'Entertainment', 'Background Tasks']
    data = pd.DataFrame({
        'Timestamp': pd.date_range(start=pd.Timestamp.now(), periods=n, freq='15T'),
        'Activity_Type': np.random.choice(activity_types, size=n, p=[0.5, 0.3, 0.2]),
        'Bandwidth_Usage_Mbps': np.random.uniform(1, 50, size=n)
    })
    return data

bandwidth_data = simulate_user_activity_data()

# Priority-based Bandwidth Allocation

def prioritize_bandwidth(data):
    """Allocate bandwidth intelligently based on activity type."""
    priority_map = {'Educational': 1, 'Entertainment': 2, 'Background Tasks': 3}
    data['Priority'] = data['Activity_Type'].map(priority_map)
    data = data.sort_values(by='Priority')
    return data

bandwidth_data = prioritize_bandwidth(bandwidth_data)
print("Bandwidth Allocation Prioritization:")
print(bandwidth_data.head())

# Visualizing Network and Energy Insights
plt.figure(figsize=(12, 6))
plt.plot(network_data['Timestamp'], network_data['Latency_ms'], label='Latency (ms)')
plt.plot(network_data['Timestamp'], network_data['Bandwidth_Mbps'], label='Bandwidth (Mbps)')
plt.legend()
plt.title('Network Performance Over Time')
plt.xlabel('Time')
plt.ylabel('Performance Metrics')
plt.show()
