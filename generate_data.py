import pandas as pd
import numpy as np
import os

# Set seed for reproducibility
np.random.seed(42)
n_rows = 500

# Generate Features
duration = np.random.exponential(scale=2.0, size=n_rows) # connection duration in seconds
packet_size = np.random.randint(40, 65535, size=n_rows) # payload in bytes
protocol = np.random.choice(['TCP', 'UDP', 'ICMP'], size=n_rows, p=[0.7, 0.2, 0.1])
port_number = np.random.choice([80, 443, 22, 21, 53, 8080], size=n_rows)

# Simulate source IP pattern frequency (e.g., high count = suspicious behavior)
source_ip_patterns = np.random.poisson(lam=10, size=n_rows)

# Create Base DataFrame
df = pd.DataFrame({
    'duration': duration,
    'protocol': protocol,
    'packet_size': packet_size,
    'source_ip_patterns': source_ip_patterns,
    'port_number': port_number
})

# Inject structured logic to assign realistic Target Labels (intrusion_type)
def assign_label(row):
    # DoS attack pattern: Huge influx of large packets over rapid bursts
    if row['packet_size'] > 55000 and row['source_ip_patterns'] > 14:
        return 'DoS'
    # Port Scan pattern: Interacting with management ports frequently
    elif row['port_number'] in [22, 21, 8080] and row['source_ip_patterns'] > 12:
        return 'PortScan'
    # General Malware/Botnet pattern
    elif row['duration'] > 8.0 and row['packet_size'] < 500:
        return 'Malware'
    else:
        return 'Normal'

df['intrusion_type'] = df.apply(assign_label, axis=1)

# Ensure data directory exists and save
os.makedirs('data', exist_ok=True)
df.to_csv('data/network_traffic.csv', index=False)
print("✅ Dataset with 500 rows successfully generated at data/network_traffic.csv")