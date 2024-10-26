import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Hyperparameters
sequence_length = 30  # Number of previous time steps to consider for prediction
hidden_size = 64
num_layers = 1
learning_rate = 0.001
epochs = 50
batch_size = 32

# 1. Load and Preprocess Data
class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, index):
        x = self.data[index:index + self.sequence_length]
        y = self.data[index + self.sequence_length]
        return x, y

def load_and_prepare_data(file_path, sequence_length):
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df = df.sort_values(by='timestamp')
    
    # Normalize the values
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df['value'].values.reshape(-1, 1))
    
    # Split into train and test sets (80% train, 20% test)
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, sequence_length)
    test_dataset = TimeSeriesDataset(test_data, sequence_length)
    
    return train_dataset, test_dataset, scaler

# 2. Define the GRU Model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, h = self.gru(x)
        out = self.fc(h[-1])
        return out

# 3. Train the Model
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            # Move data to GPU
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}')

# 4. Evaluate and Predict
def evaluate_model(model, test_loader, criterion, scaler, device):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            # Move data to GPU
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            output = model(x_batch)
            predictions.append(output.cpu().numpy())
            actuals.append(y_batch.cpu().numpy())
    
    # Inverse transform to original scale
    predictions = scaler.inverse_transform(np.concatenate(predictions))
    actuals = scaler.inverse_transform(np.concatenate(actuals))
    return predictions, actuals

# Main Execution
def main(file_path):
    # Load data and prepare datasets
    train_dataset, test_dataset, scaler = load_and_prepare_data(file_path, sequence_length)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model, Loss, Optimizer
    model = GRUModel(input_size=1, hidden_size=hidden_size, output_size=1, num_layers=num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, device)

    # Evaluate the model
    predictions, actuals = evaluate_model(model, test_loader, criterion, scaler, device)

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# Run the code
file_path = '/mnt/Data/raghav/BIGDATA/realAdExchange/realAdExchange/exchange-3_cpc_results.csv'  # Specify your CSV file path here
main(file_path)