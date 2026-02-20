
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
ITEM_ID = 27           # Example item_id from sample_head.csv
STORE_ID = 6269        # Example store_id from sample_head.csv
DATA_PATH = '../dataset/data_andre.feather' # Adjust path if needed
TARGET_COL = 'value'
DATE_COL = 'date'

N_FORECAST = 1 
N_LOOKBACK = 60
N_FUTURE = 365
EPOCHS = 50            # Reduced epochs for quicker testing
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# -----------------------------------------------------------------------------
# DATA LOADING & PREPROCESSING
# -----------------------------------------------------------------------------
print(f"Loading data from {DATA_PATH}...")

# Check if file exists
if not os.path.exists(DATA_PATH):
    # Fallback to sample_head.csv for demonstration if feather not found
    DATA_PATH = '../dataset/sample_head.csv'
    print(f"Feather file not found. Loading sample data from {DATA_PATH}...")
    df_full = pd.read_csv(DATA_PATH)
else:
    df_full = pd.read_feather(DATA_PATH)

print(f"Filtering for Item: {ITEM_ID}, Store: {STORE_ID}...")
# Filter for specific item and store
df = df_full[(df_full['item_id'] == ITEM_ID) & (df_full['store_id'] == STORE_ID)].copy()

if df.empty:
    # If filtered dataframe is empty, list some available options
    available_items = df_full['item_id'].unique()[:5]
    available_stores = df_full['store_id'].unique()[:5]
    print(f"Warning: No data found for Item {ITEM_ID} and Store {STORE_ID}.")
    print(f"Available items: {available_items}")
    print(f"Available stores: {available_stores}")
    
    # Just grab the first available combination to proceed with demonstration
    ITEM_ID = available_items[0]
    STORE_ID = df_full[df_full['item_id'] == ITEM_ID]['store_id'].unique()[0]
    print(f"Switching to Item: {ITEM_ID}, Store: {STORE_ID}...")
    df = df_full[(df_full['item_id'] == ITEM_ID) & (df_full['store_id'] == STORE_ID)].copy()

# Ensure date is datetime and sort
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values(DATE_COL)

# Prepare target variable
# Using 'value' as the target (sales/demand)
y = df[TARGET_COL].fillna(method='ffill').values.reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(y)
y_scaled = scaler.transform(y)

# Generate the training sequences
X = []
Y = []

for i in range(N_LOOKBACK, len(y_scaled) - N_FORECAST + 1):
    X.append(y_scaled[i - N_LOOKBACK: i])
    Y.append(y_scaled[i: i + N_FORECAST])

X = np.array(X)
Y = np.array(Y)

print(f"Training data shape: X={X.shape}, Y={Y.shape}")

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X).float().to(DEVICE)
Y_tensor = torch.from_numpy(Y).float().to(DEVICE)

# Create DataLoader
dataset = TensorDataset(X_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------------------------------------------------------
# MODEL DEFINITION (PyTorch)
# -----------------------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50):
        super(LSTMModel, self).__init__()
        # First LSTM layer: returns sequences equivalent to return_sequences=True
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        # Second LSTM layer: equivalent to return_sequences=False (we take last output)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, _ = self.lstm1(x)
        # out shape: (batch_size, seq_len, hidden_size)
        
        out, (h_n, c_n) = self.lstm2(out)
        # We only want the last time step output for the second LSTM layer
        # out shape: (batch_size, seq_len, hidden_size)
        
        # Take the output of the last time step
        out = out[:, -1, :] 
        
        out = self.fc1(out)
        out = self.fc2(out)
        return out

torch.manual_seed(0)
model = LSTMModel().to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -----------------------------------------------------------------------------
# TRAINING LOOP
# -----------------------------------------------------------------------------
print("Starting training...")
# Only train if we have enough data
if len(X) > 0:
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch_X, batch_Y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Reshape Y if necessary to match output shape
            loss = criterion(outputs, batch_Y.view(-1, 1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss/len(dataloader):.4f}')
else:
    print("Not enough data to train. Please check lookback period or data source.")
    exit()

# -----------------------------------------------------------------------------
# FORECASTING
# -----------------------------------------------------------------------------
print(f"Generating forecasts for {N_FUTURE} days...")

model.eval()
y_future = []

# Prepare initial input
# Make sure to keep dimensions consistent (1, 60, 1)
x_pred = X[-1:, :, :]  # Last observed input sequence (numpy)
x_pred_tensor = torch.from_numpy(x_pred).float().to(DEVICE) # Convert to tensor for model

y_pred_val = Y[-1][0] # Last observed target value (scalar)
y_last_tensor = torch.tensor([y_pred_val]).float().to(DEVICE).view(1, 1, 1)

current_input = x_pred_tensor 
last_val = y_last_tensor

with torch.no_grad():
    for i in range(N_FUTURE):
        # Shift window: remove first time step, append last known/predicted value
        # current_input shape: (1, 60, 1)
        # last_val shape: (1, 1, 1)
        
        # New input for prediction
        next_input = torch.cat((current_input[:, 1:, :], last_val), dim=1)
        
        # Predict
        output = model(next_input) # Output shape: (1, 1)
        
        val = output.item()
        y_future.append(val)
        
        # Update state - we reuse next_input for shifting in next iteration
        current_input = next_input
        # New last val is the prediction we just made
        last_val = output.view(1, 1, 1)
        # last_val shape: (1, 1, 1)
        
        # New input for prediction
        next_input = torch.cat((current_input[:, 1:, :], last_val), dim=1)
        
        # Predict
        output = model(next_input)
        # output shape (1, 1)
        
        val = output.item()
        y_future.append(val)
        
        # Update state
        current_input = next_input
        last_val = output.view(1, 1, 1)


# transform the forecasts back to the original scale
y_future = np.array(y_future).reshape(-1, 1)
y_future = scaler.inverse_transform(y_future)

# -----------------------------------------------------------------------------
# RESULTS
# -----------------------------------------------------------------------------
# Organize the results in a data frame
df_past = df[[DATE_COL, TARGET_COL]].copy()
df_past.rename(columns={DATE_COL: 'Date', TARGET_COL: 'Actual'}, inplace=True)
df_past['Forecast'] = np.nan

# Prepare forecast dataframe
last_date = df_past['Date'].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=N_FUTURE)

df_future = pd.DataFrame({
    'Date': future_dates,
    'Forecast': y_future.flatten(),
    'Actual': np.nan
})

results = pd.concat([df_past, df_future]).set_index('Date')

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(results.index, results['Actual'], label='Actual Sales')
plt.plot(results.index, results['Forecast'], label='Forecast', color='orange')
plt.title(f'Sales Forecast for Item {ITEM_ID} (Store {STORE_ID})')
plt.xlabel('Date')
plt.ylabel('Sales Value')
plt.legend()
plt.grid(True)
plt.savefig('../forecast_plot.png')
print("Plot saved as ../forecast_plot.png")





