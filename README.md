# Urban Traffic Prediction System

## Project Overview
This repository contains an AI-driven solution for urban traffic prediction across the United States, addressing the growing challenge of congestion that costs U.S. drivers $81 billion in wasted time and fuel in 2022 alone. With U.S. Census Bureau projections indicating 89% of Americans will live in urban areas by 2050, our research focused on developing accurate, real-time traffic forecasting capabilities using deep learning techniques.

## Dataset
The training data used in this project can be found on Kaggle:
[Traffic Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset)

This dataset contains comprehensive traffic data collected from multiple urban junctions between 2015 and 2017.

## Repository Structure
- **Dataset_Visualisations.ipynb**: Initial visualizations of the dataset for data understanding
- **Submission.ipynb**: Primary model training and evaluation file
- **best_traffic_cnn_lstm_model.pth**: The final trained model with best performance
- **iterations/**: Folder containing multiple iterations of models that were attempted during the development process
- **Deep Learning Report**: Final Report

## Methodology
Our approach involved:
1. Temporal feature extraction
2. Sequence-based input preparation
3. Time-based validation strategies

## Models Evaluated
We conducted a systematic evaluation of seven deep learning architectures:
- Recurrent networks (GRU, LSTM)
- Attention-based models (Transformer)
- Convolutional approaches (CNN, TCN)
- Hybrid architectures (LSTM+Transformer, CNN+LSTM)

## Key Findings
- Traditional recurrent architectures performed poorly (R² < 0.25)
- Approaches incorporating convolutional components demonstrated superior capabilities for capturing the complex spatial-temporal dynamics of urban traffic
- The CNN+LSTM hybrid model emerged as the top performer with an exceptional R² score of 0.9667
- Hybrid approaches significantly outperformed standalone architectures by simultaneously modeling both junction-to-junction relationships and temporal traffic patterns

## Best Model
The final CNN+LSTM hybrid model provides accurate hour-ahead traffic predictions, enabling:
- Proactive traffic management
- Reduced environmental impact
- Enhanced urban mobility

## Usage

```python
import torch
model = torch.load('best_traffic_CNN_LStm_model.pth')
```

## Using the Trained Model

To use the pre-trained CNN+LSTM model for traffic prediction:

### 1\. Install Dependencies

```python
pip install torch pandas numpy matplotlib scikit-learn
```

### 2\. Load the Model

```python
import torch
import torch.nn as nn

# Define model architecture (must match the architecture used for training)
class TrafficCNNLSTM(nn.Module):
    def __init__(self, seq_length=24, lstm_hidden_size=64, lstm_layers=2, cnn_filters=(64, 128, 128), 
                dropout_rate=0.2, kernel_size=3):
        super(TrafficCNNLSTM, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=cnn_filters[0], kernel_size=kernel_size, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=cnn_filters[0], out_channels=cnn_filters[1], kernel_size=kernel_size, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.conv3 = nn.Conv1d(in_channels=cnn_filters[1], out_channels=cnn_filters[2], kernel_size=kernel_size, padding=1)
        self.relu3 = nn.ReLU()
        
        self.cnn_output_size = cnn_filters[2]
        self.reduced_seq_length = seq_length // 4
        
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(lstm_hidden_size, 32)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        if x.dim() == 3 and x.size(2) == 1:
            x = x.permute(0, 2, 1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        
        x = self.dropout(lstm_out)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Create model instance with the same parameters used during training
model = TrafficCNNLSTM(
    seq_length=24,
    lstm_hidden_size=64,  # Use values from your best model
    lstm_layers=2,
    cnn_filters=(64, 128, 128),
    dropout_rate=0.2,
    kernel_size=3
)

# Load the trained weights
model.load_state_dict(torch.load('best_traffic_CNN_LStm_model.pth'))
model.eval()  # Set to evaluation mode
```

### 3\. Prepare Data for Prediction


```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Function to create sequences from data
def create_sequences(data, seq_length=24):
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
    return np.array(X)

# Prepare input data (example)
# Normalize input data with the same scaler used during training
scaler = MinMaxScaler()
# Assuming 'new_data' is your input data with the same format as training data
scaled_data = scaler.fit_transform(new_data[['Vehicles']].values)
X_pred = create_sequences(scaled_data, seq_length=24)

# Convert to tensor
X_pred_tensor = torch.tensor(X_pred.reshape(X_pred.shape[0], X_pred.shape[1], 1), dtype=torch.float32)
```

### 4\. Make Predictions


```python
# Generate predictions
with torch.no_grad():
    predictions = model(X_pred_tensor)
    
# Convert predictions back to original scale
predictions_np = predictions.numpy()
predictions_original = scaler.inverse_transform(predictions_np)

print(f"Predicted traffic volumes: {predictions_original}")
```

