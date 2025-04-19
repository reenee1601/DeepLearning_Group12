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

The model was developed through careful hyperparameter optimization to ensure robust performance in real-world scenarios.
