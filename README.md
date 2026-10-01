# MLOps Deployment Strategies: Kidney Disease Prediction

This repository demonstrates various deployment and prediction strategies for a Machine Learning model using a Chronic Kidney Disease dataset. It covers the fundamental concepts of training a robust model pipeline and then executing predictions using three distinct strategies: **Batch**, **Online**, and **Streaming**.

## Project Structure

```text
.
├── train_model.py                 # Script to train and save the model pipeline
├── kidney_disease.csv             # The dataset used for training and prediction
├── kidney_disease_model.pkl       # The serialized model (generated after training)
├── Batch/
│   └── batch_prediction.py        # Demonstrates Batch Prediction strategy
├── Online/
│   └── online_prediction.py       # Demonstrates Online Prediction strategy
└── Streaming/
    └── streaming_prediction.py    # Demonstrates Streaming Prediction strategy
```

## Deployment Strategies Explained

1. **Batch Prediction (`Batch/batch_prediction.py`)**
   - **Concept:** Processing a large volume of data at once (e.g., overnight processing, weekly reports).
   - **Implementation:** Loads the entire dataset, drops the target variables, runs predictions for all rows simultaneously, and exports the results to `batch_predictions.csv`.

2. **Online Prediction (`Online/online_prediction.py`)**
   - **Concept:** On-demand predictions for a single instance in real-time (e.g., a web application where a user submits a form and expects an immediate result).
   - **Implementation:** Defines a single data sample as a dictionary, converts it into a DataFrame, and immediately returns the prediction.

3. **Streaming Prediction (`Streaming/streaming_prediction.py`)**
   - **Concept:** Processing data continuously as it arrives in real-time or near real-time (e.g., IoT sensor data, real-time fraud detection).
   - **Implementation:** Simulates a data stream by iterating over records one by one with a time delay (`time.sleep`), making and printing predictions sequentially.

## Getting Started

### Prerequisites

Ensure you have Python installed along with the required libraries. You can install them using:

```bash
pip install pandas scikit-learn joblib
```

### 1. Train the Model

Before running any predictions, you must train the model. The training script builds a complete `scikit-learn` Pipeline that includes data imputation, scaling, one-hot encoding, and a Random Forest Classifier.

```bash
python train_model.py
```
*This will output evaluation metrics and create the `kidney_disease_model.pkl` file.*

### 2. Run Predictions

Navigate to the respective directories to test each deployment strategy:

**Batch Prediction:**
```bash
cd Batch
python batch_prediction.py
```

**Online Prediction:**
```bash
cd Online
python online_prediction.py
```

**Streaming Prediction:**
```bash
cd Streaming
python streaming_prediction.py
```
