import joblib
import pandas as pd
import time


# Load trained model
model = joblib.load("../kidney_disease_model.pkl")


# Load dataset
df = pd.read_csv("../kidney_disease.csv")


# Remove target and ID
X = df.drop(
    columns=["classification", "id"]
)


print("Starting streaming prediction...\n")


# Simulate incoming records
for i in range(10):

    # Get one record
    record = X.iloc[[i]]

    # Generate prediction
    prediction = model.predict(record)

    # Display result
    print(
        f"Incoming Record {i + 1} "
        f"Prediction: {prediction[0]}"
    )

    # Wait 1 second
    time.sleep(1)


print("\nStreaming simulation completed.")