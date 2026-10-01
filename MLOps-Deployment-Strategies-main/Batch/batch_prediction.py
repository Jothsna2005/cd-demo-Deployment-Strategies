import joblib
import pandas as pd


# Load trained model
model = joblib.load("../kidney_disease_model.pkl")


# Load dataset
df = pd.read_csv("../kidney_disease.csv")

print("Number of records:", len(df))


# Keep IDs for identification
ids = df["id"]


# Remove target and ID
X = df.drop(
    columns=["classification", "id"]
)


# Generate predictions
predictions = model.predict(X)


# Create output DataFrame
output = pd.DataFrame({
    "id": ids,
    "prediction": predictions
})


# Save predictions
output.to_csv(
    "batch_predictions.csv",
    index=False
)


print("\nBatch prediction completed.")

print("\nFirst 10 predictions:")
print(output.head(10))

print(
    "\nOutput file created: "
    "batch_predictions.csv"
)