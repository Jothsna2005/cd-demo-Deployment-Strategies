import joblib
import pandas as pd


# Load trained model
model = joblib.load("../kidney_disease_model.pkl")


# One sample for online prediction
data = {
    "age": 48,
    "bp": 80,
    "sg": 1.02,
    "al": 1,
    "su": 0,
    "rbc": "normal",
    "pc": "normal",
    "pcc": "notpresent",
    "ba": "notpresent",
    "bgr": 121,
    "bu": 36,
    "sc": 1.2,
    "sod": 138,
    "pot": 4.5,
    "hemo": 15.4,
    "pcv": "44",
    "wc": "7800",
    "rc": "5.2",
    "htn": "no",
    "dm": "no",
    "cad": "no",
    "appet": "good",
    "pe": "no",
    "ane": "no"
}


# Convert input into DataFrame
input_df = pd.DataFrame([data])


# Make prediction
prediction = model.predict(input_df)


# Display result
print("Input received:")
print(input_df)

print("\nPrediction:")
print(prediction[0])