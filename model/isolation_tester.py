import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Step 1: Load the trained model and scaler
isolation_forest = joblib.load("isolation_forest_model.joblib")
scaler = joblib.load("scaler.joblib")
label_encoder = joblib.load("label_encoder.joblib")

# Step 2: Load the labeled test dataset
test_data_df = pd.read_csv("../dataset/test_dataset.csv")

# Step 3: Preprocess the dataset
test_data_df['DateTime'] = pd.to_datetime(test_data_df['DateTime'])
test_data_df['day_of_week'] = test_data_df['DateTime'].dt.dayofweek
test_data_df['hour_of_day'] = test_data_df['DateTime'].dt.hour

# Encode 'transaction_category' using the same LabelEncoder as during training
test_data_df['transaction_category'] = label_encoder.transform(test_data_df['TransactionCategory'])

# Calculate the derived feature 'Speed'
test_data_df['Speed'] = test_data_df['DistanceFromLastTransaction'] / (test_data_df['TimeFromLastTransaction'] + 1)

# Select the same features used for training (including Speed)
features = [
    'Amount',
    'Speed',
    'day_of_week',
    'hour_of_day',
    'transaction_category'
]
X_test = test_data_df[features]

# Step 4: Scale the features using the same scaler used during training
X_test_scaled = scaler.transform(X_test)

# Step 5: Use the model to predict anomalies
test_data_df['predicted_anomaly'] = isolation_forest.predict(X_test_scaled)

# Convert predictions from Isolation Forest (-1 for anomaly, 1 for normal)
# Adjust labels to match the ground truth format (1 for normal, -1 for anomaly)
test_data_df['predicted_anomaly'] = test_data_df['predicted_anomaly'].apply(lambda x: 1 if x == 1 else -1)

# Compare to the actual labels in the dataset (assuming column name is 'actual_anomaly')
y_true = test_data_df['actual_anomaly']  # Ground truth
y_pred = test_data_df['predicted_anomaly']  # Model predictions

# Step 6: Evaluate the model
accuracy = accuracy_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)
classification_rep = classification_report(y_true, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_rep)

# Save results (optional)
test_data_df.to_csv("test_results_with_predictions.csv", index=False)
print("Results saved to 'test_results_with_predictions.csv'")

