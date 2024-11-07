import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
import joblib

# Step 1: Load and preprocess the dataset
transactions_df = pd.read_csv("../dataset/student_transactions.csv")

# Convert 'DateTime' to 'day_of_week' and 'hour_of_day'
transactions_df['DateTime'] = pd.to_datetime(transactions_df['DateTime'])
transactions_df['day_of_week'] = transactions_df['DateTime'].dt.dayofweek
transactions_df['hour_of_day'] = transactions_df['DateTime'].dt.hour

# Encode 'transaction_category'
label_encoder = LabelEncoder()
transactions_df['transaction_category'] = label_encoder.fit_transform(transactions_df['TransactionCategory'])

# Select the features for training (including speed)
features = [
    'Amount', 
    'Speed',
    'day_of_week', 
    'hour_of_day', 
    'transaction_category'
]
X_train = transactions_df[features]

# Step 2: Apply scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Step 3: Train the Isolation Forest model
isolation_forest = IsolationForest(
    n_estimators=200,    
    contamination=0.015, 
    bootstrap=True,      
    random_state=42
)
isolation_forest.fit(X_train_scaled)

# Step 4: Save the trained model and scaler using joblib
joblib.dump(isolation_forest, "isolation_forest_model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(label_encoder, "label_encoder.joblib")
print("Model, scaler, and encoder saved.")

