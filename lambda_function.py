import json
import boto3
import joblib
import os
import pandas as pd
from datetime import datetime
import pytz

# Initialize S3 client
s3 = boto3.client('s3')

# Environment Variables
BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
MODEL_KEY = os.environ.get("S3_MODEL_KEY")
SCALER_KEY = os.environ.get("S3_SCALER_KEY")
LABEL_ENCODER_KEY = os.environ.get("S3_LABEL_ENCODER_KEY")
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH")
LOCAL_SCALER_PATH = os.environ.get("LOCAL_SCALER_PATH")
LOCAL_LABEL_ENCODER_PATH = os.environ.get("LOCAL_LABEL_ENCODER_PATH")

def download_model():
    # Download the model, scaler, and label encoder to the Lambda
    s3.download_file(BUCKET_NAME, MODEL_KEY, LOCAL_MODEL_PATH)
    s3.download_file(BUCKET_NAME, SCALER_KEY, LOCAL_SCALER_PATH)
    s3.download_file(BUCKET_NAME, LABEL_ENCODER_KEY, LOCAL_LABEL_ENCODER_PATH)

def load_model():
    # If files are not already downloaded, then download them
    if not os.path.exists(LOCAL_MODEL_PATH) or not os.path.exists(LOCAL_SCALER_PATH) or not os.path.exists(LOCAL_LABEL_ENCODER_PATH):
        download_model()
    # Load the model, scaler, and label encoder
    model = joblib.load(LOCAL_MODEL_PATH)
    scaler = joblib.load(LOCAL_SCALER_PATH)
    label_encoder = joblib.load(LOCAL_LABEL_ENCODER_PATH)
    return model, scaler, label_encoder

# Load the model, scaler, and encoder during initialization
model, scaler, label_encoder = load_model()

def handler(event, context):
    try:
        transactions = event["transactions"]
        all_features = []
        all_predictions = []

        # Extract features from each transaction
        for transaction in transactions:
            features = extract_features_from_transaction(transaction)
            # If we were able to extract features, then add them to the list
            if features:
                all_features.append(features)

        # If we have features, then convert them to a DataFrame
        if all_features:
            df = pd.DataFrame(all_features)

            # Predict whether the transactions are fraudulent
            predictions = fraud_predictor(df)

            # Process the predictions
            for i, prediction in enumerate(predictions):
                transaction = transactions[i]
                is_fraudulent = True if prediction == -1 else False
                all_predictions.append({
                    "id": transaction["id"],
                    "isFraudulent": is_fraudulent,
                })
                
        else:
            print("No features extracted from transactions.")
            
        return {"predictions": all_predictions}
    except Exception as err:
        errMsg = err.args
        return {
            'statusCode': 400,
            'body': json.dumps(errMsg)
        }

def extract_features_from_transaction(transaction):
    try:                
        # Encode transaction category using the loaded label encoder
        category = transaction["category"]
        encoded_category = label_encoder.transform([category])[0]
        
        # Extract day of the week and hour from the timestamp
        timestamp = int(transaction["timestamp"]) / 1000  # Convert milliseconds to seconds
        utc_time = utc_time = datetime.fromtimestamp(timestamp, tz=pytz.utc)
        local_time = utc_time.astimezone(pytz.timezone("America/Chicago"))
                
        day_of_week = local_time.weekday()  # 0 = Monday, 6 = Sunday
        hour_of_day = local_time.hour

        # Calculate the speed feature
        speed = transaction["distanceFromPreviousTransaction"] / (transaction["timeSinceLastTransaction"] + 1)
        
        return {
            "Amount": transaction["amount"],
            "Speed": speed,
            "day_of_week": day_of_week,
            "hour_of_day": hour_of_day,
            "transaction_category": encoded_category  # Use the encoded version
        }
    except Exception as err:
        print("Failed to Parse JSON.")
        return None

def fraud_predictor(df):
    try:
        # Apply the same scaling as the training data
        df_scaled = scaler.transform(df)

        # Batch predict whether the transactions are fraudulent
        return model.predict(df_scaled)
    except Exception as err:
        print(f"Failed to predict whether the transactions were fraudulent. {err}")
        raise err

def format_timestamp(timestamp):
    chicago_tz = pytz.timezone("America/Chicago")
    return datetime.fromtimestamp(int(timestamp), chicago_tz).strftime("%b %d, %Y at %I:%M %p")
