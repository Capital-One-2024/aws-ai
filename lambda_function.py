import json
import boto3
import joblib
import os
import pandas as pd
from datetime import datetime
import pytz

# Initialize S3 client
s3 = boto3.client('s3')

# Initialize DynamoDB resource
dynamodb = boto3.resource('dynamodb')

# Environment Variables
BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
MODEL_KEY = os.environ.get("S3_MODEL_KEY")
SCALER_KEY = os.environ.get("S3_SCALER_KEY")
LABEL_ENCODER_KEY = os.environ.get("S3_LABEL_ENCODER_KEY")
LOCAL_MODEL_PATH = os.environ.get("LOCAL_MODEL_PATH")
LOCAL_SCALER_PATH = os.environ.get("LOCAL_SCALER_PATH")
LOCAL_LABEL_ENCODER_PATH = os.environ.get("LOCAL_LABEL_ENCODER_PATH")
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN")

# once SES TOPIC ARN is setup, please remove comment, and uncomment the
# code below I commented since I didn't want any possible errors
SES_SOURCE_EMAIL = os.environ.get("SES_SOURCE_EMAIL")
SEND_EMAILS_WHEN_FRAUD = os.environ.get("SEND_EMAILS_WHEN_FRAUD")
SEND_EMAILS_WHEN_NOT_FRAUD = os.environ.get("SEND_EMAILS_WHEN_NOT_FRAUD")
TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME")

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

# Initialize SNS client
sns_client = boto3.client("sns")

# Load the model, scaler, and encoder during initialization
model, scaler, label_encoder = load_model()

def handler(event, context):
    try:
        transactions = event["Records"]
        all_features = []

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
                t = transactions[i]
                process_prediction(t, prediction)

                # insert this data into dynamodb
                insert_into_dynamodb(t, prediction)
        else:
            print("No features extracted from transactions.")

        return {
            'statusCode': 200,
            'body': json.dumps('Transactions Processed!')
        }
    except Exception as err:
        errMsg = err.args
        return {
            'statusCode': 400,
            'body': json.dumps(errMsg)
        }

def getBody(transaction):
    return json.loads(transaction["body"])

def extract_features_from_transaction(transaction):
    try:
        body = getBody(transaction)

        customer_data = body["customer"]
        transaction_data = body["transaction"]
        metadata = body["metadata"]

        # Encode transaction category using the loaded label encoder
        category = transaction_data["transactionCategory"]
        encoded_category = label_encoder.transform([category])[0]

        # Extract day of the week and hour from the timestamp
        timestamp = int(transaction_data["timestamp"])
        date_time = datetime.fromtimestamp(timestamp)
        day_of_week = date_time.weekday()  # 0 = Monday, 6 = Sunday
        hour_of_day = date_time.hour

        # Calculate the speed feature
        speed = metadata["distanceFromPrevious"] / (metadata["timeSinceLastTransaction"] + 1)

        return {
            "Amount": transaction_data["amount"],
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

ses_client = boto3.client("ses")
SES_SOURCE_EMAIL = os.environ.get("SES_SOURCE_EMAIL")  # Your verified email in SES

def send_email(subject, body, recipient_email):
    """
    Sends an email using Amazon SES.
    """
    try:
        response = ses_client.send_email(
            Source=SES_SOURCE_EMAIL,
            Destination={"ToAddresses": [recipient_email]},
            Message={
                "Subject": {"Data": subject},
                "Body": {"Html": {"Data": body}}
            },
        )
        print(f"Email sent successfully to {recipient_email}.")
        return response
    except Exception as e:
        print(f"Failed to send email. {e}")
        raise e


def send_notification(account_id, amount, date_time, is_fraudulent, recipient_email):
    """
    Sends a notification email using SES based on transaction details.
    """
    subject = ""
    body = ""

    if is_fraudulent:
        # Check if email notifications for fraudulent transactions are enabled
        if SEND_EMAILS_WHEN_FRAUD == "false":
            return
        print("Sending Fraud Alert")

        subject = "Capital One - Fraud Alert"
        body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f9f9f9; }}
                .email-container {{ max-width: 600px; margin: auto; background: #fff; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .header {{ text-align: center; padding: 20px; }}
                .header img {{ max-width: 150px; }}
                .title-bar {{ background: #d32f2f; color: #fff; padding: 10px 15px; text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 20px; }}
                .content {{ padding: 20px; font-size: 16px; color: #333; line-height: 1.6; }}
                .footer {{ padding: 10px; font-size: 12px; text-align: center; color: #999; border-top: 1px solid #ddd; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="email-container">
                <div class="header">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/Capital_One_logo.svg/1200px-Capital_One_logo.svg.png" alt="Capital One Logo">
                </div>
                <div class="title-bar">Security Alert, Suspicious Activity Detected</div>
                
                <div class="content">
                    <p>Dear Customer,</p>
                    <p>We have detected a suspicious transaction on your account, which could indicate potential fraud. Please review the details below:</p>
                    <ul>
                        <li><b>Transaction Amount:</b> ${amount}</li>
                        <li><b>Transaction Location:</b> New York, NY</li>
                        <li><b>Transaction Date:</b> {date_time}</li>
                    </ul>
                    <p>For your protection, Capital One Security has temporarily blocked this transaction to safeguard your account. If this transaction is valid, you can approve it to proceed; otherwise, we recommend securing your account immediately.</p>
                    <p>If this transaction was not made by you, we strongly recommend taking immediate action to secure your account.</p>
                    <p>Thank you for your prompt attention to this matter.</p>
                    <p>Regards,</p>
                    <p>Capital One Security Team</p>
                </div>
                <div class="footer">
                    This is an automated email. Please do not reply.
                </div>
            </div>
        </body>
        </html>
        """
    else:
        # Check if email notifications for non-fraudulent transactions are enabled
        if SEND_EMAILS_WHEN_NOT_FRAUD == "false":
            return
        print("Sending Transaction Alert")

        subject = "Capital One - New Transaction"
        body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f9f9f9; }}
                .email-container {{ max-width: 600px; margin: auto; background: #fff; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .header {{ text-align: center; padding: 20px; }}
                .header img {{ max-width: 150px; }}
                .title-bar {{ background: #004878; color: #fff; padding: 10px 15px; text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 20px; }}
                .content {{ padding: 20px; font-size: 16px; color: #333; line-height: 1.6; }}
                .footer {{ padding: 10px; font-size: 12px; text-align: center; color: #999; border-top: 1px solid #ddd; margin-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="email-container">
                <div class="header">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/Capital_One_logo.svg/1200px-Capital_One_logo.svg.png" alt="Capital One Logo">
                </div>
                <div class="title-bar">Transaction Successful</div>
                
                <div class="content">
                    <p>Dear Customer,</p>
                    <p>We are pleased to inform you that your recent transaction was successfully processed. Here are the details of the transaction:</p>
                    <ul>
                        <li><b>Transaction Amount:</b> ${amount}</li>
                        <li><b>Transaction Location:</b> New York, NY</li>
                        <li><b>Transaction Date:</b> {date_time}</li>
                    </ul>
                    <p>If you recognize this transaction, no further action is required. If you have any questions or concerns, please don’t hesitate to contact our support team.</p>
                    <p>Thank you for choosing Capital One for your financial needs.</p>
                    <p>Regards,</p>
                    <p>Capital One Customer Service Team</p>
                </div>
                <div class="footer">
                    This is an automated email. Please do not reply.
                </div>
            </div>
        </body>
        </html>
        """

    # Send the email using SES
    return send_email(subject, body, recipient_email)

# please adjust the process_prediction with the new SES accordingly
def process_prediction(transaction, prediction):
    try:
        body = getBody(transaction)
        account_id = body["customer"]["accountID"]
        amount = body["transaction"]["amount"]
        timestamp = body["transaction"]["timestamp"]
        date_time = format_timestamp(timestamp)
        print(f"Processing Prediction: {prediction} for Account #{account_id} with ${amount} on {date_time}")
        send_notification(account_id, amount, date_time, prediction == -1)
    except Exception as err:
        print(f"Failed to process prediction. {err}")
        raise err

def insert_dynamodb(transaction, prediction):
    # Define the table
    table = dynamodb.Table(TABLE_NAME)

    try:
        body = getBody(transaction)

        customer_data = body["customer"]
        transaction_data = body["transaction"]
        metadata = body["metadata"]

        account_id = customer_data["accountID"]

        transaction_id = customer_data["transactionID"]
        amount = transaction_data["amount"]

        # assuming we pass vendor here, see README
        vendor = transaction_data["vendor"]

        timestamp = transaction_data["timestamp"]
        date_time = format_timestamp(timestamp)

        category = transaction_data["transactionCategory"]

        distance_from_previous = metadata["distanceFromPrevious"]
        time_from_last_transacton = metadata["timeSinceLastTransaction"]

        item = {
            "TransactionID": transaction_id,
            "AccountID": account_id,
            "Amount": amount,
            "Category": category,
            "Vendor": vendor, 
            "DistanceFromLastTransaction": distance_from_previous,
            "TimeFromLastTransaction": time_from_last_transacton,
            "DateTime": date_time,
            "Prediction": prediction,
            "PredictionAccurate": 0
        }
    
        print(f"Inserting into DynamoDB prediction: {prediction} for Account #{account_id} with ${amount} on {date_time}")
        response = table.put_item(Item=item)
    except Exception as err:
        print(f"Failed to insert into DynamoDB. {err}")
        raise err