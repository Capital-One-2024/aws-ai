# aws-ai

### Data coming into myLambda via event JSON:

customer
    - accountID

transactions
    - transactionID
    - amount
    - vendor
    - timestamp
    - transactionCategory

metadata
    - distanceFromPrevious
    - timeSinceLastTransaction

### Data being stored in DynamoDB

- TransactionID: Unique ID of this transaction
- AccountID: Account associated with the transaction
- Amount: Amount of the transaction
- Category: Transaction category (example: food, rent, etc)
- Vendor: Transaction vendor (example: McDonalds, Ubet, etc)
- DistanceFromLastTransaction: How far are we from the last transaction associated with this account
- TimeFromLastTransaction: How much time has passed between this and the last transaction associated with this account
- DateTime: When did the transaction occur
- Prediction: Model's prediction on whether the transaction was fraud (- 1) or not (1)
- PredictionAccurate: Whether the prediction was accurate (1), not accurate (-1), or no response (0) by default
