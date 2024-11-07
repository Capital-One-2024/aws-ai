import pandas as pd
import matplotlib.pyplot as plt

# Use non-interactive backend
plt.switch_backend('Agg')

# Load the dataset
transactions_df = pd.read_csv("student_transactions.csv")

# Convert 'DateTime' to datetime object for better plotting
transactions_df['DateTime'] = pd.to_datetime(transactions_df['DateTime'])

# Plot 1: Distribution of Amounts
plt.figure(figsize=(10, 6))
plt.hist(transactions_df['Amount'], bins=30, edgecolor='black', alpha=0.7)
plt.title("Distribution of Transaction Amounts")
plt.xlabel("Amount ($)")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("distribution_of_amounts.png")
plt.close()

# Plot 2: Average Transaction Amount by Category
avg_amount_by_category = transactions_df.groupby('TransactionCategory')['Amount'].mean().sort_values()
plt.figure(figsize=(10, 6))
avg_amount_by_category.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Average Transaction Amount by Category")
plt.xlabel("Transaction Category")
plt.ylabel("Average Amount ($)")
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.savefig("avg_transaction_by_category.png")
plt.close()

# Plot 3: Scatter Plot of Delta Time vs Delta Distance
plt.figure(figsize=(10, 6))
plt.scatter(transactions_df['TimeFromLastTransaction'], transactions_df['DistanceFromLastTransaction'], alpha=0.5, color='purple')
plt.title("Delta Time vs Delta Distance")
plt.xlabel("Time from Last Transaction (minutes)")
plt.ylabel("Distance from Last Transaction (km)")
plt.grid(True)
plt.savefig("delta_time_vs_delta_distance.png")
plt.close()

# Plot 4: Average Transaction Amount by Hour of the Day
transactions_df['Hour'] = transactions_df['DateTime'].dt.hour
avg_amount_by_hour = transactions_df.groupby('Hour')['Amount'].mean()
plt.figure(figsize=(10, 6))
avg_amount_by_hour.plot(kind='line', marker='o', color='blue')
plt.title("Average Transaction Amount by Hour of the Day")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Amount ($)")
plt.xticks(range(0, 24))
plt.grid(True)
plt.savefig("avg_transaction_by_hour.png")
plt.close()

# Plot 5: Total Transaction Amount by Day of the Week
transactions_df['DayOfWeek'] = transactions_df['DateTime'].dt.day_name()
total_amount_by_day = transactions_df.groupby('DayOfWeek')['Amount'].sum().reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
plt.figure(figsize=(10, 6))
total_amount_by_day.plot(kind='bar', color='green', edgecolor='black')
plt.title("Total Transaction Amount by Day of the Week")
plt.xlabel("Day of the Week")
plt.ylabel("Total Amount ($)")
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.savefig("total_transaction_by_day.png")
plt.close()

print("Selected visualizations saved as images.")

