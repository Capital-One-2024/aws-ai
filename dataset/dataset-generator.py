import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import skewnorm
import random

# Helper functions
def generate_vendor_category():
    vendors = [
        ("Uber", "Transport"), ("Lyft", "Transport"), ("Amazon", "Retail"),
        ("Walmart", "Retail"), ("Starbucks", "Food"), ("Chipotle", "Food"),
        ("McDonald's", "Food"), ("Tuition", "Education"), ("Campus Bookstore", "Books"),
        ("Local Bar", "Entertainment"), ("Gym", "Fitness"), ("Electric Company", "Utilities"),
        ("Water Supplier", "Utilities"), ("Internet Provider", "Bills"), ("Phone Carrier", "Bills"),
        ("Landlord", "Rent"), ("Property Management", "Rent")
    ]
    return random.choice(vendors)

def generate_transaction_time(day_of_week: str):
    hours = list(range(24))
    if day_of_week in ["Friday", "Saturday"]:
        probabilities = [
            0.005, 0.005, 0.005, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1,
            0.12, 0.13, 0.13, 0.1, 0.08, 0.06, 0.04, 0.04, 0.04, 0.05,
            0.05, 0.04, 0.015, 0.01
        ]
    else:
        probabilities = [
            0.005, 0.005, 0.005, 0.005, 0.01, 0.02, 0.04, 0.08, 0.1, 0.12,
            0.13, 0.14, 0.14, 0.12, 0.08, 0.06, 0.04, 0.04, 0.04, 0.04,
            0.03, 0.02, 0.01, 0.005
        ]
    hour = random.choices(hours, probabilities)[0]
    minute = random.randint(0, 59)

    return hour, minute

def generate_amount(category, hour):
    if category == "Education":
        if 8 <= hour <= 20:
            tuition_ranges = [
                (5000, 8000),    # In-state tuition
                (15000, 25000)   # Out-of-state tuition
            ]
            tuition_probabilities = [0.8, 0.2]
            selected_range = random.choices(tuition_ranges, tuition_probabilities)[0]
            return round(random.uniform(*selected_range), 2)
        else:
            return round(random.uniform(1, 300), 2)

    if category == "Rent":
        if 8 <= hour <= 20:
            return round(random.uniform(500, 1800), 2)
        else:
            return round(random.uniform(50, 300), 2)

    elif category in ["Utilities", "Bills"]:
        return round(random.uniform(50, 300), 2)

    if hour < 6 or hour > 22:
        ranges = [
            (1, 100), (101, 300)
        ]
        probabilities = [0.9, 0.1]
    else:
        ranges = [
            (1, 500), (501, 1000), (1001, 2000), (2001, 5000), (5001, 12000)
        ]
        probabilities = [0.7, 0.15, 0.1, 0.04, 0.01]

    selected_range = random.choices(ranges, probabilities)[0]
    return round(random.uniform(*selected_range), 2)

def assign_transport_mode():
    modes = ["Walking", "Biking", "Driving"]
    probabilities = [0.6, 0.3, 0.1]  # Higher likelihood of walking
    return random.choices(modes, probabilities)[0]

def generate_distance_from_last(time_from_last, mode):
    if mode == "Walking":
        typical_speed = random.uniform(0.067, 0.1)  # km/min (4-6 km/h)
    elif mode == "Biking":
        typical_speed = random.uniform(0.25, 0.42)  # km/min (15-25 km/h)
    elif mode == "Driving":
        typical_speed = random.uniform(0.5, 0.83)  # km/min (30-50 km/h)
    
    distance = typical_speed * time_from_last
    return round(min(distance, 25), 2)  # Cap distance at 25 km

def generate_time_from_last():
    short_gap = np.random.exponential(scale=60)
    long_gap = np.random.exponential(scale=720)
    return round(min(short_gap + long_gap, 1440), 0)

def generate_balanced_date():
    today = datetime.now()
    day_offset = random.randint(0, 6)
    return today - timedelta(days=day_offset)

# Generate dataset
def generate_transactions(num_transactions=100000):
    data = []

    for _ in range(num_transactions):
        vendor, category = generate_vendor_category()
        balanced_date = generate_balanced_date()
        day_of_week = balanced_date.strftime("%A")
        hour, minute = generate_transaction_time(day_of_week)
        transaction_time = balanced_date.replace(hour=hour, minute=minute)

        time_from_last = generate_time_from_last()
        transport_mode = assign_transport_mode()
        distance_from_last = generate_distance_from_last(time_from_last, transport_mode)

        # Calculate speed (km/minute)
        speed = distance_from_last / (time_from_last + 1)

        amount = generate_amount(category, hour)

        data.append({
            "Amount": amount,
            "DateTime": transaction_time.strftime("%Y-%m-%d %H:%M:%S"),
            "DistanceFromLastTransaction": distance_from_last,
            "TimeFromLastTransaction": time_from_last,
            "Speed": round(speed, 3),  # Include speed in dataset
            "Vendor": vendor,
            "TransactionCategory": category
        })

    return pd.DataFrame(data)

transactions_df = generate_transactions(2500000)
transactions_df.to_csv("student_transactions.csv", index=False)

