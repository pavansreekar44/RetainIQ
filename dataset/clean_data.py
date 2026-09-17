import pandas as pd
import numpy as np

# 1. Download the raw dataset
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# 2. Fix the blank spaces in TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan)).fillna(0)

# 3. FILTER NOW (Keep only the 7 Universal columns + ID + Target)
core_columns = [
    'customerID', 
    'Contract', 
    'tenure', 
    'MonthlyCharges', 
    'TotalCharges',
    'PaymentMethod', 
    'TechSupport', 
    'PaperlessBilling', 
    'Churn'
]
df_filtered = df[core_columns]

# 4. Save the cleanly filtered file for the UI team to use
df_filtered.to_csv("cleaned_customers.csv", index=False)

print("Saved cleaned_customers.csv with exactly", len(df_filtered.columns), "columns.")
