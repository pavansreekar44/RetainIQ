import pandas as pd
import numpy as np
import shap
import sys
sys.path.append("/mnt/data/HackForge")
from train_model import load_model, encode_categoricals, DEFAULT_DATA_PATH

model, encoders, metadata = load_model("/mnt/data/HackForge/models")
explainer = shap.TreeExplainer(model)

df = pd.read_csv("/mnt/data/HackForge/" + DEFAULT_DATA_PATH)
customer = df.iloc[[0]] # Get one customer

feature_cols = metadata["feature_columns"]
df_encoded, _ = encode_categoricals(customer, encoders=encoders, fit=False)
X = df_encoded[feature_cols]

shap_values = explainer.shap_values(X)
print("shap_values shape:", type(shap_values))
if isinstance(shap_values, np.ndarray):
    print("ndarray shape:", shap_values.shape)
    vals = shap_values[0]
    
print("Vals:", vals)

feature_impacts = list(zip(feature_cols, vals))
feature_impacts.sort(key=lambda x: x[1], reverse=True)
print("Top 3:", feature_impacts[:3])
