# xai_top_reasons.py

import pandas as pd
import joblib
import shap
import numpy as np

# 1. Load your dataset and trained model
df    = pd.read_csv('final_risk_factor.csv')
model = joblib.load('model.pkl')

# 2. Select up to 100 positive‐case records for speed
df_pos = df[df['breast_cancer_history'] == 1]
if len(df_pos) > 100:
    df_pos = df_pos.sample(100, random_state=42)
X = df_pos.drop(columns=['breast_cancer_history'])

# 3. Build the SHAP explainer and compute SHAP values
explainer = shap.TreeExplainer(model)
sv_raw    = explainer.shap_values(X)

# 4. If binary, pick the "positive" class array
if isinstance(sv_raw, (list, tuple)):
    shap_vals = sv_raw[1]
elif isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 3:
    shap_vals = sv_raw[1]
else:
    shap_vals = sv_raw

# unpack only the last two dims (samples, features)
n_samples, n_features = shap_vals.shape[-2:]
# unpack only the last two dims (samples, features)
n_samples, n_features = shap_vals.shape[-2:]
feat_names = list(X.columns)

print(f"DEBUG: shap_vals.shape = {shap_vals.shape}\n")

# 5. Count each record’s #1 driver
top_count = {}
for i in range(n_samples):
    # find the feature index with maximum absolute SHAP value
    j = int(np.argmax(np.abs(shap_vals[i])))
    feat = feat_names[j]
    top_count[feat] = top_count.get(feat, 0) + 1

# 6. Sort features by how often they were the top driver
sorted_top = sorted(top_count.items(), key=lambda x: x[1], reverse=True)

# 7. Print the overall top reasons summary
print("=== OVERALL TOP REASONS ACROSS ALL RECORDS ===")
for feat, cnt in sorted_top:
    print(f"{feat:25s}: top reason in {cnt}/{n_samples} records")
