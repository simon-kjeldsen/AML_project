import joblib
import matplotlib.pyplot as plt
import pandas as pd
from utils import load_df, basic_clean, choose_features, time_split

# Load model
model_data = joblib.load("models/baseline_lr.joblib")
pipe = model_data["pipeline"]
features = model_data["features"]
target = model_data["y"]

# Load and prepare data again (same as training)
df = basic_clean(load_df("data/DKHousingPricesSample100k.csv"))
df = df.dropna(subset=features + [target])
_, _, test = time_split(df)
Xte, yte = test[features], test[target]

# Predict
ypred = pipe.predict(Xte)

# Plot
plt.figure(figsize=(7, 7))
plt.scatter(yte, ypred, alpha=0.3)
plt.xlabel("Actual price per sqm")
plt.ylabel("Predicted price per sqm")
plt.title("Baseline Linear Regression - Actual vs Predicted")
plt.plot([yte.min(), yte.max()], [yte.min(), yte.max()], "r--")  # perfekt linje
plt.tight_layout()

# Gem som PNG i reports/
plt.savefig("reports/baseline_performance.png", dpi=300)
plt.show()
