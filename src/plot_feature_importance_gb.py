import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.inspection import permutation_importance
from utils import load_df, basic_clean, choose_features, time_split

# Loads data
df = basic_clean(load_df("data/DKHousingPricesSample100k.csv"))
features, target = choose_features(df)
train, valid, test = time_split(df)
Xte, yte = test[features], test[target]

# Loads model
model_dict = joblib.load("models/gb_model.joblib")
pipe = model_dict["pipeline"]

# Compute permutation importance
r = permutation_importance(pipe, Xte, yte, n_repeats=10, random_state=42, n_jobs=-1)

# Sorts and takes top 15
importances = pd.Series(r.importances_mean, index=features).sort_values(ascending=False)[:15]

plt.figure(figsize=(8, 5))
importances.plot(kind="barh")
plt.title("Gradient Boosting – Top 15 Feature Importances (Permutation Importance)")
plt.xlabel("Importance (Decrease in Model Performance)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("reports/gb_feature_importance.png", dpi=300)
plt.show()
