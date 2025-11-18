import matplotlib.pyplot as plt
import joblib
import pandas as pd
from utils import load_df, basic_clean, choose_features, time_split

df = basic_clean(load_df("data/DKHousingPricesSample100k.csv"))
features, target = choose_features(df)
df = df.dropna(subset=features + [target])
train, valid, test = time_split(df)
Xte, yte = test[features], test[target]

models = {
    "Linear Regression": joblib.load("models/baseline_lr.joblib")["pipeline"],
    "Random Forest": joblib.load("models/rf_model.joblib")["pipeline"], 
    "Gradient Boosting": joblib.load("models/gb_model.joblib")["pipeline"]
}

results = {}
for name, pipe in models.items():
    yhat = pipe.predict(Xte)
    mae = abs(yhat - yte).mean()
    rmse = ((yhat - yte)**2).mean()**0.5
    r2 = 1 - ((yte - yhat)**2).sum() / ((yte - yte.mean())**2).sum()
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

results_df = pd.DataFrame(results).T
print(results_df)

results_df[["MAE", "RMSE"]].plot(kind="bar")
plt.title("Model Comparison (Lower = Better)")
plt.tight_layout()
plt.savefig("reports/model_comparison.png", dpi=300)
plt.show()
