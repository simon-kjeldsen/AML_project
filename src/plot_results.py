import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import load_df, basic_clean, time_split



# Helper: evaluate model on test set

def evaluate(model, X, y):
    yhat = model.predict(X)
    return {
        "MAE": mean_absolute_error(y, yhat),
        "RMSE": sqrt(mean_squared_error(y, yhat)),
        "R2": r2_score(y, yhat)
    }



# Load data (same preprocessing as training)

df = basic_clean(load_df("data/DKHousingPricesSample100k.csv"))

df = df[(df["date"] >= "2020-01-01") & (df["date"] <= "2024-12-31")]
df = df[(df["sqm"] >= 30) & (df["sqm"] <= 400)]
df = df[(df["sqm_price"] > 1000) & (df["sqm_price"] < 100000)]

train, valid, test = time_split(df)


# Feature engineering (same as training)


# Building age
if {"year_build", "date"}.issubset(df.columns):
    for part in [train, valid, test]:
        part["building_age"] = part["date"].dt.year - part["year_build"]

TARGET = "sqm_price"

# Target encoding: ZIP
if "zip_code" in train.columns:
    avg_zip_price = train.groupby("zip_code")[TARGET].mean()
    for part in [train, valid, test]:
        part["avg_zip_price"] = part["zip_code"].map(avg_zip_price)
        part["avg_zip_price"] = part["avg_zip_price"].fillna(avg_zip_price.mean())

# Target encoding: CITY
if "city" in train.columns:
    avg_city_price = train.groupby("city")[TARGET].mean()
    for part in [train, valid, test]:
        part["avg_city_price"] = part["city"].map(avg_city_price)
        part["avg_city_price"] = part["avg_city_price"].fillna(avg_city_price.mean())

# Target encoding: REGION
if "region" in train.columns:
    avg_region_price = train.groupby("region")[TARGET].mean()
    for part in [train, valid, test]:
        part["avg_region_price"] = part["region"].map(avg_region_price)
        part["avg_region_price"] = part["avg_region_price"].fillna(avg_region_price.mean())


TARGET = "sqm_price"


# Load models

models = {
    "Baseline": joblib.load("models/baseline_simple.joblib"),
    "Linear Regression": joblib.load("models/LinearRegression.joblib"),
    "KNN": joblib.load("models/knn_model.joblib"),
    "Random Forest": joblib.load("models/rf_model.joblib"),
}


# 2. Predicted vs Actual (Random Forest, TEST)

rf = models["Random Forest"]
rf_pipe = rf["pipeline"]
rf_features = rf["features"]

X_test = test[rf_features]
y_test = test[TARGET]
y_pred = rf_pipe.predict(X_test)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle="--")
plt.xlabel("Actual sqm price")
plt.ylabel("Predicted sqm price")
plt.title("Random Forest: Predicted vs Actual (Test set)")
plt.tight_layout()
plt.show()



# 3. Feature importance (Random Forest)

rf_model = rf_pipe.named_steps["model"]
prep = rf_pipe.named_steps["prep"]

feature_names = prep.get_feature_names_out()
importances = rf_model.feature_importances_

fi = (
    pd.DataFrame({"feature": feature_names, "importance": importances})
    .sort_values("importance", ascending=False)
    .head(7)
)

fi.plot(
    kind="barh",
    x="feature",
    y="importance",
    legend=False,
    title="Random Forest Feature Importance (Top 7)"
)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
