import os, argparse, joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from math import sqrt
import numpy as np
from utils import load_df, basic_clean, time_split

def report(tag, y, yhat):
    mae  = mean_absolute_error(y, yhat)
    rmse = sqrt(mean_squared_error(y, yhat))
    r2   = r2_score(y, yhat)
    print(f"{tag}: MAE={mae:,.0f}  RMSE={rmse:,.0f}  R2={r2:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/DKHousingPricesSample100k.csv")
    ap.add_argument("--out",  default="models/baseline_simple.joblib")
    args = ap.parse_args()

    os.makedirs("models", exist_ok=True)

    # Basic cleaning 
    
    df = basic_clean(load_df(args.data))

    # Keep realistic values
    if "date" in df.columns:
        df = df[(df["date"] >= "2020-01-01") & (df["date"] <= "2024-12-31")]
    if "sqm" in df.columns:
        df = df[(df["sqm"] >= 30) & (df["sqm"] <= 400)]
    if "sqm_price" in df.columns:
        df = df[(df["sqm_price"] > 1000) & (df["sqm_price"] < 100000)]

    # Only include simple, high-level features
    features = []
    if "sqm" in df.columns:
        features.append("sqm")
    if "city" in df.columns:
        features.append("city")
    if not features:
        raise ValueError("Required columns ('sqm', 'city') not found in data")

    target = "sqm_price"

    df = df.dropna(subset=features + [target])

    train, valid, test = time_split(df)

    # Print number of observations in each split
    print("Number of observations:")
    print(f"  Train: {len(train):,}")
    print(f"  Validation: {len(valid):,}")
    print(f"  Test: {len(test):,}")
    print(f"  Total (after cleaning): {len(df):,}")

    Xtr, ytr = train[features], train[target]
    Xva, yva = valid[features], valid[target]
    Xte, yte = test[features], test[target]

    # Simple preprocessing (numeric + categorical)
    num_f = [f for f in features if df[f].dtype != "object"]
    cat_f = [f for f in features if df[f].dtype == "object"]

    prep = ColumnTransformer(
        [("num", StandardScaler(), num_f),
         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_f)],
        remainder="drop"
    )

    pipe = Pipeline([("prep", prep), ("model", LinearRegression())])

    # Train
    pipe.fit(Xtr, ytr)

    # Evaluate
    print("\n--- BASELINE (simple) MODEL ---")
    report("VALID", yva, pipe.predict(Xva))
    report("TEST ", yte, pipe.predict(Xte))

    # Save
    joblib.dump({"pipeline": pipe, "features": features, "y": target}, args.out)
    print(f"Saved -> {args.out}")
