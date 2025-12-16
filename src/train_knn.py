import argparse, joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import load_df, basic_clean, choose_features, time_split
from math import sqrt

def report(name, y, yhat):
    mae = mean_absolute_error(y, yhat)
    rmse = sqrt(mean_squared_error(y, yhat))
    r2 = r2_score(y, yhat)
    print(f"{name}: MAE={mae:,.0f}  RMSE={rmse:,.0f}  R2={r2:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, default="models/knn_model.joblib")
    args = parser.parse_args()

    df = basic_clean(load_df(args.data))

    # Filter data (samme som andre modeller)
    if "date" in df.columns:
        df = df[(df["date"] >= "2020-01-01") & (df["date"] <= "2024-12-31")]
    if "house_type" in df.columns:
        df = df[df["house_type"].isin(["Villa", "Apartment"])]
    if "sqm" in df.columns:
        df = df[(df["sqm"] >= 30) & (df["sqm"] <= 400)]
    if "sqm_price" in df.columns:
        df = df[(df["sqm_price"] > 1000) & (df["sqm_price"] < 100000)]
    if {"year_build", "date"}.issubset(df.columns):
        df = df[(df["year_build"] > 1850) & (df["year_build"] <= df["date"].dt.year)]
        df["building_age"] = df["date"].dt.year - df["year_build"]


    features, target = choose_features(df)
    df = df.dropna(subset=features + [target])

    train, valid, test = time_split(df)

    # Print number of observations in each split
    print("Number of observations:")
    print(f"  Train: {len(train):,}")
    print(f"  Validation: {len(valid):,}")
    print(f"  Test: {len(test):,}")
    print(f"  Total (after cleaning): {len(df):,}")

    # Avg price per ZIP
    if "zip_code" in train.columns and "sqm_price" in train.columns:
        avg_price_per_zip = train.groupby("zip_code")["sqm_price"].mean()
        for part in [train, valid, test]:
            part["avg_zip_price"] = part["zip_code"].map(avg_price_per_zip)
            part["avg_zip_price"] = part["avg_zip_price"].fillna(avg_price_per_zip.mean())

    # Avg price per CITY
    if "city" in train.columns and "sqm_price" in train.columns:
        avg_price_per_city = train.groupby("city")["sqm_price"].mean()
        for part in [train, valid, test]:
            part["avg_city_price"] = part["city"].map(avg_price_per_city)
            part["avg_city_price"] = part["avg_city_price"].fillna(avg_price_per_city.mean())

    # Avg price per REGION
    if "region" in train.columns and "sqm_price" in train.columns:
        avg_price_per_region = train.groupby("region")["sqm_price"].mean()
        for part in [train, valid, test]:
            part["avg_region_price"] = part["region"].map(avg_price_per_region)
            part["avg_region_price"] = part["avg_region_price"].fillna(avg_price_per_region.mean())

    # Feature list
    features = ["sqm", "building_age", "no_rooms",
                "avg_zip_price", "avg_city_price", "avg_region_price"]

    Xtr, ytr = train[features], train[target]
    Xva, yva = valid[features], valid[target]
    Xte, yte = test[features], test[target]

    prep = ColumnTransformer(
        [("num", StandardScaler(), features)],
        remainder="drop"
    )

    pipe = Pipeline([
        ("prep", prep),
        ("model", KNeighborsRegressor(
            n_neighbors=30,       # lidt højere K
            weights="distance",   # tættere naboer tæller mere
            p=1,                  # Manhattan distance
            n_jobs=-1
        ))
    ])

    pipe.fit(Xtr, ytr)

    report("VALID", yva, pipe.predict(Xva))
    report("TEST", yte, pipe.predict(Xte))

    joblib.dump({"pipeline": pipe, "features": features, "y": target}, args.out)
    print(f"Saved -> {args.out}")
