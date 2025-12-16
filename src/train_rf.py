import argparse
import os
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import load_df, basic_clean, time_split


def report(tag, y, yhat):
    mae = mean_absolute_error(y, yhat)
    rmse = mean_squared_error(y, yhat) ** 0.5
    r2 = r2_score(y, yhat)
    print(f"{tag}: MAE={mae:,.0f}  RMSE={rmse:,.0f}  R2={r2:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, default="models/rf_model.joblib")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    
    # Basic cleaning
    
    df = basic_clean(load_df(args.data))

    if "date" in df.columns:
        df = df[(df["date"] >= "2020-01-01") & (df["date"] <= "2023-12-31")]

    if "house_type" in df.columns:
        df = df[df["house_type"].isin(["Villa", "Apartment"])]

    if "sqm" in df.columns:
        df = df[(df["sqm"] >= 30) & (df["sqm"] <= 400)]

    if "sqm_price" in df.columns:
        df = df[(df["sqm_price"] > 1000) & (df["sqm_price"] < 100000)]

    
    # Feature engineering
    
    if {"year_build", "date"}.issubset(df.columns):
        df = df[(df["year_build"] > 1850) & (df["year_build"] <= df["date"].dt.year)]
        df["building_age"] = df["date"].dt.year - df["year_build"]

    
    # Train / validation / test split
    
    train, valid, test = time_split(df)

    print("Number of observations:")
    print(f"  Train: {len(train):,}")
    print(f"  Validation: {len(valid):,}")
    print(f"  Test: {len(test):,}")
    print(f"  Total (after cleaning): {len(df):,}")

    TARGET = "sqm_price"

   
    # Target encoding of location (TRAIN only)
    
    if "zip_code" in train.columns:
        avg_zip_price = train.groupby("zip_code")[TARGET].mean()
        for part in [train, valid, test]:
            part["avg_zip_price"] = part["zip_code"].map(avg_zip_price)
            part["avg_zip_price"] = part["avg_zip_price"].fillna(avg_zip_price.mean())

    if "city" in train.columns:
        avg_city_price = train.groupby("city")[TARGET].mean()
        for part in [train, valid, test]:
            part["avg_city_price"] = part["city"].map(avg_city_price)
            part["avg_city_price"] = part["avg_city_price"].fillna(avg_city_price.mean())

    if "region" in train.columns:
        avg_region_price = train.groupby("region")[TARGET].mean()
        for part in [train, valid, test]:
            part["avg_region_price"] = part["region"].map(avg_region_price)
            part["avg_region_price"] = part["avg_region_price"].fillna(avg_region_price.mean())

    
    # Explicit feature choice for RANDOM FOREST
    
    FEATURES_RF_NUM = [
        "sqm",
        "no_rooms",
        "building_age",
        "avg_zip_price",
        "avg_city_price",
        "avg_region_price"
    ]

    FEATURES_RF_CAT = [
        "house_type",
        "sales_type",
        "area"
    ]

    FEATURES_RF = FEATURES_RF_NUM + FEATURES_RF_CAT

    train = train.dropna(subset=FEATURES_RF + [TARGET])
    valid = valid.dropna(subset=FEATURES_RF + [TARGET])
    test  = test.dropna(subset=FEATURES_RF + [TARGET])

    Xtr, ytr = train[FEATURES_RF], train[TARGET]
    Xva, yva = valid[FEATURES_RF], valid[TARGET]
    Xte, yte = test[FEATURES_RF],  test[TARGET]

    
    # Preprocessing + model
    
    prep = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURES_RF_NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), FEATURES_RF_CAT)
        ],
        remainder="drop"
    )

    pipe = Pipeline([
        ("prep", prep),
        ("model", RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipe.fit(Xtr, ytr)

    report("VALID", yva, pipe.predict(Xva))
    report("TEST ", yte, pipe.predict(Xte))

    joblib.dump(
        {"pipeline": pipe, "features": FEATURES_RF, "y": TARGET},
        args.out
    )

    print(f"Saved -> {args.out}")
