import argparse, joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import load_df, basic_clean, time_split
from math import sqrt


def report(tag, y, yhat):
    mae = mean_absolute_error(y, yhat)
    rmse = sqrt(mean_squared_error(y, yhat))
    r2 = r2_score(y, yhat)
    print(f"{tag}: MAE={mae:,.0f}  RMSE={rmse:,.0f}  R2={r2:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, default="models/knn_model.joblib")
    args = parser.parse_args()
    

    # Basic cleaning
    
    df = basic_clean(load_df(args.data))

    if "date" in df.columns:
        df = df[(df["date"] >= "2020-01-01") & (df["date"] <= "2024-12-31")]

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

    
    # Target encoding of location (Train only)
    
    TARGET = "sqm_price"

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

    
    # Explicit feature choice for KNN
    
    FEATURES_KNN = [
        "sqm",
        "no_rooms",
        "building_age",
        "avg_zip_price",
        "avg_city_price",
        "avg_region_price"
    ]

    train = train.dropna(subset=FEATURES_KNN + [TARGET])
    valid = valid.dropna(subset=FEATURES_KNN + [TARGET])
    test  = test.dropna(subset=FEATURES_KNN + [TARGET])

    Xtr, ytr = train[FEATURES_KNN], train[TARGET]
    Xva, yva = valid[FEATURES_KNN], valid[TARGET]
    Xte, yte = test[FEATURES_KNN],  test[TARGET]

    
    # Preprocessing + model
    
    prep = ColumnTransformer(
        [("num", StandardScaler(), FEATURES_KNN)],
        remainder="drop"
    )

    pipe = Pipeline([
        ("prep", prep),
        ("model", KNeighborsRegressor(
            n_neighbors=50,
            weights="uniform",
            p=1,          # Manhattan distance
            n_jobs=-1
        ))
    ])

    pipe.fit(Xtr, ytr)

    report("TRAIN", ytr, pipe.predict(Xtr))
    report("VALID", yva, pipe.predict(Xva))
    report("TEST ", yte, pipe.predict(Xte))

    joblib.dump(
        {"pipeline": pipe, "features": FEATURES_KNN, "y": TARGET},
        args.out
    )

    print(f"Saved -> {args.out}")
