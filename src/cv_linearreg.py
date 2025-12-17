import argparse
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from utils import load_df, basic_clean, time_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()

    # --------------------------------------------------
    # Load + clean data (same as training)
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Train / validation / test split
    # (CV uses TRAIN only)
    # --------------------------------------------------
    train, _, _ = time_split(df)

    TARGET = "sqm_price"

    # Target encoding (TRAIN only)
    if "zip_code" in train.columns:
        avg_zip_price = train.groupby("zip_code")[TARGET].mean()
        train["avg_zip_price"] = train["zip_code"].map(avg_zip_price)
        train["avg_zip_price"] = train["avg_zip_price"].fillna(avg_zip_price.mean())

    if "city" in train.columns:
        avg_city_price = train.groupby("city")[TARGET].mean()
        train["avg_city_price"] = train["city"].map(avg_city_price)
        train["avg_city_price"] = train["avg_city_price"].fillna(avg_city_price.mean())

    if "region" in train.columns:
        avg_region_price = train.groupby("region")[TARGET].mean()
        train["avg_region_price"] = train["region"].map(avg_region_price)
        train["avg_region_price"] = train["avg_region_price"].fillna(avg_region_price.mean())

    # --------------------------------------------------
    # Features (same as Linear Regression model)
    # --------------------------------------------------
    FEATURES_LR = [
        "sqm",
        "no_rooms",
        "building_age",
        "avg_zip_price",
        "avg_city_price",
        "avg_region_price"
    ]

    train = train.dropna(subset=FEATURES_LR + [TARGET])

    X = train[FEATURES_LR]
    y = train[TARGET]

    # --------------------------------------------------
    # Pipeline
    # --------------------------------------------------
    pipe = Pipeline([
        ("prep", ColumnTransformer(
            [("num", StandardScaler(), FEATURES_LR)],
            remainder="drop"
        )),
        ("model", LinearRegression())
    ])

    # --------------------------------------------------
    # 5-fold Cross Validation
    # --------------------------------------------------
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(
        pipe,
        X,
        y,
        cv=cv,
        scoring="r2",
        n_jobs=-1
    )

    print("CV R2 scores:", np.round(scores, 3))
    print("Mean CV R2 :", scores.mean())
    print("Std CV R2  :", scores.std())
