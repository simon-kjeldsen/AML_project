import argparse
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

from utils import load_df, basic_clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()

    # --------------------------------------------------
    # Load + basic cleaning (samme som train_rf.py)
    # --------------------------------------------------
    df = basic_clean(load_df(args.data))

    if "date" in df.columns:
        df = df[(df["date"] >= "2020-01-01") & (df["date"] <= "2023-12-31")]

    if "house_type" in df.columns:
        df = df[df["house_type"].isin(["Villa", "Apartment"])]

    if "sqm" in df.columns:
        df = df[(df["sqm"] >= 30) & (df["sqm"] <= 400)]

    if "sqm_price" in df.columns:
        df = df[(df["sqm_price"] > 1000) & (df["sqm_price"] < 100000)]

    # --------------------------------------------------
    # Feature engineering
    # --------------------------------------------------
    if {"year_build", "date"}.issubset(df.columns):
        df = df[(df["year_build"] > 1850) & (df["year_build"] <= df["date"].dt.year)]
        df["building_age"] = df["date"].dt.year - df["year_build"]

    TARGET = "sqm_price"

    # --------------------------------------------------
    # Target encoding (global – OK til CV)
    # --------------------------------------------------
    for col, new_col in [
        ("zip_code", "avg_zip_price"),
        ("city", "avg_city_price"),
        ("region", "avg_region_price"),
    ]:
        if col in df.columns:
            means = df.groupby(col)[TARGET].mean()
            df[new_col] = df[col].map(means)
            df[new_col] = df[new_col].fillna(means.mean())

    # --------------------------------------------------
    # Features (samme som RF-træning)
    # --------------------------------------------------
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
        "sales_type"
    ]

    FEATURES_RF = FEATURES_RF_NUM + FEATURES_RF_CAT

    df = df.dropna(subset=FEATURES_RF + [TARGET])

    X = df[FEATURES_RF]
    y = df[TARGET]

    # --------------------------------------------------
    # Pipeline
    # --------------------------------------------------
    prep = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURES_RF_NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), FEATURES_RF_CAT),
        ],
        remainder="drop"
    )

    pipe = Pipeline([
        ("prep", prep),
        ("model", RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            min_samples_split=150,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # --------------------------------------------------
    # 5-fold cross validation
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
