import argparse
import numpy as np
from math import sqrt

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

from utils import load_df, basic_clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()

    # ----------------------------------
    # Load & clean data (same as KNN)
    # ----------------------------------
    df = basic_clean(load_df(args.data))

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

    TARGET = "sqm_price"

    # Target encoding (same logic, but on full CV data)
    for col in ["zip_code", "city", "region"]:
        if col in df.columns:
            mean_map = df.groupby(col)[TARGET].mean()
            df[f"avg_{col}_price"] = df[col].map(mean_map)
            df[f"avg_{col}_price"] = df[f"avg_{col}_price"].fillna(mean_map.mean())

    FEATURES_KNN = [
        "sqm",
        "no_rooms",
        "building_age",
        "avg_zip_code_price",
        "avg_city_price",
        "avg_region_price"
    ]

    df = df.dropna(subset=FEATURES_KNN + [TARGET])

    X = df[FEATURES_KNN]
    y = df[TARGET]

    # ----------------------------------
    # Pipeline
    # ----------------------------------
    pipe = Pipeline([
        ("prep", ColumnTransformer(
            [("num", StandardScaler(), FEATURES_KNN)],
            remainder="drop"
        )),
        ("model", KNeighborsRegressor(
            n_neighbors=50,
            weights="uniform",
            p=1
        ))
    ])

    # ----------------------------------
    # Cross validation
    # ----------------------------------
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(
        pipe,
        X,
        y,
        cv=cv,
        scoring="r2"
    )

    print("CV R² scores:", np.round(scores, 3))
    print("Mean CV R²:", scores.mean())
    print("Std CV R²:", scores.std())
