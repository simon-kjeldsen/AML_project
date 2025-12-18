import argparse
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression

from utils import load_df, basic_clean, time_split


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()

    
    # Load & clean data (same as baseline)
    
    df = basic_clean(load_df(args.data))

    if "date" in df.columns:
        df = df[(df["date"] >= "2020-01-01") & (df["date"] <= "2024-12-31")]
    if "sqm" in df.columns:
        df = df[(df["sqm"] >= 30) & (df["sqm"] <= 400)]
    if "sqm_price" in df.columns:
        df = df[(df["sqm_price"] > 1000) & (df["sqm_price"] < 100000)]

    FEATURES_BASELINE = ["sqm", "city"]
    TARGET = "sqm_price"

    df = df.dropna(subset=FEATURES_BASELINE + [TARGET])

    
    # Train / valid / test split
    
    train, _, _ = time_split(df)

    Xtr = train[FEATURES_BASELINE]
    ytr = train[TARGET]

    
    # Pipeline (same as baseline model)
    
    prep = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["sqm"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["city"])
        ],
        remainder="drop"
    )

    pipe = Pipeline([
        ("prep", prep),
        ("model", LinearRegression())
    ])

    
    # 5-fold Cross Validation
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_val_score(
        pipe,
        Xtr,
        ytr,
        cv=cv,
        scoring="r2",
        n_jobs=1
    )

    print("Baseline CV R2 scores:", scores)
    print("Mean CV R2:", scores.mean())
    print("Std CV R2:", scores.std())
