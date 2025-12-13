import argparse
import joblib
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils import load_df, basic_clean, choose_features, time_split


def report(name, y, yhat):
    mae = mean_absolute_error(y, yhat)
    rmse = mean_squared_error(y, yhat) ** 0.5
    r2 = r2_score(y, yhat)
    print(f"{name}: MAE={mae:,.0f}  RMSE={rmse:,.0f}  R2={r2:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, default="models/gb_model.joblib")
    args = parser.parse_args()

    # Loads and cleans data
    df = basic_clean(load_df(args.data))
    features, target = choose_features(df)
    df = df.dropna(subset=features + [target])  # removes missing data for fairness

    # Splits data
    train, valid, test = time_split(df)
    Xtr, ytr = train[features], train[target]
    Xva, yva = valid[features], valid[target]
    Xte, yte = test[features], test[target]

    # Separates feature types
    num_f = Xtr.select_dtypes(include="number").columns.tolist()
    cat_f = Xtr.select_dtypes(exclude="number").columns.tolist()

    # Preprocessing: encode categoricals, scale numericals
    prep = ColumnTransformer(
        [("num", StandardScaler(), num_f),
         ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), cat_f)],
        remainder="drop"
    )

    # Model pipeline (HistGradientBoosting handles NaNs and complex relations)
    pipe = Pipeline([
        ("prep", prep),
        ("model", HistGradientBoostingRegressor(
            learning_rate=0.1,
            max_depth=10,
            max_iter=300,
            random_state=42
        ))
    ])

    # Trains model
    pipe.fit(Xtr, ytr)

    # Evaluates
    report("VALID", yva, pipe.predict(Xva))
    report("TEST", yte, pipe.predict(Xte))

    # Saves
    joblib.dump({"pipeline": pipe, "features": features, "y": target}, args.out)
    print(f"Saved -> {args.out}")
