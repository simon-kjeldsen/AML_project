import os, argparse, joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils import load_df, basic_clean, choose_features, time_split

from math import sqrt
def report(tag, y, yhat):
    mae  = mean_absolute_error(y, yhat)
    rmse = sqrt(mean_squared_error(y, yhat))
    r2   = r2_score(y, yhat)
    print(f"{tag}: MAE={mae:,.0f}  RMSE={rmse:,.0f}  R2={r2:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/DKHousingPricesSample100k.csv")
    ap.add_argument("--out",  default="models/baseline_lr.joblib")
    args = ap.parse_args()

    os.makedirs("models", exist_ok=True)

    df = basic_clean(load_df(args.data))
    features, num_f, cat_f, target = choose_features(df)
    df = df.dropna(subset=features+[target])

    train, valid, test = time_split(df)
    Xtr, ytr = train[features], train[target]
    Xva, yva = valid[features], valid[target]
    Xte, yte = test[features],  test[target]

    prep = ColumnTransformer(
        [("num", StandardScaler(), num_f),
         ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat_f)],
        remainder="drop"
    )
    pipe = Pipeline([("prep", prep), ("model", LinearRegression())])
    pipe.fit(Xtr, ytr)

    report("VALID", yva, pipe.predict(Xva))
    report("TEST ", yte, pipe.predict(Xte))

    joblib.dump({"pipeline": pipe, "features": features, "y": target}, args.out)
    print(f"Saved -> {args.out}")
