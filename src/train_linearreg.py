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
    # Using only data from 2020-01-01 to 2024-12-31 if date is available
    if "date" in df.columns:
        df = df[(df["date"] >= "2020-01-01") & (df["date"] <= "2024-12-31")]
    # Keeping only Villas and Apartments if the column exists
    if "house_type" in df.columns:
        df = df[df["house_type"].isin(["Villa", "Apartment"])]

    # Remove extreme property sizes
    if "sqm" in df.columns:
        df = df[(df["sqm"] >= 30) & (df["sqm"] <= 400)]

    if "sqm_price" in df.columns:
        df = df[(df["sqm_price"] > 1000) & (df["sqm_price"] < 100000)]
    
    # Filter unrealistic construction years
    if "year_build" in df.columns:
        df = df[(df["year_build"] > 1850) & (df["year_build"] < 2024)]
    
    if "year_build" in df.columns:
        df["building_age"] = 2025 - df["year_build"]


    features, target = choose_features(df)
    df = df.dropna(subset=features+[target])

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
            part["avg_zip_price"] = part["avg_zip_price"].fillna(train["avg_zip_price"].mean())

    # Avg price per CITY
    if "city" in train.columns and "sqm_price" in train.columns:
        avg_price_per_city = train.groupby("city")["sqm_price"].mean()
        for part in [train, valid, test]:
            part["avg_city_price"] = part["city"].map(avg_price_per_city)
            part["avg_city_price"] = part["avg_city_price"].fillna(train["avg_city_price"].mean())
    
    # Avg price per REGION
    if "region" in train.columns and "sqm_price" in train.columns:
        avg_price_per_region = train.groupby("region")["sqm_price"].mean()
        for part in [train, valid, test]:
            part["avg_region_price"] = part["region"].map(avg_price_per_region)
            part["avg_region_price"] = part["avg_region_price"].fillna(train["avg_region_price"].mean())


    # Update feature list

    features = ["sqm", "building_age", "avg_zip_price", "avg_city_price", "avg_region_price"]


    Xtr, ytr = train[features], train[target]
    Xva, yva = valid[features], valid[target]
    Xte, yte = test[features],  test[target]


    # Numeric and categorical feature lists from training data
    num_f = Xtr.select_dtypes(include="number").columns.tolist()
    cat_f = Xtr.select_dtypes(exclude="number").columns.tolist()

    prep = ColumnTransformer(
        [("num", StandardScaler(), num_f)], remainder="drop"
    )
    pipe = Pipeline([("prep", prep), ("model", LinearRegression())])
    pipe.fit(Xtr, ytr)

    report("VALID", yva, pipe.predict(Xva))
    report("TEST ", yte, pipe.predict(Xte))

    joblib.dump({"pipeline": pipe, "features": features, "y": target}, args.out)
    print(f"Saved -> {args.out}")
