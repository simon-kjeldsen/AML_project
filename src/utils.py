import numpy as np, pandas as pd

def load_df(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "sqm_price" not in df.columns and {"purchase_price","sqm"}.issubset(df.columns):
        df["sqm_price"] = df["purchase_price"] / df["sqm"]
    df = df.replace([np.inf, -np.inf], np.nan)
    if "sqm" in df.columns:
        df = df[(df["sqm"] > 10) & (df["sqm"] < 1000)]
    if "sqm_price" in df.columns:
        df = df[(df["sqm_price"] > 500) & (df["sqm_price"] < 150000)]
    
     # Dropped irrelevant columns
    drop_cols = [
        "house_id",
        "address",
        "quarter",
        "purchase_price",  # since we use sqm_price as target
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df

def choose_features(df: pd.DataFrame):
    num = [c for c in ["sqm","no_rooms","year_build",
                       "nom_interest_rate%","dk_ann_infl_rate%",
                       "yield_on_mortgage_credit_bonds%", "%_change_between_offer_and_purchase"] if c in df.columns]
    cat = [c for c in ["house_type","sales_type","region","zip_code","city", "area"] if c in df.columns]
    target = "sqm_price"
    features = num + cat
    features = [f for f in features if df[f].nunique() > 1]
    return features, target


def time_split(df: pd.DataFrame):
    if "date" in df.columns and df["date"].notna().any():
        df = df.sort_values("date")
        train = df[df["date"] < "2022-01-01"]
        valid = df[(df["date"] >= "2022-01-01") & (df["date"] < "2023-01-01")]
        test  = df[df["date"] >= "2023-01-01"]
        return train, valid, test
    # fallback if no dates
    train = df.sample(frac=0.7, random_state=42)
    rest = df.drop(train.index)
    valid = rest.sample(frac=0.5, random_state=42)
    test  = rest.drop(valid.index)
    return train, valid, test
