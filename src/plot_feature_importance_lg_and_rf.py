import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from utils import load_df, basic_clean, choose_features


df = basic_clean(load_df("data/DKHousingPricesSample100k.csv"))
features, target = choose_features(df)

# Helper function
def get_feature_names(pipe):
    prep = pipe.named_steps["prep"]
    num_f = prep.transformers_[0][2]
    cat_f = prep.named_transformers_["cat"].get_feature_names_out(prep.transformers_[1][2])
    return list(num_f) + list(cat_f)


# Linear Regression (baseline)
lr = joblib.load("models/baseline_lr.joblib")["pipeline"]
lr_features = get_feature_names(lr)
lr_model = lr.named_steps["model"]
lr_importance = pd.Series(lr_model.coef_, index=lr_features).sort_values(ascending=False)

plt.figure(figsize=(8,5))
lr_importance.head(15).plot(kind="barh")
plt.title("Linear Regression – Top 15 Coefficients")
plt.xlabel("Coefficient Value (Positive ↑ / Negative ↓)")
plt.tight_layout()
plt.savefig("reports/lr_coefficients.png", dpi=300)
plt.show()


# Random Forest
rf = joblib.load("models/rf_model.joblib")["pipeline"]
rf_features = get_feature_names(rf)
rf_model = rf.named_steps["model"]
rf_importance = pd.Series(rf_model.feature_importances_, index=rf_features).sort_values(ascending=False)

plt.figure(figsize=(8,5))
rf_importance.head(15).plot(kind="barh")
plt.title("Random Forest – Top 15 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("reports/rf_feature_importance.png", dpi=300)
plt.show()
