import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor

# ==========================================
# LOAD DATA
# ==========================================
train = pd.read_csv("VFL_2026_TRAIN_SET.csv")
test = pd.read_csv("VFL_2026_TEST_SET.csv")

# ==========================================
# FEATURE ENGINEERING
# ==========================================
def add_features(df):
    df = df.copy()

    df["AGE_PLUS_ONE"] = df["PATIENT_AGE"] + 1
    df["VISITS_PLUS_ONE"] = df["NUM_VISITS"] + 1
    df["TEAM_PLUS_ONE"] = df["CARE_TEAM_SIZE"] + 1
    df["CHRONIC_PLUS_ONE"] = df["NUM_CHRONIC_COND"] + 1

    df["CHRONIC_PER_AGE"] = df["NUM_CHRONIC_COND"] / df["AGE_PLUS_ONE"]
    df["HOURS_PER_TEAM"] = df["MONITORING_HOURS"] / df["TEAM_PLUS_ONE"]
    df["ORDERS_PER_VISIT"] = df["ORDER_SET_USED"] / df["VISITS_PLUS_ONE"]
    df["COMORBIDITY_PER_CHRONIC"] = df["COMORBIDITY_INDEX"] / df["CHRONIC_PLUS_ONE"]

    return df

train = add_features(train)
test = add_features(test)

# ==========================================
# PREPARE DATA
# ==========================================
y = train["ADMIT_LOS"]
X = train.drop(columns=["ADMIT_LOS", "ENCOUNTER_KEY"]).copy()
X_test = test.drop(columns=["ENCOUNTER_KEY"]).copy()
test_ids = test["ENCOUNTER_KEY"].copy()

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

base_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("xgb", XGBRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=2.0,
        gamma=0.1,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    ))
])

model = TransformedTargetRegressor(
    regressor=base_model,
    func=np.log1p,
    inverse_func=np.expm1
)

# ==========================================
# 5-FOLD CV
# ==========================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

oof_pred = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

for fold, (train_idx, valid_idx) in enumerate(kf.split(X), 1):
    print(f"\nFold {fold}...")

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    fold_model = TransformedTargetRegressor(
        regressor=base_model,
        func=np.log1p,
        inverse_func=np.expm1
    )

    fold_model.fit(X_train, y_train)
    pred_valid = fold_model.predict(X_valid)
    pred_valid = np.clip(pred_valid, 0, None)

    oof_pred[valid_idx] = pred_valid
    fold_rmse = root_mean_squared_error(y_valid, pred_valid)
    fold_mae = mean_absolute_error(y_valid, pred_valid)

    print("Fold RMSE:", fold_rmse)
    print("Fold MAE :", fold_mae)

    test_pred += np.clip(fold_model.predict(X_test), 0, None) / kf.get_n_splits()

overall_rmse = root_mean_squared_error(y, np.clip(oof_pred, 0, None))
overall_mae = mean_absolute_error(y, np.clip(oof_pred, 0, None))

print("\nOverall CV RMSE:", overall_rmse)
print("Overall CV MAE :", overall_mae)

# ==========================================
# SAVE SUBMISSION
# ==========================================
submission = pd.DataFrame({
    "ENCOUNTER_KEY": test_ids,
    "ADMIT_LOS": test_pred
})

submission.to_csv("submission.csv", index=False)
print("\nSaved submission.csv")
print(submission.head())
