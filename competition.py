import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor

# ==========================================
# LOAD DATA
# ==========================================
train = pd.read_csv("VFL_2026_TRAIN_SET.csv")
test = pd.read_csv("VFL_2026_TEST_SET.csv")

print("Training Shape:", train.shape)
print("Testing Shape :", test.shape)

print("\nTraining Columns:")
print(train.columns.tolist())

print("\nFirst five rows:")
print(train.head())

missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

print("\nMissing Values:")
print(missing)

print("\nData Types:\n")
print(train.dtypes)

categorical_cols = train.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = train.select_dtypes(exclude=["object"]).columns.tolist()

print("\nCategorical Columns:")
print(categorical_cols)
print("\nNumber of categorical columns:", len(categorical_cols))

print("\nNumeric Columns:")
print(numeric_cols)
print("\nNumber of numeric columns:", len(numeric_cols))

# ==========================================
# PREPARE DATA
# ==========================================
y = train["ADMIT_LOS"]
X = train.drop(columns=["ADMIT_LOS", "ENCOUNTER_KEY"]).copy()

test_ids = test["ENCOUNTER_KEY"]
test_features = test.drop(columns=["ENCOUNTER_KEY"]).copy()

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

print("\nNumeric features:", len(numeric_features))
print("Categorical features:", len(categorical_features))

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

print("\nPreprocessor created successfully!")

# ==========================================
# TRAIN / VALIDATION SPLIT
# ==========================================
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print("\nTraining samples:", X_train.shape)
print("Validation samples:", X_valid.shape)

# ==========================================
# TRAIN MODEL
# ==========================================
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("xgb", XGBRegressor(
        n_estimators=700,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTraining model...")
model.fit(X_train, y_train)

# ==========================================
# VALIDATION
# ==========================================
valid_pred = model.predict(X_valid)
rmse = root_mean_squared_error(y_valid, valid_pred)
mae = mean_absolute_error(y_valid, valid_pred)

print("\nValidation RMSE:", rmse)
print("Validation MAE :", mae)

# ==========================================
# FINAL MODEL ON FULL DATA
# ==========================================
print("\nTraining final model on full data...")

final_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("xgb", XGBRegressor(
        n_estimators=700,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    ))
])

final_model.fit(X, y)

# ==========================================
# PREDICT TEST + SAVE SUBMISSION
# ==========================================
test_pred = final_model.predict(test_features)
test_pred = np.clip(test_pred, 0, None)

submission = pd.DataFrame({
    "ENCOUNTER_KEY": test_ids,
    "ADMIT_LOS": test_pred
})

submission.to_csv("submission.csv", index=False)

print("\nSaved submission.csv")
print(submission.head())