import pandas as pd
import numpy as np

print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)

try:
    import sklearn
    print("Scikit-learn:", sklearn.__version__)
except ImportError:
    print("Scikit-learn not installed")

try:
    import xgboost
    print("XGBoost:", xgboost.__version__)
except ImportError:
    print("XGBoost not installed")

try:
    import catboost
    print("CatBoost:", catboost.__version__)
except ImportError:
    print("CatBoost not installed")

try:
    import lightgbm
    print("LightGBM:", lightgbm.__version__)
except ImportError:
    print("LightGBM not installed")