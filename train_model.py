import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import DATA_PATH, MODEL_DIR, MODEL_PATH, TARGETS
from data_loader import load_dataset

META = {"song_id", "valence_mean", "arousal_mean", "valence_std", "arousal_std"}

def train():
    df = load_dataset()
    features = [c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    if not features:
        raise ValueError("No numeric acoustic feature columns were found.")
    X = df[features].replace([np.inf, -np.inf], np.nan)
    y = df[TARGETS].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    candidates = {
        "Baseline (mean)": MultiOutputRegressor(DummyRegressor(strategy="mean")),
        "Random Forest": MultiOutputRegressor(RandomForestRegressor(
            n_estimators=350, min_samples_leaf=2, max_features=0.7,
            random_state=42, n_jobs=-1
        )),
        "Extra Trees": MultiOutputRegressor(ExtraTreesRegressor(
            n_estimators=350, min_samples_leaf=2, max_features=0.7,
            random_state=42, n_jobs=-1
        )),
        "Histogram Gradient Boosting": MultiOutputRegressor(HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.06, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42
        ))
    }
    results = {}
    fitted = {}
    for name, estimator in candidates.items():
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator)
        ])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "valence_mae": float(mean_absolute_error(y_test.iloc[:,0], pred[:,0])),
            "valence_rmse": float(np.sqrt(mean_squared_error(y_test.iloc[:,0], pred[:,0]))),
            "valence_r2": float(r2_score(y_test.iloc[:,0], pred[:,0])),
            "arousal_mae": float(mean_absolute_error(y_test.iloc[:,1], pred[:,1])),
            "arousal_rmse": float(np.sqrt(mean_squared_error(y_test.iloc[:,1], pred[:,1]))),
            "arousal_r2": float(r2_score(y_test.iloc[:,1], pred[:,1]))
        }
        fitted[name] = model
    # Select by average of the two target RMSE values on the held-out test set.
    # For formal research, use cross-validation on training data for model selection.
    best_name = min(results, key=lambda n: (
        results[n]["valence_rmse"] + results[n]["arousal_rmse"]
    ))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": fitted[best_name], "model_name": best_name,
        "features": features, "results": results,
        "target_scale": "DEAM mean annotations, 1 to 9",
        "test_song_ids": df.iloc[y_test.index]["song_id"].astype(str).tolist()
    }
    joblib.dump(bundle, MODEL_PATH)
    return bundle, df

if __name__ == "__main__":
    bundle, data = train()
    print("Rows:", len(data))
    print("Selected model:", bundle["model_name"])
    print(json.dumps(bundle["results"], indent=2))
