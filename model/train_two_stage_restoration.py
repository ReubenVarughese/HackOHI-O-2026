"""Train a two-stage restoration baseline.

Stage 1 predicts same-day versus extended repair. Stage 2 predicts hours only
for extended repairs, avoiding a single regressor being dominated by zeros.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor

from train_restoration_model import build_features, feature_columns, read_damage


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"


def make_preprocessor(features):
    categorical, numeric = feature_columns(features)
    transformer = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ]
    )
    return transformer, categorical, numeric


def main() -> None:
    features, target, report_time = build_features(read_damage())
    plausible = target.le(30 * 24)
    features = features.loc[plausible].reset_index(drop=True)
    target = target.loc[plausible].reset_index(drop=True)
    report_time = report_time.loc[plausible].reset_index(drop=True)
    order = np.argsort(report_time.to_numpy())
    features = features.iloc[order].reset_index(drop=True)
    target = target.iloc[order].reset_index(drop=True)
    split = max(1, int(len(features) * 0.8))
    train_x, test_x = features.iloc[:split], features.iloc[split:]
    train_y, test_y = target.iloc[:split], target.iloc[split:]
    train_extended = train_y.gt(0).astype(int)
    test_extended = test_y.gt(0).astype(int)

    classifier_transformer, categorical, numeric = make_preprocessor(features)
    classifier = Pipeline(
        [
            ("features", classifier_transformer),
            (
                "model",
                XGBClassifier(
                    n_estimators=180,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=2,
                ),
            ),
        ]
    )
    classifier.fit(train_x, train_extended)

    positive_train = train_y.gt(0)
    regressor_transformer, _, _ = make_preprocessor(features)
    regressor = Pipeline(
        [
            ("features", regressor_transformer),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=180,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=42,
                    n_jobs=2,
                ),
            ),
        ]
    )
    regressor.fit(train_x.loc[positive_train], train_y.loc[positive_train])

    predicted_extended = classifier.predict(test_x)
    positive_prediction = np.maximum(0, regressor.predict(test_x))
    prediction = np.where(predicted_extended == 1, positive_prediction, 0)
    actual_positive = test_y.gt(0).to_numpy()
    predicted_positive = predicted_extended.astype(bool)
    positive_actual_hours = test_y.to_numpy()[actual_positive]
    positive_predicted_hours = positive_prediction[actual_positive]

    report = {
        "target": "repair_duration_hours",
        "definition": "two-stage same-day versus extended repair, then positive duration",
        "rows": {
            "total": int(len(features)),
            "train": int(len(train_x)),
            "test": int(len(test_x)),
            "positive_train": int(positive_train.sum()),
            "positive_test": int(test_extended.sum()),
        },
        "stage_one_extended_repair": {
            "accuracy": round(float(accuracy_score(test_extended, predicted_extended)), 3),
            "balanced_accuracy": round(float(balanced_accuracy_score(test_extended, predicted_extended)), 3),
            "precision": round(float(precision_score(test_extended, predicted_extended, zero_division=0)), 3),
            "recall": round(float(recall_score(test_extended, predicted_extended, zero_division=0)), 3),
        },
        "stage_two_positive_duration": {
            "mae_hours_on_actual_extended": round(float(mean_absolute_error(positive_actual_hours, positive_predicted_hours)), 2),
            "rows_evaluated": int(len(positive_actual_hours)),
        },
        "combined_prediction": {
            "mae_hours": round(float(mean_absolute_error(test_y, prediction)), 2),
            "zero_hour_predictions": int((prediction == 0).sum()),
        },
        "features": {"categorical": categorical, "numeric": numeric},
        "limitations": [
            "Repair duration is an asset-work target, not customer outage duration.",
            "The source territory is APCO, overwhelmingly Virginia.",
            "A same-day record may reflect logging precision rather than zero restoration time.",
        ],
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(classifier, MODEL_DIR / "restoration-extended-classifier.joblib")
    joblib.dump(regressor, MODEL_DIR / "restoration-positive-duration-xgboost.joblib")
    (MODEL_DIR / "two-stage-restoration-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
