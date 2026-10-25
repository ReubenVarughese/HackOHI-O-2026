"""Train a leakage-aware restoration-duration baseline.

The target is the elapsed time between report and repair timestamps. Fields
known only after completion are excluded from model inputs.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shapefile
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "aep_challenge_hack26-main" / "data" / "raw"
LABOR_REFERENCE = ROOT / "aep_challenge_hack26-main" / "data" / "Install_Labor_HackOHIO.xlsx"
MODEL_DIR = ROOT / "model"


def read_table(zip_name: str, layer_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(DATA_ROOT / zip_name) as archive:
        with archive.open(f"{layer_name}.dbf") as dbf:
            reader = shapefile.Reader(dbf=dbf)
            fields = [field[0] for field in reader.fields[1:]]
            return pd.DataFrame(reader.records(), columns=fields)


def read_damage() -> pd.DataFrame:
    return read_table("Data1.zip", "arcos_prod_damage_09_28_Clip")


def read_geo_points(zip_name: str, layer_name: str) -> pd.DataFrame:
    """Return generalized geometry points and attributes for derived features."""
    with zipfile.ZipFile(DATA_ROOT / zip_name) as archive:
        with archive.open(f"{layer_name}.shp") as shp, archive.open(
            f"{layer_name}.shx"
        ) as shx, archive.open(f"{layer_name}.dbf") as dbf:
            reader = shapefile.Reader(shp=shp, shx=shx, dbf=dbf)
            fields = [field[0] for field in reader.fields[1:]]
            rows = []
            for shape, record in zip(reader.shapes(), reader.records()):
                points = shape.points
                if not points:
                    continue
                x = float(np.mean([point[0] for point in points]))
                y = float(np.mean([point[1] for point in points]))
                rows.append({"geo_longitude": x, "geo_latitude": y, **dict(zip(fields, record))})
            return pd.DataFrame(rows)


def haversine_miles(lat_a, lon_a, lat_b, lon_b):
    lat_a, lon_a, lat_b, lon_b = map(
        np.radians, [np.asarray(lat_a), np.asarray(lon_a), np.asarray(lat_b), np.asarray(lon_b)]
    )
    dlat = lat_b - lat_a
    dlon = lon_b - lon_a
    value = np.sin(dlat / 2) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def add_spatial_features(frame: pd.DataFrame) -> None:
    assets = [
        ("primary", "Data1.zip", "Primary_ExportFeatures"),
        ("secondary", "Data2.zip", "Secondary_ExportFeatures"),
        ("transformer", "Data2.zip", "Transformer_Clip"),
        ("protective", "Data2.zip", "Switch_Clip"),
    ]
    for prefix, zip_name, layer_name in assets:
        points = read_geo_points(zip_name, layer_name)
        points = points.dropna(subset=["geo_latitude", "geo_longitude"])
        if points.empty:
            frame[f"nearest_{prefix}_miles"] = np.nan
            continue
        distances = []
        for lat, lon in zip(frame["latitude"], frame["longitude"]):
            distances.append(
                haversine_miles(
                    lat,
                    lon,
                    points["geo_latitude"].to_numpy(),
                    points["geo_longitude"].to_numpy(),
                ).min()
            )
        frame[f"nearest_{prefix}_miles"] = distances


def add_network_features(frame: pd.DataFrame) -> None:
    def circuit_key(values: pd.Series) -> pd.Series:
        return (
            values.fillna("")
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .fillna("")
            .replace("", "0")
            .astype(int)
        )

    customers = read_table("Data1.zip", "Customer_Clip")
    customer_circuit = circuit_key(customers["circuit_id"])
    frame_circuit = circuit_key(frame["Circuit"])
    customer_counts = customer_circuit.value_counts()
    critical = customers["critical_c"].fillna("").astype(str).str.strip().ne("")
    critical_counts = customer_circuit[critical].value_counts()
    frame["circuit_customer_count"] = frame_circuit.map(customer_counts).fillna(0)
    frame["circuit_critical_customer_count"] = frame_circuit.map(critical_counts).fillna(0)

    device_layers = [
        ("fuse", "Data1.zip", "Fuse_Bank_Clip"),
        ("recloser", "Data2.zip", "Recloser_Bank_Clip"),
        ("switch", "Data2.zip", "Switch_Clip"),
        ("transformer", "Data2.zip", "Transformer_Clip"),
    ]
    for name, zip_name, layer_name in device_layers:
        devices = read_table(zip_name, layer_name)
        circuit_column = "aep_circui" if "aep_circui" in devices else "circuit_id"
        device_circuit = circuit_key(devices[circuit_column])
        counts = device_circuit.value_counts()
        frame[f"circuit_{name}_count"] = frame_circuit.map(counts).fillna(0)

    topology_depth = {}
    with zipfile.ZipFile(DATA_ROOT / "matched_upstream.zip") as archive:
        lines = archive.read("matched_upstream.txt").decode("utf-8", errors="replace").splitlines()
    for line in lines:
        parts = line.split("\t")
        if parts:
            circuit_token = parts[-1].rsplit("*", 1)[-1]
            key = circuit_key(pd.Series([circuit_token])).iloc[0]
            topology_depth[key] = max(topology_depth.get(key, 0), len(parts) - 1)
    frame["topology_depth"] = frame_circuit.map(topology_depth).fillna(0)


def add_historical_features(frame: pd.DataFrame, report_time: pd.Series) -> None:
    """Create prior-only damage rates, avoiding future-record leakage."""
    ordered = pd.DataFrame(
        {
            "time": report_time.to_numpy(),
            "circuit": frame["Circuit"].fillna("").astype(str).to_numpy(),
            "asset": frame["Asset_Type"].fillna("").astype(str).to_numpy(),
            "event": frame["Event"].fillna("").astype(str).to_numpy(),
        },
        index=frame.index,
    ).sort_values("time")
    for name, group_columns in {
        "circuit_damage_history": ["circuit"],
        "asset_damage_history": ["asset"],
        "event_damage_history": ["event"],
        "circuit_asset_damage_history": ["circuit", "asset"],
    }.items():
        counts = ordered.groupby(group_columns, dropna=False).cumcount()
        frame[name] = counts.reindex(frame.index).fillna(0).astype(int)


def labor_reference() -> tuple[float, float, float]:
    """Read anonymized installation, removal, and indirect labor references."""
    cnc = pd.read_excel(LABOR_REFERENCE, sheet_name="CNC", usecols=[0, 1, 2])
    install = pd.to_numeric(cnc["INSTALL_HOURS"], errors="coerce").dropna()
    removal = pd.to_numeric(cnc["REMOVE_HOURS"], errors="coerce").dropna()
    multiplier_table = pd.read_excel(
        LABOR_REFERENCE, sheet_name="Indirect Labor Multiplier", header=None
    )
    multiplier_values = pd.to_numeric(multiplier_table.stack(), errors="coerce").dropna()
    multiplier = multiplier_values.iloc[0]
    return float(install.mean()), float(removal.mean()), float(multiplier)


def add_labor_features(frame: pd.DataFrame) -> None:
    install_hours, removal_hours, indirect_multiplier = labor_reference()
    frame["labor_reference_install_hours"] = install_hours
    frame["labor_reference_removal_hours"] = removal_hours
    frame["indirect_labor_multiplier"] = indirect_multiplier
    frame["access_labor_adder_hours"] = frame["has_access_issue"] * 0.5
    frame["estimated_labor_hours"] = (
        (install_hours + removal_hours) * indirect_multiplier + frame["access_labor_adder_hours"]
    )


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = ["Asset_Type", "Inform_Wor", "Circuit", "Substation", "Event", "Incident_W"]
    numeric = [
        "has_access_issue",
        "has_special_equipment",
        "media_count",
        "latitude",
        "longitude",
        "report_month",
        "report_hour",
        "circuit_customer_count",
        "circuit_critical_customer_count",
        "circuit_fuse_count",
        "circuit_recloser_count",
        "circuit_switch_count",
        "circuit_transformer_count",
        "topology_depth",
        "nearest_primary_miles",
        "nearest_secondary_miles",
        "nearest_transformer_miles",
        "nearest_protective_miles",
        "circuit_damage_history",
        "asset_damage_history",
        "event_damage_history",
        "circuit_asset_damage_history",
        "labor_reference_install_hours",
        "labor_reference_removal_hours",
        "indirect_labor_multiplier",
        "access_labor_adder_hours",
        "estimated_labor_hours",
    ]
    return categorical, numeric


def build_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    frame = raw.copy()
    report_time = pd.to_datetime(frame["Report_Tim"], errors="coerce")
    repair_time = pd.to_datetime(frame["Repaired_D"], errors="coerce")
    duration = (repair_time - report_time).dt.total_seconds() / 3600
    frame["report_month"] = report_time.dt.month
    frame["report_hour"] = report_time.dt.hour
    frame["has_access_issue"] = frame["Access_Iss"].fillna("").astype(str).str.strip().ne("").astype(int)
    frame["has_special_equipment"] = frame["Special_Eq"].fillna("").astype(str).str.strip().ne("").astype(int)
    frame["media_count"] = pd.to_numeric(frame["Media"], errors="coerce").fillna(0)
    frame["latitude"] = pd.to_numeric(frame["Latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["Longitude"], errors="coerce")
    frame["Event"] = frame["Event"].fillna("").astype(str).str.strip()
    frame["Incident_W"] = frame["Incident_W"].fillna("").astype(str).str.strip()
    add_network_features(frame)
    add_spatial_features(frame)
    add_historical_features(frame, report_time)
    add_labor_features(frame)
    _, numeric = feature_columns(frame)
    selected = [
        "Asset_Type",
        "Inform_Wor",
        "Circuit",
        "Substation",
        "Event",
        "Incident_W",
        "has_access_issue",
        "has_special_equipment",
        "media_count",
        "latitude",
        "longitude",
        "report_month",
        "report_hour",
        "labor_reference_install_hours",
        "labor_reference_removal_hours",
        "indirect_labor_multiplier",
        "access_labor_adder_hours",
        "estimated_labor_hours",
    ]
    selected.extend([column for column in numeric if column not in selected])
    valid = duration.notna() & duration.ge(0) & frame[selected].notna().all(axis=1)
    return frame.loc[valid, selected], duration.loc[valid], report_time.loc[valid]


def main() -> None:
    features, target, report_time = build_features(read_damage())
    source_rows = len(features)
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

    categorical, numeric = feature_columns(features)
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("features", preprocessor),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=250,
                    max_depth=4,
                    learning_rate=0.04,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    random_state=42,
                    n_jobs=2,
                ),
            ),
        ]
    )
    pipeline.fit(train_x, train_y)
    prediction = np.maximum(0, pipeline.predict(test_x))
    residual = test_y.to_numpy() - prediction
    residual_low = float(np.quantile(residual, 0.1))
    residual_high = float(np.quantile(residual, 0.9))
    report = {
        "target": "repair_duration_hours",
        "rows": {
            "source_valid": int(source_rows),
            "excluded_over_30_days": int(source_rows - len(features)),
            "total": int(len(features)),
            "train": int(len(train_x)),
            "test": int(len(test_x)),
        },
        "metrics": {
            "mae_hours": round(float(mean_absolute_error(test_y, prediction)), 2),
            "rmse_hours": round(float(mean_squared_error(test_y, prediction) ** 0.5), 2),
            "test_median_hours": round(float(test_y.median()), 2),
        },
        "uncertainty": {
            "method": "held-out residual quantiles",
            "p10_residual_hours": round(residual_low, 2),
            "p90_residual_hours": round(residual_high, 2),
            "note": "Use as a baseline interval only; it is not calibrated for customer outage probability.",
        },
        "features": {"categorical": categorical, "numeric": numeric},
        "limitations": [
            "The target is asset repair duration, not customer outage duration.",
            "The source territory is APCO, overwhelmingly Virginia.",
            "No customer-level outage/no-outage label is available.",
            "Durations over 30 days were excluded as likely placeholder or data-quality values.",
        ],
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "restoration-duration-xgboost.joblib")
    (MODEL_DIR / "restoration-model-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
