"""Audit the supplied damage data before training a prediction model.

This intentionally reports label availability and feature coverage first.
It does not claim that the source records are customer-level outage labels.
"""

from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
import shapefile


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "aep_challenge_hack26-main" / "data" / "raw"
OUTPUT = ROOT / "model" / "training-data-audit.json"


def read_layer(zip_path: Path, layer_name: str) -> pd.DataFrame:
    """Read a DBF table from a zipped shapefile without extracting raw data."""
    with zipfile.ZipFile(zip_path) as archive:
        dbf_name = f"{layer_name}.dbf"
        with archive.open(dbf_name) as dbf:
            reader = shapefile.Reader(dbf=dbf)
            field_names = [field[0] for field in reader.fields[1:]]
            return pd.DataFrame(reader.records(), columns=field_names)


def non_empty_counts(frame: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    return {
        column: int(frame[column].notna().sum() - (frame[column] == "").sum())
        for column in columns
        if column in frame.columns
    }


def main() -> None:
    damage = read_layer(DATA_ROOT / "Data1.zip", "arcos_prod_damage_09_28_Clip")
    customers = read_layer(DATA_ROOT / "Data1.zip", "Customer_Clip")

    damage["repaired_flag"] = damage["Repaired"].astype(str).str.strip().str.lower().eq("yes")
    damage["has_report_time"] = damage["Report_Tim"].notna()
    damage["has_repair_time"] = damage["Repaired_D"].notna()

    repair_duration = pd.to_datetime(damage["Repaired_D"], errors="coerce") - pd.to_datetime(
        damage["Report_Tim"], errors="coerce"
    )
    valid_duration = repair_duration[repair_duration.dt.total_seconds() >= 0]

    report = {
        "source": {
            "damage_layer": "arcos_prod_damage_09_28_Clip",
            "customer_layer": "Customer_Clip",
            "region_values": sorted(damage["Region"].dropna().astype(str).unique().tolist()),
            "customer_state_counts": {
                str(key): int(value)
                for key, value in customers["state"].fillna("").value_counts().to_dict().items()
            },
        },
        "row_counts": {
            "damage_records": int(len(damage)),
            "customer_records": int(len(customers)),
        },
        "label_coverage": {
            "repaired_yes": int(damage["repaired_flag"].sum()),
            "repaired_no": int((~damage["repaired_flag"]).sum()),
            "report_time_present": int(damage["has_report_time"].sum()),
            "repair_time_present": int(damage["has_repair_time"].sum()),
            "non_negative_repair_durations": int(len(valid_duration)),
            "repair_duration_hours_median": (
                round(float(valid_duration.dt.total_seconds().median() / 3600), 2)
                if len(valid_duration)
                else None
            ),
        },
        "feature_coverage": non_empty_counts(
            damage,
            [
                "Asset_Type",
                "Inform_Wor",
                "Circuit",
                "Substation",
                "Access_Iss",
                "Special_Eq",
                "Media",
                "Longitude",
                "Latitude",
            ],
        ),
        "damage_asset_types": {
            str(key): int(value)
            for key, value in damage["Asset_Type"].fillna("").value_counts().to_dict().items()
        },
        "damage_descriptions": {
            str(key): int(value)
            for key, value in damage["Inform_Wor"].fillna("").value_counts().to_dict().items()
        },
        "assessment": {
            "has_customer_outage_label": False,
            "has_reliable_duration_label": bool(len(valid_duration) >= 30),
            "recommendation": (
                "Use repair duration as a cautious operational baseline; do not call it "
                "customer outage probability until outage/customer labels are established."
            ),
        },
    }

    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
