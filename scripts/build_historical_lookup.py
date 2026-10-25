"""Build an aggregate, privacy-safe lookup for live prediction requests."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import shapefile

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "aep_challenge_hack26-main" / "data" / "raw"
OUTPUT = ROOT / "model" / "historical-lookup.json"


def read_damage() -> pd.DataFrame:
    with zipfile.ZipFile(DATA_ROOT / "Data1.zip") as archive:
        with archive.open("arcos_prod_damage_09_28_Clip.dbf") as dbf:
            reader = shapefile.Reader(dbf=dbf)
            names = [field[0] for field in reader.fields[1:]]
            return pd.DataFrame(reader.records(), columns=names)


def read_layer(zip_name: str, layer_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(DATA_ROOT / zip_name) as archive:
        with archive.open(f"{layer_name}.dbf") as dbf:
            reader = shapefile.Reader(dbf=dbf)
            names = [field[0] for field in reader.fields[1:]]
            return pd.DataFrame(reader.records(), columns=names)


def circuit_key(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.extract(r"(\d+)", expand=False).fillna("unknown")


def main() -> None:
    damage = read_damage()
    damage["circuit_key"] = circuit_key(damage["Circuit"])
    damage["asset_key"] = damage["Asset_Type"].fillna("Unknown").astype(str).str.strip()
    damage["latitude"] = pd.to_numeric(damage["Latitude"], errors="coerce")
    damage["longitude"] = pd.to_numeric(damage["Longitude"], errors="coerce")
    damage["duration_hours"] = (
        pd.to_datetime(damage["Repaired_D"], errors="coerce")
        - pd.to_datetime(damage["Report_Tim"], errors="coerce")
    ).dt.total_seconds() / 3600
    damage = damage.loc[damage["duration_hours"].ge(0) & damage["duration_hours"].le(720)].copy()

    circuit = damage.groupby("circuit_key").agg(
        damage_count=("circuit_key", "size"),
        median_duration_hours=("duration_hours", "median"),
        latitude=("latitude", "median"),
        longitude=("longitude", "median"),
    )
    asset = damage.groupby(["circuit_key", "asset_key"]).agg(
        damage_count=("asset_key", "size"),
        median_duration_hours=("duration_hours", "median"),
    )
    network_counts: dict[str, dict[str, int]] = {}

    def add_counts(zip_name: str, layer_name: str, column: str, output_key: str) -> None:
        table = read_layer(zip_name, layer_name)
        source_column = column if column in table else "aep_circui"
        if source_column not in table:
            return
        counts = circuit_key(table[source_column]).value_counts()
        for key, count in counts.items():
            network_counts.setdefault(str(key), {})[output_key] = int(count)

    add_counts("Data1.zip", "Customer_Clip", "circuit_id", "customer_count")
    add_counts("Data1.zip", "Fuse_Bank_Clip", "aep_circui", "fuse_count")
    add_counts("Data2.zip", "Recloser_Bank_Clip", "aep_circui", "recloser_count")
    add_counts("Data2.zip", "Switch_Clip", "aep_circui", "switch_count")
    add_counts("Data2.zip", "Transformer_Clip", "aep_circui", "transformer_count")

    asset_lookup = {
        f"{circuit}:{asset_type}": {
            "damage_count": int(row.damage_count),
            "median_duration_hours": round(float(row.median_duration_hours), 2),
        }
        for (circuit, asset_type), row in asset.iterrows()
    }
    output = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "source": "Aggregated APCO damage records; no premise or customer identifiers",
        "circuits": {
            str(key): {
                "damage_count": int(row.damage_count),
                "median_duration_hours": round(float(row.median_duration_hours), 2),
                "latitude": round(float(row.latitude), 6),
                "longitude": round(float(row.longitude), 6),
                **network_counts.get(str(key), {}),
            }
            for key, row in circuit.iterrows()
            if pd.notna(row.latitude) and pd.notna(row.longitude)
        },
        "circuit_assets": asset_lookup,
    }
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote {len(output['circuits'])} circuit aggregates and {len(asset_lookup)} asset aggregates")


if __name__ == "__main__":
    main()
