

import os
import re
import sys
from pathlib import Path

try:
    import geopandas as gpd
except Exception as e:
    print("ERROR: geopandas not available. Install via conda: conda install -c conda-forge geopandas fiona shapely rtree")
    raise

PROJECT_ROOT = Path(r"C:\AI-Financial-Predictor")
INPUT = PROJECT_ROOT / "india_adm2.geojson"   # file you downloaded
OUT_DIRS = [
    PROJECT_ROOT / "frontend" / "src" / "map" / "layers" / "split_by_state",
    PROJECT_ROOT / "frontend" / "public" / "map-layers" / "split_by_state",
]

COMMON_STATE_PROPS = [
    "shapeName", "shapeGroup", "ADM1_NAME", "admin1Name", "state", "st_name",
    "NAME_1", "NAME1", "NAME_0", "state_name", "region", "st_nm", "ADM1"
]


def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\s-]", "", name).strip()
    name = re.sub(r"[-\s]+", "_", name)
    return name[:120] or "unknown"


def main():
    if not INPUT.exists():
        print(f"ERROR: Input file not found at: {INPUT}")
        sys.exit(1)

    # ensure output dirs exist
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)

    print("Reading input GeoJSON (this may take a moment)...")
    gdf = gpd.read_file(str(INPUT))
    print(f"Loaded {len(gdf)} features and {len(gdf.columns)} properties.")

    # try to detect state property
    state_prop = None
    for p in COMMON_STATE_PROPS:
        if p in gdf.columns:
            state_prop = p
            print(f"Using common property '{p}' as state/ADM1 field.")
            break

    if state_prop is None:
        # fallback: choose a column with 5 < unique < 50
        for col in gdf.columns:
            try:
                uniques = gdf[col].nunique()
            except Exception:
                continue
            if 5 < uniques < 60:
                state_prop = col
                print(
                    f"Auto-detected state property '{col}' (unique values: {uniques}).")
                break

    if state_prop is None:
        print("ERROR: Could not find a suitable state/ADM1 property. Columns available:")
        print(list(gdf.columns))
        sys.exit(1)

    grouped = gdf.groupby(gdf[state_prop])
    print(f"Splitting into {len(grouped)} groups by '{state_prop}' ...")

    for state_name, group in grouped:
        safe_name = safe_filename(str(state_name))
        filename = f"{safe_name}_districts.geojson"
        for out_dir in OUT_DIRS:
            out_path = out_dir / filename
            # Write GeoJSON (driver auto-detected)
            group.to_file(str(out_path), driver="GeoJSON")
            print(f"Wrote {len(group)} features -> {out_path}")

    print("All done. Per-state files created in:")
    for d in OUT_DIRS:
        print("  -", d)


if __name__ == "__main__":
    main()
