import os
import json
import hashlib
import random
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# ----------------- USER CONFIG -----------------
OUT_DIR = Path(r"C:\AI-Financial-Predictor\model\data")   # where to save
DISTRICT_LISTS_DIR = Path(
    r"C:\AI-Financial-Predictor\model\district_lists")  # optional
# default synthetic district count per state (if no list provided)
DISTRICTS_PER_STATE = 30
# change to control dataset size (rows per district)
ROWS_PER_DISTRICT = 200
RANDOM_SEED = 42
# ------------------------------------------------

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Indian states + UTs list (canonical names)
states_and_uts = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal", "Andaman & Nicobar", "Chandigarh", "Dadra & Nagar Haveli and Daman & Diu",
    "Delhi", "Jammu & Kashmir", "Ladakh", "Lakshadweep", "Puducherry"
]

# Core scheme list (same as earlier)
schemes = [
    "PMAY-R", "PMAY-U", "PM-KISAN", "PMKSY", "PMFBY", "MGNREGA", "ICDS-Nutrition",
    "SamagraShiksha", "SkillMission", "PMJDY", "PPF", "Livelihoods"
]


def bounded_normal(mu, sigma, low, high, size=None):
    vals = np.random.normal(mu, sigma, size=size)
    vals = np.clip(vals, low, high)
    return vals if size is not None else float(vals)


def sha256_of_file(path):
    sha256 = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_district_list_for_state(state_name):
    """
    If a district-list file exists for this state under DISTRICT_LISTS_DIR,
    return the list read from file (one district name per line).
    File expected: <state_lower_no_spaces>_districts.csv
    """
    safe = state_name.lower().replace(" ", "_").replace("&", "and").replace(".", "")
    candidate = DISTRICT_LISTS_DIR / f"{safe}_districts.csv"
    if candidate.exists():
        try:
            df = pd.read_csv(candidate, header=None)
            names = [str(x).strip()
                     for x in df.iloc[:, 0].tolist() if str(x).strip()]
            if len(names) >= 1:
                return names
        except Exception:
            pass
    return None


def generate_state_dataset(state_name, districts, rows_per_district):
    rows = []
    for district in districts:
        for t in range(rows_per_district):
            taluk = f"{district.split()[0]}-Taluk-{t+1}"
            is_urban = ("Urban" in district) or ("Bengaluru" in district) or (
                "Delhi" in state_name) or ("Mumbai" in district) or ("City" in district)
            # Demographics
            pop_density = round(bounded_normal(
                3000 if is_urban else 400, 900, 20, 9000), 1)
            literacy_rate = round(bounded_normal(
                0.87 if is_urban else 0.77, 0.075, 0.5, 0.99), 3)
            female_ratio = round(bounded_normal(
                0.97 if is_urban else 0.99, 0.04, 0.85, 1.12), 3)
            agri_dependence = round(bounded_normal(
                0.18 if is_urban else 0.56, 0.18, 0.01, 0.98), 3)
            main_crop = random.choice(
                ["paddy", "coconut", "arecanut", "cashew", "areca", "millet", "horticulture", "services"])
            avg_income = int(bounded_normal(
                90000 if is_urban else 42000, 26000, 4000, 400000))
            employment_rate = round(bounded_normal(
                0.71 if is_urban else 0.63, 0.11, 0.25, 0.99), 3)
            bank_branches = int(bounded_normal(
                1200 if is_urban else 18, 600, 0, 7000))
            financial_awareness_index = round(bounded_normal(
                0.78 if is_urban else 0.43, 0.16, 0.02, 1.0), 3)
            avg_savings_rate = round(bounded_normal(
                0.25 if is_urban else 0.12, 0.12, 0.0, 1.0), 3)

            # Education
            primary_schools = int(bounded_normal(
                90 if is_urban else 45, 40, 2, 500))
            high_schools = int(bounded_normal(
                18 if is_urban else 8, 9, 0, 250))
            total_enrollment = int(bounded_normal(
                16000 if is_urban else 4200, 9000, 50, 350000))
            st_share = round(bounded_normal(0.05, 0.03, 0.0, 0.45), 3)
            sc_share = round(bounded_normal(0.07, 0.04, 0.0, 0.45), 3)

            # Health
            hospitals = int(bounded_normal(14 if is_urban else 3, 7, 0, 300))
            doctors_per_10k = round(bounded_normal(
                6.5 if is_urban else 1.4, 3.1, 0.0, 60.0), 2)
            beds_per_10k = round(bounded_normal(
                22.0 if is_urban else 4.5, 13.0, 0.0, 350.0), 2)
            immunization_pct = round(bounded_normal(0.92, 0.06, 0.45, 1.0), 3)
            malnutrition_pct = round(bounded_normal(
                0.075 if is_urban else 0.175, 0.07, 0.01, 0.7), 3)

            # Housing & infra
            houses_pmay = int(bounded_normal(
                2200 if is_urban else 320, 1600, 0, 80000))
            electricity_pct = round(bounded_normal(
                0.995 if is_urban else 0.83, 0.07, 0.3, 1.0), 3)
            tap_water_pct = round(bounded_normal(
                0.93 if is_urban else 0.67, 0.13, 0.03, 1.0), 3)
            lpg_pct = round(bounded_normal(
                0.88 if is_urban else 0.60, 0.17, 0.01, 1.0), 3)

            # Agriculture
            net_sown_area = int(bounded_normal(
                13000 if main_crop != "services" else 2200, 11000, 0, 300000))
            irrigated_area = int(bounded_normal(
                net_sown_area*0.46 if main_crop != "services" else net_sown_area*0.6, 9000, 0, net_sown_area))
            crop_productivity_rice = int(bounded_normal(
                3050 if main_crop == "paddy" else 1850, 900, 150, 15000))
            fertilizer_consumption = round(bounded_normal(
                58 if main_crop == "paddy" else 22, 42, 0, 900), 2)
            livestock_units = int(bounded_normal(
                3200 if not is_urban else 240, 1900, 0, 30000))

            # Banking & welfare
            pmjdy_accounts = int(bounded_normal(
                110000 if is_urban else 16000, 70000, 0, 2000000))
            pmkisan_beneficiaries = int(bounded_normal(
                2200 if main_crop != "services" else 30, 6000, 0, 400000))
            mg_nrega_days = round(bounded_normal(
                18 if is_urban else 54, 26, 0, 300), 1)

            # Women & child
            anganwadi_centres = int(bounded_normal(
                125 if is_urban else 50, 85, 0, 3000))
            shg_count = int(bounded_normal(
                480 if not is_urban else 140, 420, 0, 12000))
            female_literacy = round(
                max(0.3, min(1.0, literacy_rate - bounded_normal(0.02, 0.03, 0, 0.1))), 3)

            # Composite indices
            education_index = round(
                (literacy_rate*0.5 + (total_enrollment/300000)*0.3 + (high_schools/250)*0.2), 3)
            health_index = round((min(1.0, doctors_per_10k/12)*0.45 + min(
                1.0, beds_per_10k/60)*0.35 + (1.0 - malnutrition_pct)*0.2), 3)
            livelihood_index = round(
                (employment_rate*0.5 + (avg_savings_rate)*0.2 + (bank_branches/2000)*0.3), 3)
            agriculture_index = round((min(1.0, irrigated_area/(net_sown_area+1))*0.5 + min(
                1.0, crop_productivity_rice/5500)*0.35 + (1.0 - fertilizer_consumption/300)*0.15), 3)

            # Heuristic recommended scheme
            rec = "PPF"
            if tap_water_pct < 0.6 or electricity_pct < 0.7 or houses_pmay < 120:
                rec = "PMAY-R" if not is_urban else "PMAY-U"
            elif agri_dependence > 0.48 and crop_productivity_rice > 0:
                rec = random.choices(
                    ["PMKSY", "PM-KISAN", "PMFBY"], weights=[0.45, 0.4, 0.15])[0]
            elif malnutrition_pct > 0.16 or anganwadi_centres < 18:
                rec = "ICDS-Nutrition"
            elif literacy_rate < 0.72 or education_index < 0.38:
                rec = "SamagraShiksha"
            elif livelihood_index < 0.44:
                rec = "MGNREGA"
            else:
                rec = random.choices(["SkillMission", "PMJDY", "Livelihoods", "PPF"], weights=[
                                     0.28, 0.22, 0.33, 0.17])[0]

            rows.append({
                "taluk": taluk,
                "district": district,
                "population_density": pop_density,
                "literacy_rate": literacy_rate,
                "female_ratio": female_ratio,
                "agri_dependence": agri_dependence,
                "main_crop": main_crop,
                "avg_income": avg_income,
                "employment_rate": employment_rate,
                "bank_branches": bank_branches,
                "financial_awareness_index": financial_awareness_index,
                "avg_savings_rate": avg_savings_rate,
                "primary_schools": primary_schools,
                "high_schools": high_schools,
                "total_enrollment": total_enrollment,
                "st_share": st_share,
                "sc_share": sc_share,
                "hospitals": hospitals,
                "doctors_per_10k": doctors_per_10k,
                "beds_per_10k": beds_per_10k,
                "immunization_pct": immunization_pct,
                "malnutrition_pct": malnutrition_pct,
                "houses_pmay": houses_pmay,
                "electricity_pct": electricity_pct,
                "tap_water_pct": tap_water_pct,
                "lpg_pct": lpg_pct,
                "net_sown_area": net_sown_area,
                "irrigated_area": irrigated_area,
                "crop_productivity_rice": crop_productivity_rice,
                "fertilizer_consumption": fertilizer_consumption,
                "livestock_units": livestock_units,
                "pmjdy_accounts": pmjdy_accounts,
                "pmkisan_beneficiaries": pmkisan_beneficiaries,
                "mg_nrega_days": mg_nrega_days,
                "anganwadi_centres": anganwadi_centres,
                "shg_count": shg_count,
                "female_literacy": female_literacy,
                "education_index": education_index,
                "health_index": health_index,
                "livelihood_index": livelihood_index,
                "agriculture_index": agriculture_index,
                "recommended_scheme": rec
            })

    # build DataFrame and save
    df = pd.DataFrame(rows)
    # engineered fields
    df["beds_per_1000"] = (df["beds_per_10k"] / 10).round(2)
    df["doctors_per_1000"] = (df["doctors_per_10k"] / 10).round(3)
    df["irrigation_pct"] = (df["irrigated_area"] /
                            (df["net_sown_area"] + 1)).round(3)
    df["pmjdy_per_1000_pop"] = (
        df["pmjdy_accounts"] / (df["population_density"] + 1) * 1000).round(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_state = state_name.lower().replace(
        " ", "_").replace("&", "and").replace(".", "")
    csv_path = OUT_DIR / f"{safe_state}_dataset.csv"
    jsonl_path = OUT_DIR / f"{safe_state}_dataset.jsonl"
    df.to_csv(csv_path, index=False)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            f.write(json.dumps(r.dropna().to_dict(), ensure_ascii=False) + "\n")

    csv_sha = sha256_of_file(csv_path)
    jsonl_sha = sha256_of_file(jsonl_path)
    print(f"[{datetime.now().isoformat()}] Saved {state_name}: rows={len(df)} cols={len(df.columns)}")
    print(" CSV:", csv_path)
    print(" JSONL:", jsonl_path)
    print(" CSV SHA256:", csv_sha)
    print(" JSONL SHA256:", jsonl_sha)
    # clear rows for next state
    rows.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--districts-per-state", type=int,
                        default=DISTRICTS_PER_STATE)
    parser.add_argument("--rows-per-district", type=int,
                        default=ROWS_PER_DISTRICT)
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = parser.parse_args()

    outdir = Path(args.out_dir)
    print("Generating datasets into:", outdir.resolve())
    for state in states_and_uts:
        # try to load district list file
        dlist = load_district_list_for_state(state)
        if dlist:
            districts = dlist
        else:
            # create synthetic district names
            districts = [
                f"{state.split()[0]}_District_{i+1}" for i in range(args.districts_per_state)]
        generate_state_dataset(state, districts, args.rows_per_district)


if __name__ == "__main__":
    main()
