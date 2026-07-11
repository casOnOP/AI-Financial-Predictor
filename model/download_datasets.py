import os
import subprocess

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# ✅ Updated list of public Kaggle datasets
datasets = {
    "demographic": "thedevastator/india-census-data",  # Census alternative
    "agriculture": "rajanand/crop-production-in-india",
    "financial": "iabhishekofficial/india-bank-branch-dataset",
    "economic": "vivek468/india-gdp-per-capita",
}

for name, dataset in datasets.items():
    print(f"⬇️ Downloading {name} dataset from Kaggle...")
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d",
                       dataset, "-p", "data", "--unzip"], check=True)
        print(f"✅ {name.capitalize()} dataset downloaded successfully.")
    except Exception as e:
        print(f"⚠️ Failed to download {name}: {e}")

print("✅ All available datasets processed.")
