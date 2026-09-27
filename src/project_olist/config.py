from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT_DIR / "dataset" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "dataset" / "processed"