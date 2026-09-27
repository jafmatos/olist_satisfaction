from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = ROOT_DIR / "dataset" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "dataset" / "processed"
MODEL_PATH = ROOT_DIR / "models" / "model_v1.pkl"

NUMERICAL_FEATURES = [
    "dias_para_entrega_estimada",
    "dias_para_postagem",
    "dias_para_entrega",
]

CATEGORICAL_FEATURES = [

]

TARGET_FEATURE = "cliente_satisfeito"

TEST_SIZE=0.2
RANDOM_STATE=42