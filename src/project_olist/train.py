from project_olist.config import MODEL_PATH
from project_olist.data import y_train
from project_olist.data import X_train
from project_olist.pipeline import pipeline
import joblib


def train_and_save_model():
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, MODEL_PATH)


if __name__ == "__main__":
    train_and_save_model()