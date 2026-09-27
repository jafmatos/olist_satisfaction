from project_olist.config import RANDOM_STATE
from sklearn.ensemble import RandomForestClassifier
from project_olist.config import NUMERICAL_FEATURES
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FunctionTransformer
from pandas import DataFrame

def feature_engineering(dataframe: DataFrame):
    dataframe = dataframe.copy()

    dataframe["entrega_atrasada"] = dataframe.apply(
        lambda row: (
            1 if row["dias_para_entrega"] > row["dias_para_entrega_estimada"] 
            else 0
        ),
        axis=1
    )

    dataframe["atraso_entrega_dias"] = dataframe.apply(
        lambda row: (
            0 if row["entrega_atrasada"] == 0 
            else row["dias_para_entrega"] - row["dias_para_entrega_estimada"]
        ),
        axis=1
    )  

    return dataframe 

feature_transformer = FunctionTransformer(feature_engineering)

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

column_transformer = ColumnTransformer(
    transformers=[
        ("feature_engineering", feature_transformer, NUMERICAL_FEATURES),
        ("numerical_pipeline", numerical_pipeline, NUMERICAL_FEATURES)
    ],
    remainder="drop"
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", column_transformer),
        ("model", RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE))
    ]
)