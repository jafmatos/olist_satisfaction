from project_olist.config import RANDOM_STATE
from project_olist.config import TEST_SIZE
from project_olist.config import TARGET_FEATURE
from project_olist.config import CATEGORICAL_FEATURES
from project_olist.config import NUMERICAL_FEATURES
from project_olist.config import PROCESSED_DATA_DIR
import pandas
from sklearn.model_selection import train_test_split

dataframe = pandas.read_csv(PROCESSED_DATA_DIR / "pedidos_com_avaliacao.csv")
X = dataframe[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
y = dataframe[TARGET_FEATURE]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
