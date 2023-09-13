from zenml import pipeline

from steps.ingest_data import ingest_data
from steps.clearn_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluation


@pipeline(enable_cache=True)
def train_pipeline(data_path: str):
    df = ingest_data(data_path)
    x_train, x_test, y_train, y_test = clean_data(df)
    model = train_model(x_train, x_test, y_train, y_test)
    r2_score, rmse = evaluation(model, x_test, y_test)