from zenml import pipeline

from step.clearn_data import clean_data
from step.evaluation import evaluation
from step.ingest_data import ingest_data
from step.model_train import train_model

@pipeline(enable_cache=True)
def train_pipeline(data_path: str):
    df = ingest_data(data_path)
    x_train, x_test, y_train, y_test = clean_data(df)
    model = train_model(x_train, x_test, y_train, y_test)
    r2_score, rmse = evaluation(model, x_test, y_test)