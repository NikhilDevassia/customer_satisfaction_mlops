from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    # for viewing tracking uri (mlflow etc..)
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    # Run the pipeline
    train_pipeline(data_path="/home/nikhil/Projects/customer_satisfaction_mlops/data/olist_customers_dataset_latest.csv")

