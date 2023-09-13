from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    # for viewing tracking uri (mlflow etc..)
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    # Run the pipeline
    train_pipeline(data_path="C:\Projects\customer_satisfaction_mlops\data\olist_customers_dataset_latest.csv")


    """
    While running this code we will get the data uri then 

    run this command with data uri

    mlflow ui --backdend-store-uri file:C:\Users\nikhi\AppData\Roaming\zenml\local_stores\a397eeab-4a67-4e16-ac68-e0dc84c9c25f\mlruns
    """