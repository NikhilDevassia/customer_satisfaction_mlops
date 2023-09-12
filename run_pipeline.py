from pipelines.training_pipeline import train_pipeline

if __name__ == "__main__":
    # Run the pipeline
    train_pipeline(data_path=r"C:\Projects\customer_satisfaction_mlops\data\olist_customers_dataset_latest.csv")