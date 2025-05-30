from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "tomi",
    "depends_on_past": False,
    "email": ["tomi@example.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "weather_lstm_train_pipeline",
    default_args=default_args,
    description="Train LSTM weather model with MLflow tracking",
    schedule_interval=None,  # vagy pl. "@daily"
    start_date=datetime(2024, 5, 30),
    catchup=False,
    tags=["ml", "weather", "lstm"],
) as dag:

    train_task = BashOperator(
        task_id="train_lstm_model",
        bash_command="python /app/src/train.py",
        env={
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            # Egyéb környezeti változók, pl. batch_size, epochs, ha kell
        },
        cwd="/app",
    )

    # Ha lesz predikció task, vagy model monitoring
    # predict_task = BashOperator(
    #    ...
    # )
    # train_task >> predict_task
