from datetime import datetime, timedelta
import os
import hashlib
import json
import tempfile

from airflow import DAG
from airflow.operators.python import PythonOperator

import boto3
import pandas as pd
import mlflow
import mlflow.pyfunc
from sklearn.metrics import accuracy_score


# ---------- ENV & CONFIG ----------

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MINIO_ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

TEAM = "team_a"
ENV = os.getenv("PLATFORM_ENV", "dev").lower()

# Use the same bucket as training; just a different key for test data
S3_BUCKET = os.getenv("TEAM_S3_BUCKET", "team-a-bucket")
S3_TEST_KEY = os.getenv("TEAM_S3_TEST_KEY", "iris/iris_test.csv")
LOCAL_TEST_PATH = "/tmp/iris_test.csv"

PREDICTIONS_BUCKET = os.getenv("TEAM_PREDICTIONS_BUCKET", S3_BUCKET)
PREDICTIONS_PREFIX = os.getenv("TEAM_PREDICTIONS_PREFIX", "iris/predictions")

RUN_OWNER = os.getenv("RUN_OWNER", "alice_user_team_a")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_NAME = f"{TEAM}_{ENV}_iris_model"
INFERENCE_EXPERIMENT_NAME = f"{TEAM}-{ENV}-iris_batch_inference"


# ---------- HELPERS ----------

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_log(event_type: str, payload: dict):
    record = {
        "event_type": event_type,
        "env": ENV,
        "team": TEAM,
        "run_owner": RUN_OWNER,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **payload,
    }
    print("[AUDIT]", json.dumps(record))


# ---------- TASK CALLABLES ----------

def ingest_test_from_s3(**context):
    """
    Download test CSV from MinIO / S3 to local disk.
    """
    s3 = get_s3_client()
    print(f"Downloading s3://{S3_BUCKET}/{S3_TEST_KEY} from MinIO {MINIO_ENDPOINT}")
    s3.download_file(S3_BUCKET, S3_TEST_KEY, LOCAL_TEST_PATH)

    if not os.path.exists(LOCAL_TEST_PATH):
        raise FileNotFoundError(LOCAL_TEST_PATH)

    dataset_hash = hash_file(LOCAL_TEST_PATH)
    audit_log(
        "test_ingest_success",
        {"dataset_path": LOCAL_TEST_PATH, "dataset_hash": dataset_hash},
    )

    ti = context["ti"]
    ti.xcom_push(key="test_dataset_path", value=LOCAL_TEST_PATH)
    ti.xcom_push(key="test_dataset_hash", value=dataset_hash)


def validate_test_data(**context):
    """
    Validate that test CSV has required columns.
    """
    ti = context["ti"]
    path = ti.xcom_pull(key="test_dataset_path", task_ids="ingest_test")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(path or "Missing test_dataset_path XCom")

    df = pd.read_csv(path)

    required_cols = ["sepal_length", "sepal_width", "pedal_length", "pedal_width"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Test data is missing required feature columns: {missing}")

    has_label = "class" in df.columns

    audit_log(
        "test_validation_success",
        {
            "dataset_path": path,
            "row_count": len(df),
            "has_label": has_label,
        },
    )

    ti.xcom_push(key="test_has_label", value=bool(has_label))


def run_batch_inference(**context):
    """
    Load latest Staging model from MLflow, run batch inference, and log results.
    """
    ti = context["ti"]

    test_path = ti.xcom_pull(key="test_dataset_path", task_ids="ingest_test")
    test_hash = ti.xcom_pull(key="test_dataset_hash", task_ids="ingest_test")
    has_label = bool(ti.xcom_pull(key="test_has_label", task_ids="validate_test"))

    if not test_path or not os.path.exists(test_path):
        raise FileNotFoundError(test_path or "Missing test_dataset_path XCom")

    df = pd.read_csv(test_path)
    feature_cols = ["sepal_length", "sepal_width", "pedal_length", "pedal_width"]
    X = df[feature_cols]

    # MLflow experiment for inference
    mlflow.set_experiment(INFERENCE_EXPERIMENT_NAME)

    # Load model from registry
    model_uri = f"models:/{MODEL_NAME}/Staging"
    print(f"Loading model from MLflow URI: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)

    y_true = None
    if has_label:
        y_true = df["class"]

    with mlflow.start_run(run_name=f"{TEAM}_{ENV}_iris_batch_inference") as run:
        # Run inference
        preds = model.predict(X)

        # Attach predictions to a copy of the dataframe
        result_df = df.copy()
        result_df["prediction"] = preds

        # If we have labels, compute accuracy
        accuracy = None
        if y_true is not None:
            accuracy = accuracy_score(y_true, preds)
            mlflow.log_metric("batch_accuracy", float(accuracy))

        # Log metadata
        mlflow.log_param("team", TEAM)
        mlflow.log_param("env", ENV)
        mlflow.log_param("run_owner", RUN_OWNER)
        mlflow.log_param("test_dataset_path", test_path)
        mlflow.set_tag("test_dataset_hash", test_hash)
        mlflow.set_tag("has_label", has_label)
        mlflow.set_tag("model_name", MODEL_NAME)
        mlflow.set_tag("model_stage", "Staging")

        # Log predictions as an artifact (CSV)
        # with tempfile.TemporaryDirectory() as tmpdir:
        #     pred_path = os.path.join(tmpdir, "iris_batch_predictions.csv")
        #     result_df.to_csv(pred_path, index=False)
        #     mlflow.log_artifact(pred_path, artifact_path="batch_predictions")

        s3 = get_s3_client()

        # Log predictions as an artifact (CSV) and push to MinIO
        with tempfile.TemporaryDirectory() as tmpdir:
            pred_filename = "iris_batch_predictions.csv"
            pred_path = os.path.join(tmpdir, pred_filename)
            result_df.to_csv(pred_path, index=False)

            # 1) Log to MLflow (same as before)
            mlflow.log_artifact(pred_path, artifact_path="batch_predictions")

            # 2) Push to MinIO (S3)
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            object_key = f"{PREDICTIONS_PREFIX}/iris_batch_predictions_{ENV}_{ts}.csv"

            print(f"Uploading predictions to s3://{PREDICTIONS_BUCKET}/{object_key} via MinIO {MINIO_ENDPOINT}")
            s3.upload_file(pred_path, PREDICTIONS_BUCKET, object_key)
            print(f"Uploading predictions Completed to s3://{PREDICTIONS_BUCKET}/{object_key} via MinIO {MINIO_ENDPOINT}")

            # (Optional) audit log if you’re using the same helper as train DAG
            try:
                audit_log(
                    "batch_inference_predictions_uploaded",
                    {
                        "bucket": PREDICTIONS_BUCKET,
                        "object_key": object_key,
                        "row_count": len(result_df),
                        "env": ENV,
                        "team": TEAM,
                    },
                )
            except NameError:
                # audit_log not defined in this DAG – safe to ignore
                pass

        audit_payload = {
            "mlflow_run_id": run.info.run_id,
            "test_dataset_hash": test_hash,
            "row_count": len(df),
            "has_label": has_label,
            "model_name": MODEL_NAME,
            "model_stage": "Staging",
        }
        if accuracy is not None:
            audit_payload["batch_accuracy"] = float(accuracy)

        audit_log("batch_inference_complete", audit_payload)


# ---------- DAG DEFINITION ----------

default_args = {
    "owner": TEAM,
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="team_a_dev_iris_batch_inference",
    default_args=default_args,
    schedule_interval="@daily",  # run on-demand (e.g., triggered by Jenkins)
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["iris", TEAM, ENV, "batch_inference"],
) as dag:

    ingest_test_task = PythonOperator(
        task_id="ingest_test",
        python_callable=ingest_test_from_s3,
    )

    validate_test_task = PythonOperator(
        task_id="validate_test",
        python_callable=validate_test_data,
    )

    batch_inference_task = PythonOperator(
        task_id="batch_inference",
        python_callable=run_batch_inference,
    )

    ingest_test_task >> validate_test_task >> batch_inference_task