from datetime import datetime, timedelta
import os
import json
import tempfile

import boto3
import pandas as pd
import mlflow

from airflow import DAG
from airflow.operators.python import PythonOperator


# =========
# Config & Env
# =========
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MINIO_ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")

TEAM = "team_a"
ENV = os.getenv("PLATFORM_ENV", "dev").lower()

# Train / Test input locations in MinIO
S3_BUCKET = os.getenv("TEAM_S3_BUCKET", "team-a-bucket")
S3_TRAIN_KEY = os.getenv("TEAM_S3_TRAIN_KEY", "iris/iris_train.csv")
S3_TEST_KEY = os.getenv("TEAM_S3_TEST_KEY", "iris/iris_test.csv")

# Local temp paths
LOCAL_TRAIN_PATH = "/tmp/iris_train.csv"
LOCAL_TEST_PATH = "/tmp/iris_test.csv"

# Drift report output (MinIO)
DRIFT_RESULTS_BUCKET = os.getenv("TEAM_DRIFT_BUCKET", S3_BUCKET)
DRIFT_RESULTS_PREFIX = os.getenv("TEAM_DRIFT_PREFIX", "iris/drif_results")

RUN_OWNER = os.getenv("RUN_OWNER", "alice_user_team_a")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


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


# =========
# Task 1: Download train & test data from MinIO
# =========
def ingest_train_and_test_from_s3(**context):
    s3 = get_s3_client()

    print(
        f"Downloading train: s3://{S3_BUCKET}/{S3_TRAIN_KEY} "
        f"and test: s3://{S3_BUCKET}/{S3_TEST_KEY} from MinIO {MINIO_ENDPOINT}"
    )

    s3.download_file(S3_BUCKET, S3_TRAIN_KEY, LOCAL_TRAIN_PATH)
    s3.download_file(S3_BUCKET, S3_TEST_KEY, LOCAL_TEST_PATH)

    if not os.path.exists(LOCAL_TRAIN_PATH):
        raise FileNotFoundError(LOCAL_TRAIN_PATH)
    if not os.path.exists(LOCAL_TEST_PATH):
        raise FileNotFoundError(LOCAL_TEST_PATH)

    audit_log(
        "drift_ingest_success",
        {
            "train_dataset_path": LOCAL_TRAIN_PATH,
            "test_dataset_path": LOCAL_TEST_PATH,
            "train_s3_uri": f"s3://{S3_BUCKET}/{S3_TRAIN_KEY}",
            "test_s3_uri": f"s3://{S3_BUCKET}/{S3_TEST_KEY}",
        },
    )

    ti = context["ti"]
    ti.xcom_push(key="train_dataset_path", value=LOCAL_TRAIN_PATH)
    ti.xcom_push(key="test_dataset_path", value=LOCAL_TEST_PATH)


# =========
# Simple numeric drift stats
# =========
def compute_feature_drift(**context):
    ti = context["ti"]
    train_path = ti.xcom_pull(
        key="train_dataset_path", task_ids="ingest_train_and_test"
    )
    test_path = ti.xcom_pull(
        key="test_dataset_path", task_ids="ingest_train_and_test"
    )

    if not train_path or not test_path:
        raise ValueError("Train/test paths not found in XCom.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # For iris we know these numeric features:
    numeric_cols = ["sepal_length", "sepal_width", "pedal_length", "pedal_width"]
    for col in numeric_cols:
        if col not in train_df.columns or col not in test_df.columns:
            raise ValueError(f"Required numeric column missing: {col}")

    stats = []
    for col in numeric_cols:
        train_col = train_df[col].dropna()
        test_col = test_df[col].dropna()

        train_mean = float(train_col.mean())
        test_mean = float(test_col.mean())
        train_std = float(train_col.std(ddof=1))
        test_std = float(test_col.std(ddof=1))
        train_min = float(train_col.min())
        train_max = float(train_col.max())
        test_min = float(test_col.min())
        test_max = float(test_col.max())

        abs_diff = abs(test_mean - train_mean)
        rel_diff = abs_diff / (abs(train_mean) + 1e-8)

        if rel_diff > 0.3:
            drift_level = "high"
        elif rel_diff > 0.1:
            drift_level = "medium"
        else:
            drift_level = "low"

        stats.append(
            {
                "feature": col,
                "train_mean": train_mean,
                "test_mean": test_mean,
                "mean_abs_diff": abs_diff,
                "mean_rel_diff": rel_diff,
                "train_std": train_std,
                "test_std": test_std,
                "train_min": train_min,
                "train_max": train_max,
                "test_min": test_min,
                "test_max": test_max,
                "drift_level": drift_level,
                "train_n": int(len(train_col)),
                "test_n": int(len(test_col)),
            }
        )

    drift_df = pd.DataFrame(stats)
    print("=== Drift stats ===")
    print(drift_df)

    # =========
    # Log to MLflow + push to MinIO
    # =========
    mlflow.set_experiment(f"{TEAM}-{ENV}-iris_drift_monitor")
    s3 = get_s3_client()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    with mlflow.start_run(run_name=f"{TEAM}_{ENV}_iris_drift_run") as run:
        mlflow.log_param("team", TEAM)
        mlflow.log_param("env", ENV)
        mlflow.log_param("run_owner", RUN_OWNER)
        mlflow.log_param("train_s3_uri", f"s3://{S3_BUCKET}/{S3_TRAIN_KEY}")
        mlflow.log_param("test_s3_uri", f"s3://{S3_BUCKET}/{S3_TEST_KEY}")

        # You can also log some aggregate metrics if you like:
        high_drift_count = sum(d["drift_level"] == "high" for d in stats)
        mlflow.log_metric("num_features", len(stats))
        mlflow.log_metric("num_high_drift_features", high_drift_count)

        with tempfile.TemporaryDirectory() as tmpdir:
            drift_filename = "iris_feature_drift.csv"
            local_drift_path = os.path.join(tmpdir, drift_filename)
            drift_df.to_csv(local_drift_path, index=False)

            # 1) Log as MLflow artifact
            mlflow.log_artifact(local_drift_path, artifact_path="drift_report")

            # 2) Upload to MinIO
            object_key = f"{DRIFT_RESULTS_PREFIX}/iris_feature_drift_{ENV}_{ts}.csv"
            print(
                f"Uploading drift report to s3://{DRIFT_RESULTS_BUCKET}/{object_key} "
                f"via MinIO {MINIO_ENDPOINT}"
            )
            s3.upload_file(local_drift_path, DRIFT_RESULTS_BUCKET, object_key)

            drift_s3_uri = f"s3://{DRIFT_RESULTS_BUCKET}/{object_key}"

        audit_log(
            "drift_stats_complete",
            {
                "drift_s3_uri": drift_s3_uri,
                "num_features": len(stats),
                "num_high_drift_features": high_drift_count,
            },
        )

        # Keep for downstream / debugging if needed
        ti.xcom_push(key="drift_s3_uri", value=drift_s3_uri)
        ti.xcom_push(key="mlflow_run_id", value=run.info.run_id)


# =========
# Airflow DAG Definition
# =========
default_args = {
    "owner": TEAM,
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="team_a_dev_iris_drift_pipeline",
    default_args=default_args,
    schedule_interval="@daily",  # periodic drift monitoring
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["iris", TEAM, ENV, "drift"],
) as dag:

    ingest_task = PythonOperator(
        task_id="ingest_train_and_test",
        python_callable=ingest_train_and_test_from_s3,
    )

    drift_task = PythonOperator(
        task_id="compute_feature_drift",
        python_callable=compute_feature_drift,
    )

    ingest_task >> drift_task