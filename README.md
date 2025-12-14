# Iris MLOps Pipelines

This repository contains a small collection of Airflow DAGs that demonstrate an end‑to‑end MLOps workflow for the classic Iris classification problem. The pipelines cover model training and registration, scheduled batch inference, and simple dataset drift monitoring while integrating with MLflow for experiment tracking and MinIO (S3 API) for storage.

## Repository Layout

| Path | Description |
| --- | --- |
| `airflow/dags/iris_pipeline.py` | Daily training workflow that ingests Iris data from MinIO, validates it, trains a logistic regression model, and registers approved runs in MLflow Model Registry. |
| `airflow/dags/iris_batch_inferencing.py` | Batch scoring workflow that loads the latest Staging model from MLflow, generates predictions for a held‑out CSV, and writes the results back to MinIO and MLflow artifacts. |
| `airflow/dags/iris_drift_pipeline.py` | Drift monitor that compares simple statistics between the training and inference datasets and stores a drift report in MLflow and MinIO. |
| `project.json` | Lightweight metadata describing the project name, default team, Airflow DAG directory, and tags. |

## Dependencies

The DAGs expect the following Python packages to be available in your Airflow environment:

- `apache-airflow` (with the PythonOperator)
- `pandas`, `scikit-learn`
- `mlflow`
- `boto3`

They also assume access to a running MLflow Tracking server + Model Registry and an S3‑compatible storage backend such as MinIO.

## Configuration
| Variable | Purpose |
| --- | --- |
| `MLFLOW_TRACKING_URI` | URL of the MLflow Tracking server (default `http://mlflow:5000`). |
| `MLFLOW_S3_ENDPOINT_URL` | Endpoint for MinIO/S3 used by MLflow (default `http://minio:9000`). |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | Credentials for MinIO/S3. |
| `PLATFORM_ENV` | Logical environment tag that becomes part of experiment/DAG names (default `dev`). |
| `TEAM_S3_BUCKET`, `TEAM_S3_KEY`, `TEAM_S3_TRAIN_KEY`, `TEAM_S3_TEST_KEY` | Locations of the training/testing CSVs. |
| `TEAM_PREDICTIONS_BUCKET`, `TEAM_PREDICTIONS_PREFIX` | Where batch inference uploads the scored CSV. |
| `TEAM_DRIFT_BUCKET`, `TEAM_DRIFT_PREFIX` | Where the drift pipeline stores reports. |
| `ACCURACY_THRESHOLD` | Minimum validation accuracy required before a model is registered (default `0.1`). |
| `RUN_OWNER` | Tag applied to audit logs and MLflow runs. |

All DAGs also honor `TEAM` (default `team_a`) for namespacing and include lightweight audit logging for traceability.

## Airflow DAGs

- **`team_a_dev_iris_pipeline`**  
  Ingests the training dataset from MinIO, validates schema/emptiness, trains a logistic regression classifier, and registers the model in MLflow. Approved runs are promoted to the `Staging` stage and have the `staging` alias applied while older versions are archived.

- **`team_a_dev_iris_batch_inference`**  
  Fetches the latest inference dataset, optionally scores ground‑truth labels, logs prediction accuracy, and writes the predictions both as an MLflow artifact and to MinIO under a timestamped key for downstream consumers.

- **`team_a_dev_iris_drift_pipeline`**  
  Periodically compares summary statistics between the training and inference datasets (mean, std, min/max) for the four numeric Iris features, assigns a basic drift severity level, and publishes the report to MLflow and MinIO.

## Local Development / Quick Start

1. **Provision services** – You need Airflow, MLflow (tracking + model registry), and MinIO/S3. A simple local stack can be brought up using Docker Compose (outside the scope of this repo) as long as the endpoints match the environment variables listed above.
2. **Install dependencies** into the Airflow image or environment:
   ```bash
   pip install apache-airflow pandas scikit-learn mlflow boto3
   ```
3. **Set environment variables** for the Airflow scheduler/webserver (example):
   ```bash
   export MLFLOW_TRACKING_URI=http://localhost:5000
   export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
   export AWS_ACCESS_KEY_ID=minioadmin
   export AWS_SECRET_ACCESS_KEY=minioadmin
   export TEAM_S3_BUCKET=team-a-bucket
   export TEAM_S3_KEY=iris/iris_train.csv
   export TEAM_S3_TEST_KEY=iris/iris_test.csv
   ```
4. **Deploy the DAGs** by copying the files under `airflow/dags` into your Airflow DAGs folder (or mount the entire repo).
5. **Trigger runs** from the Airflow UI or CLI. The DAGs default to daily schedules (`@daily`), but you can trigger ad hoc runs for development/testing.

## Observability & Outputs

- **MLflow Experiments**:  
  - `team_a-dev-iris_classifier` for training metrics and artifacts.  
  - `team_a-dev-iris_batch_inference` for inference jobs.  
  - `team_a-dev-iris_drift_monitor` for drift reports.

- **Model Registry**: Models are registered under `team_a_dev_iris_model`, and the DAG ensures only one version stays in `Staging` with a `staging` alias.

- **S3/MinIO Artifacts**: Training data is read from the configured bucket, predictions are written to `TEAM_PREDICTIONS_PREFIX`, and drift reports are stored under `TEAM_DRIFT_PREFIX`.