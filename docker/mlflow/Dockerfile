FROM python:3.12-slim

ARG MLFLOW_VERSION=3.17.1

# psycopg2: Backend Store в PostgreSQL, boto3: Artifact Store в S3 (MinIO)
RUN pip install --no-cache-dir \
    "mlflow==${MLFLOW_VERSION}" \
    "psycopg[binary]" \
    psycopg2-binary \
    boto3

EXPOSE 5000
