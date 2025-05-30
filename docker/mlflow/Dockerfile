# docker/mlflow/Dockerfile

FROM python:3.9-slim

# Telepítjük az MLflow-t (extras-zel, ha S3, GCS support kell)
RUN pip install mlflow[extras]

# Artifact kötetet beállítjuk
VOLUME /mlflow/artifacts

# Környezeti változók (SQLite backend + lokális artifact root)
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts \
    MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db

# Port kitetes
EXPOSE 5000

# Indító parancs
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
