#!/bin/bash

# Host check
export MLFLOW_SERVER_ALLOWED_HOSTS=$MLFLOW_ALLOWED_HOSTS

# Backend Store: Cloud SQL
export MLFLOW_BACKEND_STORE_URI=$POSTGRESQL_URL

# Artifact Store: GCS
export MLFLOW_DEFAULT_ARTIFACT_ROOT=$STORAGE_URL

# Basic auth for login
export WSGI_AUTH_CREDENTIALS="${MLFLOW_ID}:${MLFLOW_PASS}"

# Start MLflow server using Gunicorn WSGI server
exec gunicorn -b "0.0.0.0:5000" wsgi_server:app