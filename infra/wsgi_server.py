from wsgi_basic_auth import BasicAuth
from mlflow.server import app as mlflow_app

app = BasicAuth(mlflow_app)