from dotenv import load_dotenv
import os
import mlflow
from mlflow import MlflowClient
from pprint import pprint
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# environment variables
load_dotenv("e1.env")
experiment_id = os.environ.get("MLFLOW_EXPERIMENT_ID")
databricks_host = os.environ.get("DATABRICKS_HOST")
mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
