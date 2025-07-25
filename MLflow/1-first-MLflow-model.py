import mlflow
from mlflow import MlflowClient
from pprint import pprint
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


#1. start MLflow tracking server

# see https://mlflow.org/docs/latest/ml/getting-started/logging-first-model/step1-tracking-server/
# run on terminal: mlflow server --host 127.0.0.1 --port 8080

# 2. Using MLflow client API

# configur MLflow tracking client using settings of 1 above
client = MlflowClient(tracking_uri="http://127.0.0.1:8080")

# searching experiments
all_experiments = client.search_experiments()
print(all_experiments)

# extract name and lifecycle_stage
default_experiment = [
    {"name": experiment.name, "lifecycle_stage": experiment.lifecycle_stage}
    for experiment in all_experiments
    if experiment.name == "Default"
][0]

pprint(default_experiment)

# 3. Creating experiments

# view by navigating to http://127.0.0.1:8080

# create Apples experiment with meaningful tags

# provide experiment description that will appear in UI
experiment_desc = (
    "This is the grocery forecasting project."
    "This experiment contains the produce models for apples."
)
# provide searchable tags that define characteristics of the runs
# that will be in this experiment

experiment_tags = {
    "project_name": "grocery-forecasting",
    "store_dep": "produce",
    "team": "stores-ml",
    "mlflow.note.content": experiment_desc
}

# create the experiment, providing a unique name
produce_apples_experiment = client.create_experiment(
    name = "Apple_ML", tags=experiment_tags
)

# search based on tags

apples_experiment = client.search_experiments(
    filter_string="tags.`project_name` = 'grocery-forecasting'"
)
print(vars(apples_experiment[0]))

# data generation

def generate_apple_sales_data_with_promo_adjustment(
    base_demand: int = 1000, n_rows: int = 5000
):
    """
    Generates a synthetic dataset for predicting apple sales demand with seasonality
    and inflation.

    This function creates a pandas DataFrame with features relevant to apple sales.
    The features include date, average_temperature, rainfall, weekend flag, holiday flag,
    promotional flag, price_per_kg, and the previous day's demand. The target variable,
    'demand', is generated based on a combination of these features with some added noise.

    Args:
        base_demand (int, optional): Base demand for apples. Defaults to 1000.
        n_rows (int, optional): Number of rows (days) of data to generate. Defaults to 5000.

    Returns:
        pd.DataFrame: DataFrame with features and target variable for apple sales prediction.

    Example:
        >>> df = generate_apple_sales_data_with_seasonality(base_demand=1200, n_rows=6000)
        >>> df.head()
    """

    # Set seed for reproducibility
    np.random.seed(9999)

    # Create date range
    dates = [datetime.now() - timedelta(days=i) for i in range(n_rows)]
    dates.reverse()

    # Generate features
    df = pd.DataFrame(
        {
            "date": dates,
            "average_temperature": np.random.uniform(10, 35, n_rows),
            "rainfall": np.random.exponential(5, n_rows),
            "weekend": [(date.weekday() >= 5) * 1 for date in dates],
            "holiday": np.random.choice([0, 1], n_rows, p=[0.97, 0.03]),
            "price_per_kg": np.random.uniform(0.5, 3, n_rows),
            "month": [date.month for date in dates],
        }
    )

    # Introduce inflation over time (years)
    df["inflation_multiplier"] = (
        1 + (df["date"].dt.year - df["date"].dt.year.min()) * 0.03
    )

    # Incorporate seasonality due to apple harvests
    df["harvest_effect"] = np.sin(2 * np.pi * (df["month"] - 3) / 12) + np.sin(
        2 * np.pi * (df["month"] - 9) / 12
    )

    # Modify the price_per_kg based on harvest effect
    df["price_per_kg"] = df["price_per_kg"] - df["harvest_effect"] * 0.5

    # Adjust promo periods to coincide with periods lagging peak harvest by 1 month
    peak_months = [4, 10]  # months following the peak availability
    df["promo"] = np.where(
        df["month"].isin(peak_months),
        1,
        np.random.choice([0, 1], n_rows, p=[0.85, 0.15]),
    )

    # Generate target variable based on features
    base_price_effect = -df["price_per_kg"] * 50
    seasonality_effect = df["harvest_effect"] * 50
    promo_effect = df["promo"] * 200

    df["demand"] = (
        base_demand
        + base_price_effect
        + seasonality_effect
        + promo_effect
        + df["weekend"] * 300
        + np.random.normal(0, 50, n_rows)
    ) * df[
        "inflation_multiplier"
    ]  # adding random noise

    # Add previous day's demand
    df["previous_days_demand"] = df["demand"].shift(1)
    df["previous_days_demand"].fillna(
        method="bfill", inplace=True
    )  # fill the first row

    # Drop temporary columns
    df.drop(columns=["inflation_multiplier", "harvest_effect", "month"], inplace=True)

    return df

data = generate_apple_sales_data_with_promo_adjustment(base_demand=1_000, n_rows=1_000)

data[-20:]

# 4 Using MLflow Tracking to keep track of training

# to use fluent API set global reference
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# set the current active experiment to Apple_ML
apples_experiment = mlflow.set_experiment("Apple_ML")

# define a run name for this iteration of training
run_name = "apples_rf_test"

# define an artifact path that the model will br saved to
artifact_path = "rf_apples"

# data
X = data.drop(columns=["date", "demand"])
y = data["demand"]

# split to traint, test set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
    "n_estimators": 100,
    "max_depth": 6,
    "min_samples_split": 10,
    "min_samples_leaf": 4,
    "bootstrap": True,
    "oob_score": False,
    "random_state": 888
}

# train the RandomForestRegressor
rf = RandomForestRegressor(**params)

# fit the model on training data
rf.fit(X_train, y_train)

# predict on validation set
y_pred = rf.predict(X_val)

# calculate error metrics
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)

# assemble the metrics we are going to write into a collectin
metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

# initiate the MLflow run context
with mlflow.start_run(run_name=run_name) as run:
    #log the parameters used for the model fit
    mlflow.log_params(params)
    #log the error metrics that were calculated during validation
    mlflow.log_metrics(metrics)
    # log an instance of the trained model for later use
    mlflow.sklearn.log_model(sk_model=rf, input_example=X_val, name=artifact_path)

