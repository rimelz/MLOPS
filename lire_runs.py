import mlflow
import os

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db'))

runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"])

colonnes = ["tags.mlflow.runName", "params.n_estimators", "params.max_depth",
            "metrics.accuracy", "metrics.f1"]
print(runs[colonnes].to_string(index=False))

meilleur = runs.iloc[0]
print()
print("Meilleur run :", meilleur["tags.mlflow.runName"], "| id :", meilleur["run_id"][:8],
      "...")
