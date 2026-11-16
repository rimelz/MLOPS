"""
Enregistre le meilleur modèle dans le registre MLflow
et lui attribue l'alias 'champion'.
"""
import mlflow
from mlflow.tracking import MlflowClient
import os

# Configuration
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db'))
client = MlflowClient()

# Recherche du meilleur run par f1-score
experiment = client.get_experiment_by_name("churn-prediction")
if experiment is None:
    print("✗ Expérience 'churn-prediction' introuvable")
    print("  Exécuter d'abord train_mlflow.py")
    exit(1)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.f1 DESC"],
    max_results=1
)

if not runs:
    print("✗ Aucun run trouvé")
    exit(1)

best_run = runs[0]
run_id = best_run.info.run_id
f1_score = best_run.data.metrics.get('f1', 0)

print(f"Meilleur run : {run_id}")
print(f"  F1-score : {f1_score:.4f}")

# Enregistrement du modèle
model_name = "churn-model"
model_uri = f"runs:/{run_id}/model"

try:
    # Tentative d'enregistrement
    model_version = mlflow.register_model(model_uri, model_name)
    version_number = model_version.version
    print(f"✓ Modèle enregistré : {model_name} version {version_number}")
    
    # Attribution de l'alias 'champion'
    client.set_registered_model_alias(model_name, "champion", version_number)
    print(f"✓ Alias 'champion' attribué à la version {version_number}")
    
except Exception as e:
    print(f"✗ Erreur : {e}")
    exit(1)
