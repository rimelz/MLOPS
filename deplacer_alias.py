"""
Script pour déplacer ou restaurer l'alias 'champion'
entre différentes versions du modèle.
"""
import mlflow
from mlflow.tracking import MlflowClient
import os

# Configuration
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db'))
client = MlflowClient()

model_name = "churn-model"

try:
    # Liste des versions
    versions = client.search_model_versions(f"name='{model_name}'")
    
    if not versions:
        print(f"✗ Aucune version du modèle '{model_name}' trouvée")
        exit(1)
    
    print(f"Versions disponibles pour {model_name} :")
    for v in versions:
        aliases = v.aliases if hasattr(v, 'aliases') else []
        alias_str = f" [aliases: {', '.join(aliases)}]" if aliases else ""
        print(f"  - Version {v.version}{alias_str}")
    
    # Demande de la version cible
    print("\nEntrez le numéro de version pour l'alias 'champion' :")
    version = input("> ").strip()
    
    # Attribution de l'alias
    client.set_registered_model_alias(model_name, "champion", version)
    print(f"✓ Alias 'champion' déplacé vers la version {version}")
    
except Exception as e:
    print(f"✗ Erreur : {e}")
