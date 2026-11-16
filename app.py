"""
API FastAPI pour la prédiction du churn client.
Endpoints : /health et /predict
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
import pandas as pd
import os

app = FastAPI(title="Churn Prediction API")

# Configuration MLflow
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db')
MODEL_URI = os.getenv('MODEL_URI', 'model')

# Chargement du modèle
model = None

def load_model():
    global model
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        if MODEL_URI == 'model':
            # Mode Docker : chargement depuis le dossier local
            model = mlflow.pyfunc.load_model(MODEL_URI)
        else:
            # Mode MLflow : chargement depuis le registre
            model = mlflow.pyfunc.load_model(MODEL_URI)
        print(f"✓ Modèle chargé depuis {MODEL_URI}")
    except Exception as e:
        print(f"✗ Erreur de chargement du modèle : {e}")
        raise

@app.on_event("startup")
async def startup_event():
    load_model()

class PredictionRequest(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Âge du client")
    salaire: int = Field(..., ge=0, description="Salaire annuel")
    abonnement: str = Field(..., description="Type d'abonnement (Basic, Premium, Gold)")
    anciennete: int = Field(..., ge=0, description="Années d'ancienneté")
    consommation: int = Field(..., ge=0, le=100, description="Niveau de consommation")

class PredictionResponse(BaseModel):
    churn: int
    proba_churn: float

@app.get("/health")
def health_check():
    """Vérifie que l'API est opérationnelle."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "mlflow_uri": MLFLOW_TRACKING_URI,
        "model_uri": MODEL_URI
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Prédit si un client va churner."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    # Validation de l'abonnement
    if request.abonnement not in ['Basic', 'Premium', 'Gold']:
        raise HTTPException(
            status_code=400, 
            detail="abonnement doit être 'Basic', 'Premium' ou 'Gold'"
        )
    
    # Encodage de l'abonnement
    abonnement_map = {'Basic': 0, 'Premium': 2, 'Gold': 1}
    abonnement_encoded = abonnement_map[request.abonnement]
    
    # Préparation des features
    # Note : adapter selon les features du modèle
    try:
        input_data = pd.DataFrame({
            'age': [request.age],
            'salaire': [request.salaire],
            'abonnement_encoded': [abonnement_encoded],
            'anciennete': [request.anciennete],
            'consommation': [request.consommation]
        })
        
        # Prédiction
        prediction = model.predict(input_data)
        churn = int(prediction[0])
        
        # Probabilité (si disponible)
        proba_churn = 0.0
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_data)
            proba_churn = float(proba[0][1])
        
        return PredictionResponse(churn=churn, proba_churn=proba_churn)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
