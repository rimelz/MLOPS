# Guide de démarrage rapide

## Installation

```powershell
# Cloner le repository
git clone https://github.com/rimelz/MLOPS.git
cd MLOPS

# Créer un environnement virtuel
python -m venv venv
.\venv\Scripts\Activate.ps1

# Installer les dépendances
pip install -r requirements.txt
```

## Workflow complet

### 1. Générer les données

```powershell
python generate_data.py
```

Crée `churn_data.csv` avec 500 clients fictifs.

### 2. Entraîner les modèles avec MLflow

```powershell
python train_mlflow.py
```

Lance 3 expériences avec différentes configurations.

### 3. Visualiser les expériences

```powershell
mlflow ui
```

Ouvrir http://localhost:5000 dans le navigateur.

### 4. Enregistrer le meilleur modèle

```powershell
python register_model.py
```

Le modèle avec le meilleur F1-score est enregistré comme "champion".

### 5. Tester le champion

```powershell
python load_champion.py
```

### 6. Lancer l'API

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 7. Tester l'API

```powershell
python test_api.py
```

## Docker

### Image standalone

```powershell
# Exporter le champion
python export_champion.py

# Construire l'image
docker build -t churn-api:1.0 .

# Lancer le conteneur
docker run -d -p 8000:8000 --name api churn-api:1.0

# Tester
curl.exe http://localhost:8000/health

# Nettoyer
docker stop api
docker rm api
```

### Docker Compose (Stack complète)

```powershell
# Lancer MLflow
docker compose up -d mlflow

# Attendre que MLflow soit prêt
curl.exe http://localhost:5000/health

# Entraîner et enregistrer
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
$env:PYTHONUTF8="1"
python train_mlflow.py
python register_model.py
Remove-Item Env:MLFLOW_TRACKING_URI
Remove-Item Env:PYTHONUTF8

# Lancer la stack complète
docker compose up -d --build

# Vérifier
docker compose ps
curl.exe http://localhost:8000/health
python test_api.py

# Voir les logs
docker compose logs -f api

# Arrêter
docker compose down
```

## Gestion des branches

```powershell
# Développement
git checkout develop

# Nouvelle fonctionnalité
git checkout -b feature/nom-feature

# Retour à master
git checkout master
```

## Structure du projet

```
MLOPS/
├── generate_data.py       # Génération des données
├── train_mlflow.py        # Entraînement avec MLflow
├── register_model.py      # Enregistrement du champion
├── load_champion.py       # Test du champion
├── deplacer_alias.py      # Gestion des alias
├── export_champion.py     # Export pour Docker
├── app.py                 # API FastAPI
├── test_api.py            # Tests de l'API
├── requirements.txt       # Dépendances Python
├── Dockerfile             # Image Docker
├── .dockerignore          # Exclusions Docker
├── docker-compose.yml     # Stack MLflow + API
└── README.md              # Documentation complète
```

## Commandes utiles

```powershell
# MLflow
mlflow ui --host 0.0.0.0 --port 5000

# API
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Docker
docker compose up -d
docker compose down
docker compose logs -f
docker compose ps

# Git
git status
git log --oneline --graph --all
git branch -a
```
