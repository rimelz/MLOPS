"""
Tests de l'API churn-prediction.
Vérifie /health et /predict avec différents cas.
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test du endpoint /health"""
    print("=== Test /health ===")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        print("✓ Test /health réussi\n")
        return True
    except Exception as e:
        print(f"✗ Test /health échoué : {e}\n")
        return False

def test_predict_churn():
    """Test de prédiction : cas churn probable"""
    print("=== Test /predict - Cas CHURN ===")
    payload = {
        "age": 25,
        "salaire": 30000,
        "abonnement": "Basic",
        "anciennete": 1,
        "consommation": 15
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        print(f"Requête: {json.dumps(payload, indent=2)}")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        assert response.status_code == 200
        result = response.json()
        assert "churn" in result
        print(f"✓ Prédiction : {'CHURN' if result['churn'] == 1 else 'NO CHURN'}\n")
        return True
    except Exception as e:
        print(f"✗ Test predict échoué : {e}\n")
        return False

def test_predict_no_churn():
    """Test de prédiction : cas sans churn"""
    print("=== Test /predict - Cas NO CHURN ===")
    payload = {
        "age": 45,
        "salaire": 75000,
        "abonnement": "Premium",
        "anciennete": 5,
        "consommation": 80
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        print(f"Requête: {json.dumps(payload, indent=2)}")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        assert response.status_code == 200
        result = response.json()
        assert "churn" in result
        print(f"✓ Prédiction : {'CHURN' if result['churn'] == 1 else 'NO CHURN'}\n")
        return True
    except Exception as e:
        print(f"✗ Test predict échoué : {e}\n")
        return False

def test_predict_invalid():
    """Test de validation : données invalides"""
    print("=== Test /predict - Validation ===")
    payload = {
        "age": 25,
        "salaire": 30000,
        "abonnement": "InvalidType",  # Abonnement invalide
        "anciennete": 1,
        "consommation": 15
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        print(f"Requête: {json.dumps(payload, indent=2)}")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        assert response.status_code == 400
        print("✓ Validation correctement rejetée\n")
        return True
    except Exception as e:
        print(f"✗ Test validation échoué : {e}\n")
        return False

if __name__ == "__main__":
    print("🧪 Tests de l'API Churn Prediction\n")
    
    results = []
    results.append(test_health())
    results.append(test_predict_churn())
    results.append(test_predict_no_churn())
    results.append(test_predict_invalid())
    
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Résultats : {passed}/{total} tests réussis")
    
    if passed == total:
        print("✓ Tous les tests sont passés !")
    else:
        print(f"✗ {total - passed} test(s) échoué(s)")
