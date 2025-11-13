import argparse
import requests
import os
from pathlib import Path
import json
from typing import Dict, Any, List
import time

# Base URL for API endpoints
base_url = "https://www.techtonique.net"

class TestResult:
    def __init__(self, name: str, success: bool, error: str = None):
        self.name = name
        self.success = success
        self.error = error

test_results: List[TestResult] = []

def get_token() -> str:
    """Get token from command line argument or prompt"""
    parser = argparse.ArgumentParser(description='Test API endpoints')
    parser.add_argument('--token', help='JWT token for authentication')
    args = parser.parse_args()
    
    if args.token:
        return args.token
    
    return input("Please enter your JWT token: ")

def make_request(url: str, token: str, method: str = "POST", files: Dict = None, params: Dict = None) -> Dict[str, Any]:
    """Make an API request and return the response"""
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        if method == "POST":
            # If files are provided, ensure they have the correct format
            if files and 'file' in files:
                file_path = files['file'].name
                file_name = os.path.basename(file_path)
                files = {
                    'file': (file_name, open(file_path, 'rb'), 'text/csv')
                }
            response = requests.post(url, headers=headers, files=files, params=params)
        else:
            response = requests.get(url, headers=headers, params=params)
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making request to {url}:")
        print(f"Status code: {e.response.status_code if hasattr(e, 'response') else 'N/A'}")
        print(f"Response: {e.response.text if hasattr(e, 'response') else str(e)}")
        return None
    finally:
        # Close any open files
        if files and 'file' in files:
            files['file'][1].close()

def check_file_exists(file_path: str) -> bool:
    """Check if a file exists and is readable"""
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return False
    if not os.access(file_path, os.R_OK):
        print(f"Error: File not readable: {file_path}")
        return False
    return True

def test_forecasting(token: str):
    """Test forecasting endpoints"""
    test_files = {
        "univariate": "/data/a10.csv",
        "multivariate": "/data/ice_cream_vs_heater.csv"
    }
    
    print("\n=== Testing Forecasting Endpoints ===")
    
    # Test univariate forecasting
    print("\nTesting univariate forecasting...")
    files = {"file": open(test_files["univariate"], "rb")}
    params = {
        "base_model": "RidgeCV",
        "n_hidden_features": 5,
        "lags": 25,
        "type_pi": "kde",
        "replications": 4,
        "h": 3
    }
    response = make_request(f"{base_url}/forecasting", token, files=files, params=params)
    test_results.append(TestResult(
        "Univariate Forecasting",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        print("✓ Univariate forecasting successful")
    
    # Test multivariate forecasting
    print("\nTesting multivariate forecasting...")
    files = {"file": open(test_files["multivariate"], "rb")}
    params = {
        "base_model": "RidgeCV",
        "n_hidden_features": 5,
        "lags": 25,
        "h": 3
    }
    response = make_request(f"{base_url}/forecasting", token, files=files, params=params)
    test_results.append(TestResult(
        "Multivariate Forecasting",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        print("✓ Multivariate forecasting successful")

def test_ml(token: str):
    """Test machine learning endpoints"""
    test_files = {
        "classification": "/data/breast_cancer_dataset2.csv",
        "regression": "/data/boston_dataset2.csv"
    }
    
    print("\n=== Testing Machine Learning Endpoints ===")
    
    # Test classification
    print("\nTesting classification...")
    files = {"file": open(test_files["classification"], "rb")}
    params = {
        "base_model": "RandomForestClassifier",
        "n_hidden_features": 5,
        "predict_proba": True
    }
    response = make_request(f"{base_url}/mlclassification", token, files=files, params=params)
    test_results.append(TestResult(
        "Classification",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        print("✓ Classification successful")
    
    # Test regression
    print("\nTesting regression...")
    files = {"file": open(test_files["regression"], "rb")}
    params = {
        "base_model": "RidgeCV",
        "n_hidden_features": 5,
        "return_pi": True
    }
    response = make_request(f"{base_url}/mlregression", token, files=files, params=params)
    test_results.append(TestResult(
        "Regression",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        print("✓ Regression successful")
    
    # Test GBDT classification
    print("\nTesting GBDT classification...")
    files = {"file": open(test_files["classification"], "rb")}
    response = make_request(f"{base_url}/gbdtclassification", token, files=files)
    test_results.append(TestResult(
        "GBDT Classification",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        print("✓ GBDT classification successful")
    
    # Test GBDT classification with SHAP
    print("\nTesting GBDT classification with SHAP...")
    files = {"file": open(test_files["classification"], "rb")}
    params = {
        "model_type": "xgboost",
        "predict_proba": False,
        "interpretability": "shap"
    }
    response = make_request(f"{base_url}/gbdtclassification", token, files=files, params=params)
    test_results.append(TestResult(
        "GBDT Classification with SHAP",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        print("✓ GBDT classification with SHAP successful")
    
    # Test GBDT classification with permutation importance
    print("\nTesting GBDT classification with permutation importance...")
    files = {"file": open(test_files["classification"], "rb")}
    params = {
        "model_type": "xgboost",
        "predict_proba": False,
        "interpretability": "permutation"
    }
    response = make_request(f"{base_url}/gbdtclassification", token, files=files, params=params)
    test_results.append(TestResult(
        "GBDT Classification with Permutation",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        print("✓ GBDT classification with permutation importance successful")
    
    # Test GBDT regression
    print("\nTesting GBDT regression...")
    files = {"file": open(test_files["regression"], "rb")}
    params = {"return_pi": True}
    response = make_request(f"{base_url}/gbdtregression", token, files=files, params=params)
    test_results.append(TestResult(
        "GBDT Regression",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        print("✓ GBDT regression successful")
    
    # Test GBDT regression with SHAP
    print("\nTesting GBDT regression with SHAP...")
    files = {"file": open(test_files["regression"], "rb")}
    params = {
        "model_type": "xgboost",
        "interpretability": "shap"
    }
    response = make_request(f"{base_url}/gbdtregression", token, files=files, params=params)
    test_results.append(TestResult(
        "GBDT Regression with SHAP",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        print("✓ GBDT regression with SHAP successful")
    
    # Test GBDT regression with permutation importance
    print("\nTesting GBDT regression with permutation importance...")
    files = {"file": open(test_files["regression"], "rb")}
    params = {
        "model_type": "xgboost",
        "interpretability": "permutation"
    }
    response = make_request(f"{base_url}/gbdtregression", token, files=files, params=params)
    test_results.append(TestResult(
        "GBDT Regression with Permutation",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        print("✓ GBDT regression with permutation importance successful")

def test_reserving(token: str):
    """Test reserving endpoints"""
    test_files = {
        "chain_ladder": "/data/raa.csv",
        "mack": "/data/abc.csv"
    }
    
    print("\n=== Testing Reserving Endpoints ===")
    
    # Test chain ladder
    print("\nTesting chain ladder...")
    files = {"file": open(test_files["chain_ladder"], "rb")}
    params = {"method": "chainladder"}
    response = make_request(f"{base_url}/reserving", token, files=files, params=params)
    test_results.append(TestResult(
        "Chain Ladder",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        print("✓ Chain ladder successful")
    
    # Test Mack chain ladder
    print("\nTesting Mack chain ladder...")
    files = {"file": open(test_files["mack"], "rb")}
    params = {"method": "mack"}
    response = make_request(f"{base_url}/reserving", token, files=files, params=params)
    test_results.append(TestResult(
        "Mack Chain Ladder",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        print("✓ Mack chain ladder successful")
    
    # Test RidgeCV reserving
    print("\nTesting RidgeCV reserving...")
    files = {"file": open(test_files["mack"], "rb")}
    params = {"method": "RidgeCV"}
    response = make_request(f"{base_url}/mlreserving", token, files=files, params=params)
    test_results.append(TestResult(
        "RidgeCV reserving",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        print("✓ RidgeCV reserving successful")

    # Test lightgbm reserving
    print("\nTesting lightgbm reserving...")
    files = {"file": open(test_files["mack"], "rb")}
    params = {"method": "lightgbm"}
    response = make_request(f"{base_url}/mlreserving", token, files=files, params=params)
    test_results.append(TestResult(
        "lightgbm reserving",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        print("✓ lightgbm reserving successful")

def test_survival(token: str):
    """Test survival analysis endpoints"""
    test_files = {
        "km": "/data/kidney.csv",
        "ridge": "/data/gbsg2_2.csv"
    }
    
    print("\n=== Testing Survival Analysis Endpoints ===")
    
    # Test Kaplan-Meier
    print("\nTesting Kaplan-Meier...")
    files = {"file": open(test_files["km"], "rb")}
    params = {"method": "km"}
    response = make_request(f"{base_url}/survivalcurve", token, files=files, params=params)
    test_results.append(TestResult(
        "Kaplan-Meier",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        print("✓ Kaplan-Meier successful")
    
    # Test Ridge survival
    print("\nTesting Ridge survival...")
    files = {"file": open(test_files["ridge"], "rb")}
    params = {"method": "RidgeCV", "patient_id": 0}
    response = make_request(f"{base_url}/survivalcurve", token, files=files, params=params)
    test_results.append(TestResult(
        "Ridge Survival",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        print("✓ Ridge survival successful")

def test_simulations(token: str):
    """Test simulation endpoints"""
    print("\n=== Testing Simulation Endpoints ===")
    
    # Test GBM simulation
    print("\nTesting GBM simulation...")
    params = {
        "model": "GBM",
        "n": 6,
        "horizon": 5,
        "frequency": "quarterly",
        "x0": 100,
        "theta1": 0.1,
        "theta2": 0.2,
        "theta3": 0.3,
        "seed": 123
    }
    response = make_request(f"{base_url}/scenarios/simulate/", token, method="GET", params=params)
    test_results.append(TestResult(
        "GBM Simulation",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        print("✓ GBM simulation successful")
    
    # Test CIR simulation
    print("\nTesting CIR simulation...")
    params["model"] = "CIR"
    response = make_request(f"{base_url}/scenarios/simulate/", token, method="GET", params=params)
    test_results.append(TestResult(
        "CIR Simulation",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        print("✓ CIR simulation successful")
    
    # Test OU simulation
    print("\nTesting OU simulation...")
    params["model"] = "OU"
    response = make_request(f"{base_url}/scenarios/simulate/", token, method="GET", params=params)
    test_results.append(TestResult(
        "OU Simulation",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        print("✓ OU simulation successful")
    
    # Test shocks simulation
    print("\nTesting shocks simulation...")
    params = {
        "model": "shocks",
        "n": 6,
        "horizon": 5,
        "frequency": "quarterly",
        "seed": 123
    }
    response = make_request(f"{base_url}/scenarios/simulate/", token, method="GET", params=params)
    test_results.append(TestResult(
        "Shocks Simulation",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        print("✓ Shocks simulation successful")

def print_test_summary():
    """Print a summary of all test results"""
    print("\n=== Test Summary ===")
    print(f"Total tests: {len(test_results)}")
    successful = sum(1 for r in test_results if r.success)
    print(f"Successful: {successful}")
    print(f"Failed: {len(test_results) - successful}")
    
    if len(test_results) - successful > 0:
        print("\nFailed Tests:")
        for result in test_results:
            if not result.success:
                print(f"✗ {result.name}: {result.error}")

def main():
    # Get token
    token = get_token()
    
    # Run tests
    test_forecasting(token)
    test_ml(token)
    test_reserving(token)
    test_survival(token)
    test_simulations(token)
    
    # Print summary
    print_test_summary()

if __name__ == "__main__":
    main() 