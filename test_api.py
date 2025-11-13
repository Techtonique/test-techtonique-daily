import argparse
import time
import logging 
import os 
import requests
import json
import jwt

from pathlib import Path
from typing import Dict, Any, List, Union
from functools import wraps
from datetime import datetime, timedelta
from dotenv import load_dotenv
from functools import wraps

# Load environment variables
load_dotenv(override=True)

SECRETKEY = os.getenv("SECRETKEY")
ADMIN_USER_ID = os.getenv("ADMIN_USER_ID")
ADMIN_USER_EMAIL = os.getenv("ADMIN_USER_EMAIL")
DATA_DIR = Path(__file__).parent / "data"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Base URL for API endpoints
base_url = "https://www.techtonique.net"

def create_access_token(data: dict):
    user_id = data.get("user_id")
    user_email = data.get("user_email")
    if not user_id:
        raise ValueError("user_id cannot be None or empty")
    if not user_email:
        raise ValueError("user_email cannot be None or empty")
    to_encode = {"sub": user_id, "email": user_email}
    expires_delta = timedelta(minutes=60)
    expire = datetime.now() + expires_delta
    to_encode.update({"exp": int(expire.timestamp())})  # Use Unix timestamp    
    token = jwt.encode(to_encode, SECRETKEY, algorithm="HS256")    
    return token, expire

def get_admin_token():
    user_id = ADMIN_USER_ID
    user_email = ADMIN_USER_EMAIL
    return create_access_token({"user_id": user_id, 
                                "user_email": user_email})[0]

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
        logger.info(f"Error making request to {url}:")
        logger.info(f"Status code: {e.response.status_code if hasattr(e, 'response') else 'N/A'}")
        logger.info(f"Response: {e.response.text if hasattr(e, 'response') else str(e)}")
        return None
    finally:
        # Close any open files
        if files and 'file' in files:
            files['file'][1].close()

def check_file_exists(file_path: str) -> bool:
    """Check if a file exists and is readable"""
    if not os.path.exists(file_path):
        logger.info(f"Error: File not found: {file_path}")
        return False
    if not os.access(file_path, os.R_OK):
        logger.info(f"Error: File not readable: {file_path}")
        return False
    return True

def test_forecasting(token: str):
    """Test forecasting endpoints"""
    test_files = {
        "univariate": DATA_DIR / "a10.csv",
        "multivariate": DATA_DIR / "ice_cream_vs_heater.csv"
    }
    
    logger.info("\n=== Testing Forecasting Endpoints ===")
    
    # Test univariate forecasting
    logger.info("\nTesting univariate forecasting...")
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
        logger.info("✓ Univariate forecasting successful")
    
    # Test multivariate forecasting
    logger.info("\nTesting multivariate forecasting...")
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
        logger.info("✓ Multivariate forecasting successful")

def test_ml(token: str):
    """Test machine learning endpoints"""
    test_files = {
        "classification": DATA_DIR / "breast_cancer_dataset2.csv",
        "regression": DATA_DIR / "boston_dataset2.csv"
    }
    
    logger.info("\n=== Testing Machine Learning Endpoints ===")
    
    # Test classification
    logger.info("\nTesting classification...")
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
        logger.info("✓ Classification successful")
    
    # Test regression
    logger.info("\nTesting regression...")
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
        logger.info("✓ Regression successful")
    
    # Test GBDT classification
    logger.info("\nTesting GBDT classification...")
    files = {"file": open(test_files["classification"], "rb")}
    response = make_request(f"{base_url}/gbdtclassification", token, files=files)
    test_results.append(TestResult(
        "GBDT Classification",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        logger.info("✓ GBDT classification successful")
    
    # Test GBDT classification with SHAP
    logger.info("\nTesting GBDT classification with SHAP...")
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
        logger.info("✓ GBDT classification with SHAP successful")
    
    # Test GBDT classification with permutation importance
    logger.info("\nTesting GBDT classification with permutation importance...")
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
        logger.info("✓ GBDT classification with permutation importance successful")
    
    # Test GBDT regression
    logger.info("\nTesting GBDT regression...")
    files = {"file": open(test_files["regression"], "rb")}
    params = {"return_pi": True}
    response = make_request(f"{base_url}/gbdtregression", token, files=files, params=params)
    test_results.append(TestResult(
        "GBDT Regression",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        logger.info("✓ GBDT regression successful")
    
    # Test GBDT regression with SHAP
    logger.info("\nTesting GBDT regression with SHAP...")
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
        logger.info("✓ GBDT regression with SHAP successful")
    
    # Test GBDT regression with permutation importance
    logger.info("\nTesting GBDT regression with permutation importance...")
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
        logger.info("✓ GBDT regression with permutation importance successful")

def test_reserving(token: str):
    """Test reserving endpoints"""
    test_files = {
        "chain_ladder": DATA_DIR / "raa.csv",
        "mack": DATA_DIR / "abc.csv"
    }
    
    logger.info("\n=== Testing Reserving Endpoints ===")
    
    # Test chain ladder
    logger.info("\nTesting chain ladder...")
    files = {"file": open(test_files["chain_ladder"], "rb")}
    params = {"method": "chainladder"}
    response = make_request(f"{base_url}/reserving", token, files=files, params=params)
    test_results.append(TestResult(
        "Chain Ladder",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        logger.info("✓ Chain ladder successful")
    
    # Test Mack chain ladder
    logger.info("\nTesting Mack chain ladder...")
    files = {"file": open(test_files["mack"], "rb")}
    params = {"method": "mack"}
    response = make_request(f"{base_url}/reserving", token, files=files, params=params)
    test_results.append(TestResult(
        "Mack Chain Ladder",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        logger.info("✓ Mack chain ladder successful")
    
    # Test RidgeCV reserving
    logger.info("\nTesting RidgeCV reserving...")
    files = {"file": open(test_files["mack"], "rb")}
    params = {"method": "RidgeCV"}
    response = make_request(f"{base_url}/mlreserving", token, files=files, params=params)
    test_results.append(TestResult(
        "RidgeCV reserving",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        logger.info("✓ RidgeCV reserving successful")

    # Test lightgbm reserving
    logger.info("\nTesting lightgbm reserving...")
    files = {"file": open(test_files["mack"], "rb")}
    params = {"method": "lightgbm"}
    response = make_request(f"{base_url}/mlreserving", token, files=files, params=params)
    test_results.append(TestResult(
        "lightgbm reserving",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        logger.info("✓ lightgbm reserving successful")

def test_survival(token: str):
    """Test survival analysis endpoints"""
    test_files = {
        "km": DATA_DIR / "kidney.csv",
        "ridge": DATA_DIR / "gbsg2_2.csv"
    }
    
    logger.info("\n=== Testing Survival Analysis Endpoints ===")
    
    # Test Kaplan-Meier
    logger.info("\nTesting Kaplan-Meier...")
    files = {"file": open(test_files["km"], "rb")}
    params = {"method": "km"}
    response = make_request(f"{base_url}/survivalcurve", token, files=files, params=params)
    test_results.append(TestResult(
        "Kaplan-Meier",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        logger.info("✓ Kaplan-Meier successful")
    
    # Test Ridge survival
    logger.info("\nTesting Ridge survival...")
    files = {"file": open(test_files["ridge"], "rb")}
    params = {"method": "RidgeCV", "patient_id": 0}
    response = make_request(f"{base_url}/survivalcurve", token, files=files, params=params)
    test_results.append(TestResult(
        "Ridge Survival",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        logger.info("✓ Ridge survival successful")

def test_simulations(token: str):
    """Test simulation endpoints"""
    logger.info("\n=== Testing Simulation Endpoints ===")
    
    # Test GBM simulation
    logger.info("\nTesting GBM simulation...")
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
        logger.info("✓ GBM simulation successful")
    
    # Test CIR simulation
    logger.info("\nTesting CIR simulation...")
    params["model"] = "CIR"
    response = make_request(f"{base_url}/scenarios/simulate/", token, method="GET", params=params)
    test_results.append(TestResult(
        "CIR Simulation",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        logger.info("✓ CIR simulation successful")
    
    # Test OU simulation
    logger.info("\nTesting OU simulation...")
    params["model"] = "OU"
    response = make_request(f"{base_url}/scenarios/simulate/", token, method="GET", params=params)
    test_results.append(TestResult(
        "OU Simulation",
        bool(response),
        None if response else "Failed to get response"
    ))
    if response:
        logger.info("✓ OU simulation successful")
    
    # Test shocks simulation
    logger.info("\nTesting shocks simulation...")
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
        logger.info("✓ Shocks simulation successful")

def print_test_summary():
    """Print a summary of all test results"""
    logger.info("\n=== Test Summary ===")
    logger.info(f"Total tests: {len(test_results)}")
    successful = sum(1 for r in test_results if r.success)
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {len(test_results) - successful}")
    
    if len(test_results) - successful > 0:
        logger.info("\nFailed Tests:")
        for result in test_results:
            if not result.success:
                logger.info(f"✗ {result.name}: {result.error}")

def main():

    # Use token from secrets if available
    if ADMIN_USER_ID and ADMIN_USER_EMAIL and SECRETKEY:
        token = get_admin_token()
        logger.info("🔑 Using admin token from GitHub secrets")
    else:
        token = get_token()
    
    # Run tests
    test_forecasting(token)
    test_ml(token)
    test_reserving(token)
    test_survival(token)
    test_simulations(token)
    
    # Print summary
    print_test_summary()

    # Exit with error code if any test failed
    if any(not r.success for r in test_results):
        raise SystemExit(1)

if __name__ == "__main__":
    main() 