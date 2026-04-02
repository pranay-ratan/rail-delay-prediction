#!/usr/bin/env python3
"""
CPKC Rail Risk Intelligence Dashboard - Backend API Testing
Tests all 11 API endpoints for functionality and data integrity.
"""

import requests
import sys
import json
from datetime import datetime

class CPKCAPITester:
    def __init__(self, base_url="https://a128d83b-65a6-4962-b4a5-5653a57cf381.preview.emergentagent.com"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.failed_tests = []

    def run_test(self, name, method, endpoint, expected_status=200, data=None, validate_func=None):
        """Run a single API test with optional validation"""
        url = f"{self.base_url}/api/{endpoint}"
        headers = {'Content-Type': 'application/json'}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=10)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=10)

            print(f"   Status: {response.status_code}")
            
            if response.status_code == expected_status:
                try:
                    response_data = response.json()
                    print(f"   Response size: {len(json.dumps(response_data))} chars")
                    
                    # Run custom validation if provided
                    if validate_func:
                        validation_result = validate_func(response_data)
                        if validation_result is True:
                            self.tests_passed += 1
                            print(f"✅ {name} - PASSED")
                            return True, response_data
                        else:
                            print(f"❌ {name} - FAILED: {validation_result}")
                            self.failed_tests.append(f"{name}: {validation_result}")
                            return False, response_data
                    else:
                        self.tests_passed += 1
                        print(f"✅ {name} - PASSED")
                        return True, response_data
                        
                except json.JSONDecodeError:
                    print(f"❌ {name} - FAILED: Invalid JSON response")
                    self.failed_tests.append(f"{name}: Invalid JSON response")
                    return False, {}
            else:
                print(f"❌ {name} - FAILED: Expected {expected_status}, got {response.status_code}")
                self.failed_tests.append(f"{name}: Status {response.status_code} (expected {expected_status})")
                return False, {}

        except requests.exceptions.RequestException as e:
            print(f"❌ {name} - FAILED: Request error - {str(e)}")
            self.failed_tests.append(f"{name}: Request error - {str(e)}")
            return False, {}

    def validate_health(self, data):
        """Validate health endpoint response"""
        required_fields = ['status', 'model_loaded', 'data_loaded', 'timestamp']
        for field in required_fields:
            if field not in data:
                return f"Missing field: {field}"
        
        if data['status'] != 'healthy':
            return f"Status not healthy: {data['status']}"
        
        if not data['model_loaded']:
            return "Model not loaded"
        
        if not data['data_loaded']:
            return "Data not loaded"
        
        return True

    def validate_stats(self, data):
        """Validate stats endpoint response"""
        required_fields = ['total_incidents', 'provinces_covered', 'best_auc', 'best_model']
        for field in required_fields:
            if field not in data:
                return f"Missing field: {field}"
        
        if data['total_incidents'] <= 0:
            return f"Invalid total_incidents: {data['total_incidents']}"
        
        if data['provinces_covered'] <= 0:
            return f"Invalid provinces_covered: {data['provinces_covered']}"
        
        if not (0 <= data['best_auc'] <= 100):
            return f"Invalid best_auc: {data['best_auc']}"
        
        return True

    def validate_provinces(self, data):
        """Validate provinces endpoint response"""
        if 'provinces' not in data:
            return "Missing 'provinces' field"
        
        provinces = data['provinces']
        if not isinstance(provinces, list) or len(provinces) == 0:
            return "Provinces should be non-empty list"
        
        required_fields = ['province', 'code', 'incidents', 'high_risk', 'risk_pct', 'top_incident_type', 'risk_score']
        for i, prov in enumerate(provinces[:3]):  # Check first 3
            for field in required_fields:
                if field not in prov:
                    return f"Province {i} missing field: {field}"
        
        return True

    def validate_annual(self, data):
        """Validate annual incidents endpoint"""
        required_fields = ['years', 'total', 'high_risk', 'low_risk']
        for field in required_fields:
            if field not in data:
                return f"Missing field: {field}"
            if not isinstance(data[field], list):
                return f"Field {field} should be list"
        
        # Check all arrays have same length
        lengths = [len(data[field]) for field in required_fields]
        if len(set(lengths)) > 1:
            return f"Array length mismatch: {lengths}"
        
        return True

    def validate_seasonal(self, data):
        """Validate seasonal incidents endpoint"""
        required_fields = ['seasons', 'total', 'high_risk']
        for field in required_fields:
            if field not in data:
                return f"Missing field: {field}"
        
        expected_seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        if data['seasons'] != expected_seasons:
            return f"Unexpected seasons: {data['seasons']}"
        
        return True

    def validate_by_type(self, data):
        """Validate incidents by type endpoint"""
        required_fields = ['types', 'total', 'high_risk']
        for field in required_fields:
            if field not in data:
                return f"Missing field: {field}"
        
        if len(data['types']) == 0:
            return "No incident types returned"
        
        return True

    def validate_heatmap(self, data):
        """Validate heatmap endpoint"""
        required_fields = ['provinces', 'incident_types', 'values']
        for field in required_fields:
            if field not in data:
                return f"Missing field: {field}"
        
        if not isinstance(data['values'], list):
            return "Values should be list"
        
        return True

    def validate_correlation(self, data):
        """Validate correlation endpoint"""
        required_fields = ['features', 'values']
        for field in required_fields:
            if field not in data:
                return f"Missing field: {field}"
        
        if not isinstance(data['values'], list):
            return "Values should be list"
        
        return True

    def validate_model_metrics(self, data):
        """Validate model metrics endpoint"""
        if 'models' not in data:
            return "Missing 'models' field"
        
        models = data['models']
        if not isinstance(models, list):
            return "Models should be list"
        
        if len(models) > 0:
            required_fields = ['name', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            for field in required_fields:
                if field not in models[0]:
                    return f"Model missing field: {field}"
        
        return True

    def validate_feature_importance(self, data):
        """Validate feature importance endpoint"""
        required_fields = ['features', 'importances']
        for field in required_fields:
            if field not in data:
                return f"Missing field: {field}"
        
        if len(data['features']) != len(data['importances']):
            return "Features and importances length mismatch"
        
        return True

    def validate_prediction(self, data):
        """Validate prediction endpoint"""
        required_fields = ['probability', 'risk_level', 'contributions', 'recommendations', 'baseline', 'timestamp']
        for field in required_fields:
            if field not in data:
                return f"Missing field: {field}"
        
        if not (0 <= data['probability'] <= 1):
            return f"Invalid probability: {data['probability']}"
        
        if data['risk_level'] not in ['LOW', 'MEDIUM', 'HIGH']:
            return f"Invalid risk_level: {data['risk_level']}"
        
        if not isinstance(data['contributions'], list):
            return "Contributions should be list"
        
        if not isinstance(data['recommendations'], list):
            return "Recommendations should be list"
        
        return True

    def run_all_tests(self):
        """Run all API endpoint tests"""
        print("🚀 Starting CPKC Rail Risk Intelligence API Tests")
        print("=" * 60)

        # Test 1: Health Check
        self.run_test(
            "Health Check", "GET", "health", 
            validate_func=self.validate_health
        )

        # Test 2: Stats
        self.run_test(
            "System Stats", "GET", "stats",
            validate_func=self.validate_stats
        )

        # Test 3: Provinces
        self.run_test(
            "Province Data", "GET", "provinces",
            validate_func=self.validate_provinces
        )

        # Test 4: Annual Incidents
        self.run_test(
            "Annual Incidents", "GET", "incidents/annual",
            validate_func=self.validate_annual
        )

        # Test 5: Seasonal Incidents
        self.run_test(
            "Seasonal Incidents", "GET", "incidents/by-season",
            validate_func=self.validate_seasonal
        )

        # Test 6: Incidents by Type
        self.run_test(
            "Incidents by Type", "GET", "incidents/by-type",
            validate_func=self.validate_by_type
        )

        # Test 7: Heatmap Data
        self.run_test(
            "Province-Type Heatmap", "GET", "incidents/heatmap",
            validate_func=self.validate_heatmap
        )

        # Test 8: Correlation Data
        self.run_test(
            "Feature Correlation", "GET", "incidents/correlation",
            validate_func=self.validate_correlation
        )

        # Test 9: Model Metrics
        self.run_test(
            "Model Metrics", "GET", "models/metrics",
            validate_func=self.validate_model_metrics
        )

        # Test 10: Feature Importance
        self.run_test(
            "Feature Importance", "GET", "models/feature-importance",
            validate_func=self.validate_feature_importance
        )

        # Test 11: Prediction
        prediction_payload = {
            "province": "Ontario",
            "incident_type": "Derailment",
            "cargo_type": "Dangerous Goods",
            "season": "Winter",
            "year": 2024,
            "month": 1,
            "rolling_12m": 45,
            "fatalities": 0,
            "injuries": 0,
            "is_weekend": False,
            "mile_post": 150.0
        }
        
        self.run_test(
            "Risk Prediction", "POST", "predict",
            data=prediction_payload,
            validate_func=self.validate_prediction
        )

        # Print final results
        print("\n" + "=" * 60)
        print(f"📊 TEST SUMMARY")
        print(f"   Total Tests: {self.tests_run}")
        print(f"   Passed: {self.tests_passed}")
        print(f"   Failed: {len(self.failed_tests)}")
        print(f"   Success Rate: {(self.tests_passed/self.tests_run*100):.1f}%")

        if self.failed_tests:
            print(f"\n❌ FAILED TESTS:")
            for failure in self.failed_tests:
                print(f"   • {failure}")
        else:
            print(f"\n🎉 ALL TESTS PASSED!")

        return len(self.failed_tests) == 0

def main():
    tester = CPKCAPITester()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())