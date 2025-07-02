"""
Comprehensive Test Suite for Health Agent
This script performs thorough testing of all health agent functionality
"""

import sys
import os
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required modules at module level
try:
    from agents.health_agent import HealthAgent
    from event_bus import event_bus, EventTypes, publish_health_alert
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_AVAILABLE = False
    HealthAgent = None
    event_bus = None
    EventTypes = None
    publish_health_alert = None

def test_imports():
    """Test if all required imports work correctly"""
    print("🔍 Testing imports...")
    if IMPORTS_AVAILABLE:
        print("✅ All imports successful")
        return True
    else:
        print("❌ Import error occurred during module loading")
        return False

def test_health_agent_initialization():
    """Test if HealthAgent can be initialized properly"""
    print("\n🔍 Testing HealthAgent initialization...")
    
    if not IMPORTS_AVAILABLE or HealthAgent is None:
        print("❌ Cannot test initialization - imports failed")
        return False, None
    
    try:
        # Test with default parameters
        agent1 = HealthAgent()
        print("✅ Default initialization successful")
        
        # Test with custom parameters
        agent2 = HealthAgent(data_path="data/health_monitoring.csv", model_save_path="test_models")
        print("✅ Custom initialization successful")
        
        # Check if attributes are set correctly
        assert hasattr(agent1, 'scaler'), "Scaler not initialized"
        assert hasattr(agent1, 'anomaly_detector'), "Anomaly detector not initialized"
        assert hasattr(agent1, 'alert_predictor'), "Alert predictor not initialized"
        assert hasattr(agent1, 'normal_ranges'), "Normal ranges not set"
        print("✅ All attributes properly initialized")
        
        return True, agent1
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        traceback.print_exc()
        return False, None

def test_data_loading(agent):
    """Test data loading and preprocessing"""
    print("\n🔍 Testing data loading and preprocessing...")
    try:
        # Check if data file exists
        if not os.path.exists(agent.data_path):
            print(f"❌ Data file not found: {agent.data_path}")
            return False, None
        
        # Test data loading
        df = agent.load_and_preprocess_data()
        print(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Check required columns exist
        required_columns = [
            'Device-ID/User-ID', 'Timestamp', 'Heart Rate', 'Blood Pressure',
            'Glucose Levels', 'Oxygen Saturation (SpO₂%)', 'Alert Triggered (Yes/No)'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False, None
        
        print("✅ All required columns present")
        
        # Check derived columns were created
        derived_columns = ['Systolic_BP', 'Diastolic_BP', 'Hour', 'DayOfWeek', 'IsWeekend', 
                          'Pulse_Pressure', 'MAP', 'HR_Deviation', 'Glucose_Deviation', 
                          'SpO2_Deviation', 'Total_Violations']
        
        missing_derived = [col for col in derived_columns if col not in df.columns]
        if missing_derived:
            print(f"❌ Missing derived columns: {missing_derived}")
            return False, None
        
        print("✅ All derived columns created successfully")
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(df['Timestamp']), "Timestamp not datetime"
        assert pd.api.types.is_numeric_dtype(df['Heart Rate']), "Heart Rate not numeric"
        assert pd.api.types.is_numeric_dtype(df['Systolic_BP']), "Systolic BP not numeric"
        print("✅ Data types are correct")
        
        return True, df
    except Exception as e:
        print(f"❌ Data loading error: {e}")
        traceback.print_exc()
        return False, None

def test_feature_preparation(agent, df):
    """Test feature preparation for machine learning"""
    print("\n🔍 Testing feature preparation...")
    try:
        X, y = agent.prepare_features(df)
        
        print(f"✅ Features prepared: X shape {X.shape}, y shape {y.shape}")
        
        # Check feature dimensions
        expected_features = 14  # Number of features we expect
        if X.shape[1] != expected_features:
            print(f"❌ Unexpected number of features: {X.shape[1]}, expected {expected_features}")
            return False
        
        # Check target values are binary
        unique_targets = np.unique(y)
        if not all(val in [0, 1] for val in unique_targets):
            print(f"❌ Target values not binary: {unique_targets}")
            return False
        
        print("✅ Feature preparation successful")
        return True
    except Exception as e:
        print(f"❌ Feature preparation error: {e}")
        traceback.print_exc()
        return False

def test_model_training(agent):
    """Test model training functionality"""
    print("\n🔍 Testing model training...")
    try:
        # Train the models
        agent.train_models()
        
        # Check if models are trained
        assert agent.is_trained, "Agent not marked as trained"
        assert hasattr(agent, 'feature_columns') and agent.feature_columns, "Feature columns not set"
        print("✅ Model training completed successfully")
        
        # Check if model files are saved
        model_files = ['scaler.pkl', 'anomaly_detector.pkl', 'alert_predictor.pkl', 
                      'feature_columns.pkl', 'normal_ranges.pkl']
        
        for model_file in model_files:
            file_path = os.path.join(agent.model_save_path, model_file)
            if not os.path.exists(file_path):
                print(f"❌ Model file not saved: {model_file}")
                return False
        
        print("✅ All model files saved successfully")
        return True
    except Exception as e:
        print(f"❌ Model training error: {e}")
        traceback.print_exc()
        return False

def test_prediction_functionality(agent):
    """Test health status prediction with various scenarios"""
    print("\n🔍 Testing prediction functionality...")
    
    test_cases = [
        {
            'name': 'Normal Health',
            'data': {
                'heart_rate': 75,
                'systolic_bp': 120,
                'diastolic_bp': 80,
                'glucose': 95,
                'oxygen_saturation': 98
            },
            'expected_risk': 'low'
        },
        {
            'name': 'High Blood Pressure',
            'data': {
                'heart_rate': 85,
                'systolic_bp': 160,
                'diastolic_bp': 95,
                'glucose': 110,
                'oxygen_saturation': 97
            },
            'expected_violations': 2
        },
        {
            'name': 'Critical Multiple Issues',
            'data': {
                'heart_rate': 45,  # Low
                'systolic_bp': 180,  # High
                'diastolic_bp': 110,  # High
                'glucose': 250,  # High
                'oxygen_saturation': 88  # Low
            },
            'expected_risk': 'critical',
            'expected_alert': True
        },
        {
            'name': 'With Timestamp',
            'data': {
                'heart_rate': 80,
                'systolic_bp': 130,
                'diastolic_bp': 85,
                'glucose': 105,
                'oxygen_saturation': 96,
                'timestamp': '2025-07-02 14:30:00'
            }
        },
        {
            'name': 'Blood Pressure String Format',
            'data': {
                'heart_rate': 72,
                'blood_pressure': '125/82',
                'glucose': 92,
                'oxygen_saturation': 97
            }
        }
    ]
    
    success_count = 0
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n   Test Case {i}: {test_case['name']}")
        try:
            result = agent.predict_health_status(test_case['data'])
            
            # Check required fields in result
            required_fields = ['alert_required', 'alert_probability', 'is_anomaly', 
                             'anomaly_score', 'risk_level', 'threshold_violations', 
                             'recommendations', 'timestamp']
            
            missing_fields = [field for field in required_fields if field not in result]
            if missing_fields:
                print(f"   ❌ Missing result fields: {missing_fields}")
                continue
            
            # Check specific expectations
            if 'expected_risk' in test_case:
                if result['risk_level'] != test_case['expected_risk']:
                    print(f"   ⚠️  Risk level mismatch: got {result['risk_level']}, expected {test_case['expected_risk']}")
                else:
                    print(f"   ✅ Risk level correct: {result['risk_level']}")
            
            if 'expected_violations' in test_case:
                if result['threshold_violations'] < test_case['expected_violations']:
                    print(f"   ⚠️  Violations lower than expected: got {result['threshold_violations']}, expected >= {test_case['expected_violations']}")
                else:
                    print(f"   ✅ Violations detected: {result['threshold_violations']}")
            
            if 'expected_alert' in test_case:
                if result['alert_required'] != test_case['expected_alert']:
                    print(f"   ⚠️  Alert status mismatch: got {result['alert_required']}, expected {test_case['expected_alert']}")
                else:
                    print(f"   ✅ Alert status correct: {result['alert_required']}")
            
            print(f"   📊 Results: Alert={result['alert_required']}, Risk={result['risk_level']}, "
                  f"Violations={result['threshold_violations']}, Recommendations={len(result['recommendations'])}")
            
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ Prediction error: {e}")
            traceback.print_exc()
    
    print(f"\n✅ Prediction tests: {success_count}/{len(test_cases)} successful")
    return success_count == len(test_cases)

def test_health_insights(agent):
    """Test health insights functionality"""
    print("\n🔍 Testing health insights...")
    try:
        # Test general insights
        insights = agent.get_health_insights(days=30)
        
        if 'error' in insights:
            print(f"❌ Insights error: {insights['error']}")
            return False
        
        required_insight_fields = ['analysis_period', 'total_readings', 'alert_rate', 
                                 'avg_heart_rate', 'avg_systolic_bp', 'avg_diastolic_bp']
        
        missing_fields = [field for field in required_insight_fields if field not in insights]
        if missing_fields:
            print(f"❌ Missing insight fields: {missing_fields}")
            return False
        
        print(f"✅ Health insights generated successfully")
        print(f"   📈 Analysis period: {insights['analysis_period']}")
        print(f"   📊 Total readings: {insights['total_readings']}")
        print(f"   ⚠️  Alert rate: {insights['alert_rate']:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Health insights error: {e}")
        traceback.print_exc()
        return False

def test_model_persistence(agent):
    """Test model saving and loading"""
    print("\n🔍 Testing model persistence...")
    
    if not IMPORTS_AVAILABLE or HealthAgent is None:
        print("❌ Cannot test model persistence - imports failed")
        return False
    
    try:
        # Create a new agent instance
        new_agent = HealthAgent(model_save_path=agent.model_save_path)
        
        # Try to load models
        new_agent._load_models()
        
        if not new_agent.is_trained:
            print("❌ Models not loaded properly")
            return False
        
        # Test prediction with loaded models
        test_data = {
            'heart_rate': 80,
            'systolic_bp': 125,
            'diastolic_bp': 82,
            'glucose': 100,
            'oxygen_saturation': 97
        }
        
        result = new_agent.predict_health_status(test_data)
        
        if 'error' in result:
            print(f"❌ Prediction error with loaded models: {result['error']}")
            return False
        
        print("✅ Model persistence working correctly")
        return True
    except Exception as e:
        print(f"❌ Model persistence error: {e}")
        traceback.print_exc()
        return False

def test_event_bus_integration():
    """Test integration with event bus"""
    print("\n🔍 Testing event bus integration...")
    
    if not IMPORTS_AVAILABLE or event_bus is None or EventTypes is None or publish_health_alert is None:
        print("❌ Cannot test event bus integration - imports failed")
        return False
    
    try:
        # Test event publishing
        test_event_received = {'received': False}
        
        def test_handler(event):
            test_event_received['received'] = True
            print(f"   📨 Event received: {event['type']}")
        
        # Subscribe to health alerts
        event_bus.subscribe(EventTypes.HEALTH_ALERT, test_handler)
        
        # Publish a test health alert
        publish_health_alert(
            device_id="TEST_DEVICE",
            alert_type="test_alert",
            severity="medium",
            data={'test': True}
        )
        
        # Give a moment for the event to be processed
        import time
        time.sleep(0.1)
        
        if not test_event_received['received']:
            print("❌ Event not received")
            return False
        
        print("✅ Event bus integration working")
        return True
    except Exception as e:
        print(f"❌ Event bus integration error: {e}")
        traceback.print_exc()
        return False

def test_error_handling(agent):
    """Test error handling with invalid inputs"""
    print("\n🔍 Testing error handling...")
    
    error_test_cases = [
        {
            'name': 'Empty input',
            'data': {}
        },
        {
            'name': 'Invalid data types',
            'data': {
                'heart_rate': 'invalid',
                'glucose': 'not_a_number'
            }
        },
        {
            'name': 'Extreme values',
            'data': {
                'heart_rate': -50,
                'systolic_bp': 500,
                'glucose': -100,
                'oxygen_saturation': 150
            }
        }
    ]
    
    success_count = 0
    for test_case in error_test_cases:
        print(f"\n   Testing: {test_case['name']}")
        try:
            result = agent.predict_health_status(test_case['data'])
            
            # Should handle errors gracefully
            if 'error' in result or result.get('alert_required') is not None:
                print(f"   ✅ Error handled gracefully")
                success_count += 1
            else:
                print(f"   ❌ Error not handled properly")
        except Exception as e:
            print(f"   ❌ Unhandled exception: {e}")
    
    print(f"\n✅ Error handling tests: {success_count}/{len(error_test_cases)} successful")
    return success_count == len(error_test_cases)

def run_comprehensive_test():
    """Run all tests and provide a comprehensive report"""
    print("🚀 STARTING COMPREHENSIVE HEALTH AGENT TESTING")
    print("=" * 60)
    
    test_results = {}
    
    # Test 1: Imports
    test_results['imports'] = test_imports()
    if not test_results['imports']:
        print("\n❌ CRITICAL: Import test failed. Cannot continue testing.")
        return False
    
    # Test 2: Initialization
    test_results['initialization'], agent = test_health_agent_initialization()
    if not test_results['initialization']:
        print("\n❌ CRITICAL: Initialization test failed. Cannot continue testing.")
        return False
    
    # Test 3: Data Loading
    if test_results['initialization'] and agent is not None:
        test_results['data_loading'], df = test_data_loading(agent)
    else:
        test_results['data_loading'], df = False, None
        print("\n❌ Skipping data loading test - initialization failed")
    
    if not test_results['data_loading']:
        print("\n❌ CRITICAL: Data loading test failed. Cannot continue with ML tests.")
        df = None
    
    # Test 4: Feature Preparation
    if test_results['data_loading'] and agent is not None and df is not None:
        test_results['feature_preparation'] = test_feature_preparation(agent, df)
    else:
        test_results['feature_preparation'] = False
        print("\n❌ Skipping feature preparation test - data loading failed")
    
    # Test 5: Model Training
    if test_results['initialization'] and agent is not None:
        test_results['model_training'] = test_model_training(agent)
    else:
        test_results['model_training'] = False
        print("\n❌ Skipping model training test - initialization failed")
    
    # Test 6: Prediction Functionality
    if test_results['model_training'] and agent is not None:
        test_results['prediction'] = test_prediction_functionality(agent)
    else:
        test_results['prediction'] = False
        print("\n❌ Skipping prediction test - model training failed")
    
    # Test 7: Health Insights
    if test_results['initialization'] and agent is not None:
        test_results['health_insights'] = test_health_insights(agent)
    else:
        test_results['health_insights'] = False
        print("\n❌ Skipping health insights test - initialization failed")
    
    # Test 8: Model Persistence
    if test_results['model_training'] and agent is not None:
        test_results['model_persistence'] = test_model_persistence(agent)
    else:
        test_results['model_persistence'] = False
        print("\n❌ Skipping model persistence test - model training failed")
    
    # Test 9: Event Bus Integration
    test_results['event_bus'] = test_event_bus_integration()
    
    # Test 10: Error Handling
    if test_results['initialization'] and agent is not None:
        test_results['error_handling'] = test_error_handling(agent)
    else:
        test_results['error_handling'] = False
        print("\n❌ Skipping error handling test - initialization failed")
    
    # Generate final report
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():.<40} {status}")
        if result:
            passed_tests += 1
    
    print("-" * 60)
    print(f"OVERALL RESULT: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Health Agent is working perfectly!")
        return True
    elif passed_tests >= total_tests * 0.8:
        print("⚠️  Most tests passed. Health Agent is mostly working correctly.")
        return True
    else:
        print("❌ Multiple test failures. Health Agent needs fixing.")
        return False

if __name__ == "__main__":
    try:
        success = run_comprehensive_test()
        
        if success:
            print("\n🎯 Health Agent is ready for production use!")
        else:
            print("\n⚠️  Health Agent needs attention before production use.")
            
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        traceback.print_exc()
