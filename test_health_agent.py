"""
Test script for the Health Agent
This script demonstrates the capabilities of the health monitoring system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.health_agent import HealthAgent
from event_bus import event_bus, EventTypes, publish_health_alert
import pandas as pd
from datetime import datetime, timedelta

def test_health_agent():
    """Test the health agent functionality"""
    print("=" * 60)
    print("ELDERLY CARE HEALTH AGENT TEST")
    print("=" * 60)
    
    # Initialize the health agent
    print("\n1. Initializing Health Agent...")
    agent = HealthAgent(data_path="data/health_monitoring.csv")
    
    # Train the models
    print("\n2. Training Machine Learning Models...")
    print("   Loading and preprocessing data...")
    df = agent.load_and_preprocess_data()
    print(f"   Dataset loaded: {len(df)} records")
    
    print("   Training anomaly detection and alert prediction models...")
    agent.train_models(df)
    
    # Test with various health scenarios
    print("\n3. Testing Health Predictions...")
    
    test_scenarios = [
        {
            'name': 'Normal Health Status',
            'data': {
                'heart_rate': 75,
                'systolic_bp': 120,
                'diastolic_bp': 80,
                'glucose': 95,
                'oxygen_saturation': 98,
                'timestamp': datetime.now().isoformat()
            }
        },
        {
            'name': 'High Blood Pressure Alert',
            'data': {
                'heart_rate': 85,
                'systolic_bp': 160,
                'diastolic_bp': 95,
                'glucose': 110,
                'oxygen_saturation': 97,
                'timestamp': datetime.now().isoformat()
            }
        },
        {
            'name': 'Low Oxygen Saturation Emergency',
            'data': {
                'heart_rate': 110,
                'systolic_bp': 140,
                'diastolic_bp': 90,
                'glucose': 120,
                'oxygen_saturation': 88,
                'timestamp': datetime.now().isoformat()
            }
        },
        {
            'name': 'Diabetes Alert (High Glucose)',
            'data': {
                'heart_rate': 88,
                'systolic_bp': 130,
                'diastolic_bp': 85,
                'glucose': 180,
                'oxygen_saturation': 96,
                'timestamp': datetime.now().isoformat()
            }
        },
        {
            'name': 'Bradycardia (Low Heart Rate)',
            'data': {
                'heart_rate': 45,
                'systolic_bp': 100,
                'diastolic_bp': 65,
                'glucose': 90,
                'oxygen_saturation': 97,
                'timestamp': datetime.now().isoformat()
            }
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n   Scenario {i}: {scenario['name']}")
        print(f"   Input: {scenario['data']}")
        
        result = agent.predict_health_status(scenario['data'])
        
        print(f"   Results:")
        print(f"     Alert Required: {result.get('alert_required', 'N/A')}")
        print(f"     Risk Level: {result.get('risk_level', 'N/A')}")
        print(f"     Alert Probability: {result.get('alert_probability', 0):.3f}")
        print(f"     Is Anomaly: {result.get('is_anomaly', 'N/A')}")
        print(f"     Threshold Violations: {result.get('threshold_violations', 0)}")
        
        if result.get('recommendations'):
            print(f"     Recommendations:")
            for rec in result['recommendations']:
                print(f"       - {rec}")
        
        # Publish alert if required
        if result.get('alert_required'):
            publish_health_alert(
                device_id="TEST_DEVICE",
                alert_type=scenario['name'].lower().replace(' ', '_'),
                severity=result.get('risk_level', 'medium'),
                data=scenario['data']
            )
        
        print()
    
    # Test health insights
    print("\n4. Testing Health Insights...")
    insights = agent.get_health_insights(days=30)
    
    if 'error' not in insights:
        print("   Health Analytics Summary:")
        print(f"     Analysis Period: {insights.get('analysis_period', 'N/A')}")
        print(f"     Total Readings: {insights.get('total_readings', 0)}")
        print(f"     Alert Rate: {insights.get('alert_rate', 0):.3f}")
        print(f"     Average Heart Rate: {insights.get('avg_heart_rate', 0):.1f} bpm")
        print(f"     Average Blood Pressure: {insights.get('avg_systolic_bp', 0):.1f}/{insights.get('avg_diastolic_bp', 0):.1f} mmHg")
        print(f"     Average Glucose: {insights.get('avg_glucose', 0):.1f} mg/dL")
        print(f"     Average Oxygen Saturation: {insights.get('avg_oxygen_sat', 0):.1f}%")
        print(f"     Total Violations: {insights.get('total_violations', 0)}")
        print(f"     Most Common Violation Time: {insights.get('most_common_violation_time', 'N/A')}:00")
    else:
        print(f"   Error: {insights['error']}")
    
    # Test event bus integration
    print("\n5. Testing Event Bus Integration...")
    
    def health_event_handler(event):
        print(f"   Event Received: {event['type']} from {event['source']}")
        print(f"   Data: {event['data']}")
    
    # Subscribe to health events
    event_bus.subscribe(EventTypes.HEALTH_ALERT, health_event_handler)
    
    # Simulate a critical health event
    critical_data = {
        'heart_rate': 150,
        'systolic_bp': 180,
        'diastolic_bp': 110,
        'glucose': 250,
        'oxygen_saturation': 85,
        'timestamp': datetime.now().isoformat()
    }
    
    print("   Simulating critical health event...")
    result = agent.predict_health_status(critical_data)
    
    if result.get('alert_required'):
        publish_health_alert(
            device_id="CRITICAL_TEST",
            alert_type="multiple_critical_values",
            severity="critical",
            data=critical_data
        )
    
    # Show event history
    history = event_bus.get_event_history(EventTypes.HEALTH_ALERT, limit=5)
    print(f"\n   Recent Health Events: {len(history)} events")
    for event in history[-3:]:  # Show last 3 events
        print(f"     - {event['type']} at {event['timestamp']}")
    
    print("\n" + "=" * 60)
    print("HEALTH AGENT TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return agent

def demonstrate_real_time_monitoring():
    """Demonstrate real-time health monitoring simulation"""
    print("\n" + "=" * 60)
    print("REAL-TIME MONITORING SIMULATION")
    print("=" * 60)
    
    agent = HealthAgent()
    
    # Try to load pre-trained models
    try:
        agent._load_models()
        print("Loaded pre-trained models.")
    except:
        print("No pre-trained models found. Training new models...")
        agent.train_models()
    
    # Simulate 24 hours of monitoring data
    print("\nSimulating 24-hour health monitoring...")
    
    import random
    import time
    
    base_time = datetime.now()
    
    for hour in range(0, 24, 2):  # Every 2 hours
        # Simulate varying health parameters throughout the day
        current_time = base_time + timedelta(hours=hour)
        
        # Simulate circadian rhythm effects
        if 6 <= hour <= 18:  # Daytime
            hr_base = 75
            bp_sys_base = 125
            bp_dia_base = 80
        else:  # Nighttime
            hr_base = 65
            bp_sys_base = 115
            bp_dia_base = 75
        
        # Add some random variation
        health_data = {
            'heart_rate': hr_base + random.randint(-15, 20),
            'systolic_bp': bp_sys_base + random.randint(-20, 25),
            'diastolic_bp': bp_dia_base + random.randint(-10, 15),
            'glucose': 100 + random.randint(-20, 40),
            'oxygen_saturation': 97 + random.randint(-5, 3),
            'timestamp': current_time.isoformat()
        }
        
        result = agent.predict_health_status(health_data)
        
        status_icon = "🔴" if result.get('alert_required') else ("🟡" if result.get('risk_level') == 'medium' else "🟢")
        
        print(f"{status_icon} {current_time.strftime('%H:%M')} - HR: {health_data['heart_rate']}, "
              f"BP: {health_data['systolic_bp']}/{health_data['diastolic_bp']}, "
              f"Glucose: {health_data['glucose']}, SpO2: {health_data['oxygen_saturation']}% "
              f"[Risk: {result.get('risk_level', 'unknown')}]")
        
        if result.get('alert_required'):
            print(f"    ⚠️  ALERT: {', '.join(result.get('recommendations', [])[:2])}")
        
        # Small delay to simulate real-time
        time.sleep(0.1)
    
    print("\n24-hour monitoring simulation completed.")

if __name__ == "__main__":
    try:
        # Run main test
        agent = test_health_agent()
        
        # Ask user if they want to see real-time simulation
        print("\nWould you like to see a 24-hour monitoring simulation? (y/n): ", end="")
        if input().lower().startswith('y'):
            demonstrate_real_time_monitoring()
            
    except Exception as e:
        print(f"\nError during testing: {e}")
        print("Please ensure the data file exists at 'data/health_monitoring.csv'")
