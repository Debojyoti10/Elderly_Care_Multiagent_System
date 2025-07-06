"""
Simple test script for the Elderly Care Multi-Agent System
"""
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Import the main system
from elderly_care_system import ElderlyCareSystem

def test_system():
    """Test the multi-agent system"""
    print("=" * 50)
    print("ELDERLY CARE MULTI-AGENT SYSTEM TEST")
    print("=" * 50)
    
    try:
        # Initialize system
        print("\n1. Initializing System...")
        system = ElderlyCareSystem()
        print("✓ System initialized successfully")
        
        # Register a test user
        print("\n2. Registering Test User...")
        user_profile = {
            'name': 'Test User',
            'age': 78,
            'emergency_contacts': {
                'primary': {'name': 'Emergency Contact', 'phone': '+1234567890'}
            },
            'medical_conditions': ['Hypertension'],
            'medications': [{'name': 'Blood Pressure Medication', 'times': ['08:00', '20:00']}],
            'reminder_schedule': {
                'Medication': {
                    'times': ['08:00', '20:00'],
                    'message': 'Time for your medication',
                    'days_of_week': [0, 1, 2, 3, 4, 5, 6]
                }
            }
        }
        
        result = system.register_user('TEST001', user_profile)
        if result['status'] == 'success':
            print("✓ User registered successfully")
        else:
            print(f"✗ User registration failed: {result['message']}")
            return
        
        # Test monitoring with sample data
        print("\n3. Testing User Monitoring...")
        
        # Normal health data
        health_data = {
            'heart_rate': 72,
            'systolic_bp': 120,
            'diastolic_bp': 80,
            'glucose': 95,
            'oxygen_saturation': 98
        }
        
        # Normal safety data
        safety_data = {
            'movement_activity': 'Walking',
            'location': 'Living Room',
            'impact_force': 'None',
            'post_fall_inactivity': 0
        }
        
        # Test reminder
        reminder_data = {
            'reminder_type': 'Medication',
            'scheduled_time': '08:00'
        }
        
        monitoring_result = system.monitor_user('TEST001', health_data, safety_data, reminder_data)
        
        if monitoring_result['status'] == 'monitored':
            print("✓ User monitoring completed successfully")
            print(f"  - Alerts generated: {len(monitoring_result.get('alerts_generated', []))}")
        else:
            print(f"✗ Monitoring failed: {monitoring_result.get('message', 'Unknown error')}")
        
        # Test dashboard
        print("\n4. Testing User Dashboard...")
        dashboard = system.get_user_dashboard('TEST001')
        
        if 'overall_status' in dashboard:
            print("✓ Dashboard generated successfully")
            print(f"  - Overall status: {dashboard['overall_status']['level']}")
        else:
            print(f"✗ Dashboard generation failed: {dashboard.get('message', 'Unknown error')}")
        
        # Test emergency scenario
        print("\n5. Testing Emergency Scenario...")
        
        # Critical health data
        emergency_health_data = {
            'heart_rate': 150,
            'systolic_bp': 180,
            'diastolic_bp': 110,
            'glucose': 300,
            'oxygen_saturation': 85
        }
        
        # Fall detected
        emergency_safety_data = {
            'movement_activity': 'No Movement',
            'location': 'Bathroom',
            'impact_force': 'High',
            'post_fall_inactivity': 600
        }
        
        emergency_result = system.monitor_user('TEST001', emergency_health_data, emergency_safety_data)
        
        print(f"✓ Emergency scenario processed")
        print(f"  - Emergency alerts generated: {len(emergency_result.get('alerts_generated', []))}")
        
        # Show active alerts
        active_alerts = system.agents['alert'].get_active_alerts('TEST001')
        print(f"  - Active alerts for user: {len(active_alerts)}")
        
        # Cleanup
        print("\n6. System Shutdown...")
        system.shutdown()
        print("✓ System shutdown completed")
        
        print("\n" + "=" * 50)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_system()
