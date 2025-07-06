import os
import sys
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Import all agents from the agents package
from agents import HealthAgent, SafetyAgent, ReminderAgent, AlertAgent
from event_bus import event_bus, EventTypes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('elderly_care_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ElderlyCareSystem:
    """
    Multi-Agent Elderly Care System Coordinator
    
    This class orchestrates all agents (Health, Safety, Reminder, Alert) to provide
    comprehensive elderly care monitoring and response. It handles:
    - Agent initialization and coordination
    - Model persistence and loading
    - Real-time monitoring and analysis
    - Emergency response coordination
    - User profile management
    """
    
    def __init__(self, config_path: str = "config/system_config.json"):
        """
        Initialize the Elderly Care System
        
        Args:
            config_path: Path to system configuration file
        """
        self.config_path = config_path
        self.config = {}
        self.agents = {}
        self.user_profiles = {}
        self.monitoring_active = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize system
        self._load_configuration()
        self._initialize_agents()
        self._setup_event_handlers()
        
        logger.info("Elderly Care System initialized successfully")
    
    def _load_configuration(self):
        """Load system configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                self._create_default_configuration()
                
            logger.info("System configuration loaded")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._create_default_configuration()
    
    def _create_default_configuration(self):
        """Create default system configuration"""
        default_config = {
            "system": {
                "monitoring_interval": 60,  # seconds
                "model_persistence": True,
                "auto_train_models": True,
                "emergency_response_enabled": True
            },
            "data_paths": {
                "health_data": "data/health_monitoring.csv",
                "safety_data": "data/safety_monitoring.csv",
                "reminder_data": "data/daily_reminder.csv"
            },
            "model_paths": {
                "health_models": "models/health/",
                "safety_models": "models/safety/",
                "reminder_models": "models/reminder/"
            },
            "thresholds": {
                "health_critical": 0.8,
                "safety_critical": 0.9,
                "compliance_critical": 0.3
            },
            "users": {}
        }
        
        # Create directories
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        for path in default_config["model_paths"].values():
            os.makedirs(path, exist_ok=True)
        
        # Save configuration
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        self.config = default_config
        logger.info("Default configuration created")
    
    def _initialize_agents(self):
        """Initialize all agents with persistent models"""
        try:
            # Initialize Health Agent
            logger.info("Initializing Health Agent...")
            self.agents['health'] = HealthAgent(
                data_path=self.config["data_paths"]["health_data"],
                model_save_path=self.config["model_paths"]["health_models"]
            )
            
            # Load or train health models
            if self._models_exist('health') and self.config["system"]["model_persistence"]:
                logger.info("Loading existing health models...")
                self.agents['health']._load_models()
            elif self.config["system"]["auto_train_models"]:
                logger.info("Training health models...")
                self.agents['health'].train_models()
            
            # Initialize Safety Agent
            logger.info("Initializing Safety Agent...")
            self.agents['safety'] = SafetyAgent(
                data_path=self.config["data_paths"]["safety_data"],
                model_save_path=self.config["model_paths"]["safety_models"]
            )
            
            # Load or train safety models
            if self._models_exist('safety') and self.config["system"]["model_persistence"]:
                logger.info("Loading existing safety models...")
                self.agents['safety']._load_models()
            elif self.config["system"]["auto_train_models"]:
                logger.info("Training safety models...")
                self.agents['safety'].train_models()
            
            # Initialize Reminder Agent
            logger.info("Initializing Reminder Agent...")
            self.agents['reminder'] = ReminderAgent(
                data_path=self.config["data_paths"]["reminder_data"],
                model_save_path=self.config["model_paths"]["reminder_models"]
            )
            
            # Load or train reminder models
            if self._models_exist('reminder') and self.config["system"]["model_persistence"]:
                logger.info("Loading existing reminder models...")
                self.agents['reminder']._load_models()
            elif self.config["system"]["auto_train_models"]:
                logger.info("Training reminder models...")
                self.agents['reminder'].train_models()
            
            # Initialize Alert Agent
            logger.info("Initializing Alert Agent...")
            self.agents['alert'] = AlertAgent()
            
            logger.info("All agents initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
            raise
    
    def _models_exist(self, agent_type: str) -> bool:
        """Check if models exist for an agent"""
        model_path = self.config["model_paths"][f"{agent_type}_models"]
        
        if agent_type == 'health':
            required_files = ['scaler.pkl', 'anomaly_detector.pkl', 'alert_predictor.pkl']
        elif agent_type == 'safety':
            required_files = ['safety_scaler.pkl', 'safety_fall_detector.pkl', 'safety_risk_assessor.pkl']
        elif agent_type == 'reminder':
            required_files = ['reminder_scaler.pkl', 'reminder_compliance_predictor.pkl']
        else:
            return False
        
        return all(os.path.exists(os.path.join(model_path, f)) for f in required_files)
    
    def _setup_event_handlers(self):
        """Setup event handlers for inter-agent communication"""
        # Health agent events
        event_bus.subscribe(EventTypes.HEALTH_ALERT, self._handle_health_alert)
        event_bus.subscribe(EventTypes.HEALTH_ANOMALY, self._handle_health_anomaly)
        
        # Safety agent events
        event_bus.subscribe(EventTypes.FALL_DETECTED, self._handle_fall_detected)
        event_bus.subscribe(EventTypes.EMERGENCY_DETECTED, self._handle_emergency)
        
        # Reminder agent events
        event_bus.subscribe(EventTypes.REMINDER_MISSED, self._handle_reminder_missed)
        event_bus.subscribe(EventTypes.MEDICATION_DUE, self._handle_medication_due)
        
        logger.info("Event handlers configured")
    
    def register_user(self, user_id: str, user_profile: Dict) -> Dict:
        """
        Register a new user with the system
        
        Args:
            user_id: Unique user identifier
            user_profile: User profile information
            
        Returns:
            Registration status
        """
        try:
            # Validate required fields
            required_fields = ['name', 'age', 'emergency_contacts', 'medical_conditions']
            for field in required_fields:
                if field not in user_profile:
                    return {'status': 'error', 'message': f'Missing required field: {field}'}
            
            # Create comprehensive user profile
            enhanced_profile = {
                'user_id': user_id,
                'name': user_profile['name'],
                'age': user_profile['age'],
                'emergency_contacts': user_profile['emergency_contacts'],
                'medical_conditions': user_profile['medical_conditions'],
                'medications': user_profile.get('medications', []),
                'mobility_level': user_profile.get('mobility_level', 'normal'),
                'risk_factors': user_profile.get('risk_factors', []),
                'preferences': user_profile.get('preferences', {}),
                'registered_at': datetime.now().isoformat(),
                'status': 'active'
            }
            
            # Store in system
            self.user_profiles[user_id] = enhanced_profile
            self.config['users'][user_id] = enhanced_profile
            
            # Setup user-specific reminder schedule
            if 'reminder_schedule' in user_profile:
                self.agents['reminder'].create_reminder_schedule(
                    user_id, user_profile['reminder_schedule']
                )
            
            # Save configuration
            self._save_configuration()
            
            logger.info(f"User {user_id} registered successfully")
            return {'status': 'success', 'message': 'User registered successfully'}
            
        except Exception as e:
            logger.error(f"Error registering user {user_id}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def monitor_user(self, user_id: str, health_data: Optional[Dict] = None,
                    safety_data: Optional[Dict] = None, reminder_data: Optional[Dict] = None) -> Dict:
        """
        Monitor a user across all dimensions
        
        Args:
            user_id: User identifier
            health_data: Health monitoring data
            safety_data: Safety monitoring data
            reminder_data: Reminder compliance data
            
        Returns:
            Comprehensive monitoring results
        """
        if user_id not in self.user_profiles:
            return {'status': 'error', 'message': 'User not registered'}
        
        results = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'status': 'monitored',
            'alerts_generated': []
        }
        
        try:
            # Health monitoring
            if health_data:
                health_result = self.agents['health'].predict_health_status(health_data)
                results['health_analysis'] = health_result
                
                # Check for health alerts
                if health_result.get('alert_required') or health_result.get('risk_level') in ['critical', 'high']:
                    alert_id = self.agents['alert'].receive_alert('health_agent', health_result)
                    results['alerts_generated'].append(alert_id)
                    
                    # Publish health event
                    event_bus.publish(
                        EventTypes.HEALTH_ALERT,
                        health_result,
                        source='health_agent'
                    )
            
            # Safety monitoring
            if safety_data:
                safety_result = self.agents['safety'].assess_safety_status(safety_data)
                results['safety_analysis'] = safety_result
                
                # Check for safety alerts
                if (safety_result.get('fall_detected') or 
                    safety_result.get('emergency_level') in ['critical', 'high']):
                    alert_id = self.agents['alert'].receive_alert('safety_agent', safety_result)
                    results['alerts_generated'].append(alert_id)
                    
                    # Publish safety event
                    event_type = EventTypes.FALL_DETECTED if safety_result.get('fall_detected') else EventTypes.EMERGENCY_DETECTED
                    event_bus.publish(event_type, safety_result, source='safety_agent')
            
            # Reminder monitoring
            if reminder_data:
                reminder_result = self.agents['reminder'].predict_compliance(
                    user_id, reminder_data.get('reminder_type', 'Medication'), 
                    reminder_data.get('scheduled_time', '08:00')
                )
                results['reminder_analysis'] = reminder_result
                
                # Check for compliance alerts
                if reminder_result.get('risk_level') in ['critical', 'high']:
                    alert_id = self.agents['alert'].receive_alert('reminder_agent', reminder_result)
                    results['alerts_generated'].append(alert_id)
                    
                    # Publish reminder event
                    event_bus.publish(
                        EventTypes.REMINDER_MISSED,
                        reminder_result,
                        source='reminder_agent'
                    )
            
            # Get daily reminders
            results['daily_reminders'] = self.agents['reminder'].get_daily_reminders(user_id)
            
            # Get active alerts
            results['active_alerts'] = self.agents['alert'].get_active_alerts(user_id)
            
            return results
            
        except Exception as e:
            logger.error(f"Error monitoring user {user_id}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_user_dashboard(self, user_id: str) -> Dict:
        """
        Get comprehensive dashboard for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            User dashboard data
        """
        if user_id not in self.user_profiles:
            return {'status': 'error', 'message': 'User not registered'}
        
        try:
            dashboard = {
                'user_profile': self.user_profiles[user_id],
                'timestamp': datetime.now().isoformat()
            }
            
            # Health insights
            dashboard['health_insights'] = self.agents['health'].get_health_insights(user_id, days=7)
            
            # Safety insights
            dashboard['safety_insights'] = self.agents['safety'].get_safety_insights(user_id, days=7)
            
            # Compliance insights
            dashboard['compliance_insights'] = self.agents['reminder'].get_compliance_insights(user_id, days=7)
            
            # Active alerts
            dashboard['active_alerts'] = self.agents['alert'].get_active_alerts(user_id)
            
            # Today's reminders
            dashboard['todays_reminders'] = self.agents['reminder'].get_daily_reminders(user_id)
            
            # Overall status assessment
            dashboard['overall_status'] = self._assess_overall_status(user_id, dashboard)
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Error generating dashboard for user {user_id}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def start_continuous_monitoring(self):
        """Start continuous monitoring for all registered users"""
        if self.monitoring_active:
            return {'status': 'error', 'message': 'Monitoring already active'}
        
        self.monitoring_active = True
        
        def monitoring_loop():
            logger.info("Continuous monitoring started")
            while self.monitoring_active:
                try:
                    # Check escalations
                    self.agents['alert'].check_escalations()
                    
                    # Sleep for monitoring interval
                    time.sleep(self.config["system"]["monitoring_interval"])
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(30)  # Wait before retrying
        
        # Start monitoring in separate thread
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        logger.info("Continuous monitoring started")
        return {'status': 'success', 'message': 'Monitoring started'}
    
    def stop_continuous_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        logger.info("Continuous monitoring stopped")
        return {'status': 'success', 'message': 'Monitoring stopped'}
    
    # Event handlers
    def _handle_health_alert(self, event: Dict):
        """Handle health alert events"""
        logger.warning(f"Health alert received: {event['data']}")
    
    def _handle_health_anomaly(self, event: Dict):
        """Handle health anomaly events"""
        logger.info(f"Health anomaly detected: {event['data']}")
    
    def _handle_fall_detected(self, event: Dict):
        """Handle fall detection events"""
        logger.critical(f"Fall detected: {event['data']}")
    
    def _handle_emergency(self, event: Dict):
        """Handle emergency events"""
        logger.critical(f"Emergency detected: {event['data']}")
    
    def _handle_reminder_missed(self, event: Dict):
        """Handle missed reminder events"""
        logger.warning(f"Reminder missed: {event['data']}")
    
    def _handle_medication_due(self, event: Dict):
        """Handle medication due events"""
        logger.info(f"Medication due: {event['data']}")
    
    def _assess_overall_status(self, user_id: str, dashboard: Dict) -> Dict:
        """Assess overall user status"""
        status = {
            'level': 'normal',
            'concerns': [],
            'recommendations': []
        }
        
        # Check health status
        health_insights = dashboard.get('health_insights', {})
        if health_insights.get('alert_rate', 0) > self.config["thresholds"]["health_critical"]:
            status['level'] = 'critical'
            status['concerns'].append('High health alert rate')
        
        # Check safety status
        safety_insights = dashboard.get('safety_insights', {})
        if safety_insights.get('fall_rate', 0) > self.config["thresholds"]["safety_critical"]:
            status['level'] = 'critical'
            status['concerns'].append('High fall risk')
        
        # Check compliance status
        compliance_insights = dashboard.get('compliance_insights', {})
        if compliance_insights.get('overall_compliance_rate', 1) < self.config["thresholds"]["compliance_critical"]:
            if status['level'] != 'critical':
                status['level'] = 'warning'
            status['concerns'].append('Low medication compliance')
        
        # Generate recommendations
        if status['concerns']:
            status['recommendations'].append('Schedule immediate caregiver check-in')
            if status['level'] == 'critical':
                status['recommendations'].append('Consider emergency medical evaluation')
        
        return status
    
    def _save_configuration(self):
        """Save system configuration"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def shutdown(self):
        """Shutdown the system gracefully"""
        logger.info("Shutting down Elderly Care System...")
        
        # Stop monitoring
        self.stop_continuous_monitoring()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Save configuration
        self._save_configuration()
        
        logger.info("Elderly Care System shutdown complete")

# Example usage and testing
if __name__ == "__main__":
    # Initialize the system
    system = ElderlyCareSystem()
    
    print("=== Elderly Care Multi-Agent System Demo ===")
    
    # Register a user
    print("\n1. Registering User")
    user_profile = {
        'name': 'John Doe',
        'age': 75,
        'emergency_contacts': {
            'primary': {'name': 'Jane Doe', 'phone': '+1234567890'},
            'secondary': {'name': 'Dr. Smith', 'phone': '+1234567891'}
        },
        'medical_conditions': ['Hypertension', 'Diabetes'],
        'medications': [
            {'name': 'Lisinopril', 'times': ['08:00', '20:00']},
            {'name': 'Metformin', 'times': ['08:00', '12:00', '18:00']}
        ],
        'reminder_schedule': {
            'Medication': {
                'times': ['08:00', '12:00', '18:00', '20:00'],
                'message': 'Time to take your medication',
                'days_of_week': [0, 1, 2, 3, 4, 5, 6]
            },
            'Exercise': {
                'times': ['10:00'],
                'message': 'Time for your daily walk',
                'days_of_week': [0, 1, 2, 3, 4, 5]
            }
        }
    }
    
    result = system.register_user('D1001', user_profile)
    print(f"Registration result: {result}")
    
    # Monitor user with sample data
    print("\n2. Monitoring User")
    health_data = {
        'heart_rate': 95,
        'systolic_bp': 140,
        'diastolic_bp': 85,
        'glucose': 120,
        'oxygen_saturation': 96,
        'timestamp': datetime.now().isoformat()
    }
    
    safety_data = {
        'movement_activity': 'Walking',
        'location': 'Living Room',
        'impact_force': 'None',
        'post_fall_inactivity': 0,
        'timestamp': datetime.now().isoformat()
    }
    
    reminder_data = {
        'reminder_type': 'Medication',
        'scheduled_time': '08:00'
    }
    
    monitoring_result = system.monitor_user('D1001', health_data, safety_data, reminder_data)
    print(f"Monitoring completed. Alerts generated: {len(monitoring_result.get('alerts_generated', []))}")
    
    # Get user dashboard
    print("\n3. User Dashboard")
    dashboard = system.get_user_dashboard('D1001')
    if 'overall_status' in dashboard:
        print(f"Overall Status: {dashboard['overall_status']['level']}")
        print(f"Active Alerts: {len(dashboard.get('active_alerts', []))}")
        print(f"Today's Reminders: {len(dashboard.get('todays_reminders', []))}")
    
    # Start continuous monitoring
    print("\n4. Starting Continuous Monitoring")
    system.start_continuous_monitoring()
    
    # Wait for a few seconds
    time.sleep(5)
    
    # Stop monitoring and shutdown
    print("\n5. Shutting Down System")
    system.shutdown()
    
    print("\n=== Demo Complete ===")
