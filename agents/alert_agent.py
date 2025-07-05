import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import os
import pickle
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertPriority(Enum):
    """Alert priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

class AlertStatus(Enum):
    """Alert status types"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    EXPIRED = "expired"

class AlertType(Enum):
    """Types of alerts"""
    HEALTH_EMERGENCY = "health_emergency"
    FALL_DETECTED = "fall_detected"
    MEDICATION_MISSED = "medication_missed"
    SAFETY_ANOMALY = "safety_anomaly"
    COMPLIANCE_CRITICAL = "compliance_critical"
    SYSTEM_ALERT = "system_alert"

class AlertAgent:
    """
    Central Alert Management Agent for Elderly Care System
    
    This agent coordinates alerts from all other agents (Health, Safety, Reminder)
    and manages the complete emergency response workflow including:
    - Alert prioritization and routing
    - Contact management and escalation
    - Response coordination
    - Alert lifecycle management
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Alert Agent
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or "config/alert_config.json"
        self.active_alerts = {}
        self.alert_history = deque(maxlen=10000)  # Store last 10,000 alerts
        self.contact_registry = {}
        self.escalation_rules = {}
        self.response_protocols = {}
        self.alert_stats = defaultdict(int)
        
        # Load configuration
        self._load_configuration()
        
        # Initialize default protocols
        self._initialize_default_protocols()
    
    def _load_configuration(self):
        """Load alert configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.contact_registry = config.get('contacts', {})
                    self.escalation_rules = config.get('escalation_rules', {})
                    self.response_protocols = config.get('response_protocols', {})
                    logger.info("Alert configuration loaded successfully")
            else:
                logger.warning(f"Configuration file not found: {self.config_path}")
                self._create_default_configuration()
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._create_default_configuration()
    
    def _create_default_configuration(self):
        """Create default configuration"""
        default_config = {
            "contacts": {
                "primary_caregiver": {
                    "name": "Primary Caregiver",
                    "phone": "+1234567890",
                    "email": "caregiver@example.com",
                    "role": "primary",
                    "priority": 1
                },
                "family_member": {
                    "name": "Family Member",
                    "phone": "+1234567891",
                    "email": "family@example.com",
                    "role": "family",
                    "priority": 2
                },
                "emergency_services": {
                    "name": "Emergency Services",
                    "phone": "911",
                    "email": "emergency@local.gov",
                    "role": "emergency",
                    "priority": 1
                },
                "healthcare_provider": {
                    "name": "Healthcare Provider",
                    "phone": "+1234567892",
                    "email": "doctor@clinic.com",
                    "role": "medical",
                    "priority": 2
                }
            },
            "escalation_rules": {
                "critical": {
                    "immediate_contacts": ["emergency_services", "primary_caregiver"],
                    "escalation_time": 300,  # 5 minutes
                    "escalation_contacts": ["family_member", "healthcare_provider"]
                },
                "high": {
                    "immediate_contacts": ["primary_caregiver"],
                    "escalation_time": 900,  # 15 minutes
                    "escalation_contacts": ["family_member", "healthcare_provider"]
                },
                "medium": {
                    "immediate_contacts": ["primary_caregiver"],
                    "escalation_time": 3600,  # 1 hour
                    "escalation_contacts": ["family_member"]
                },
                "low": {
                    "immediate_contacts": ["primary_caregiver"],
                    "escalation_time": 7200,  # 2 hours
                    "escalation_contacts": []
                }
            },
            "response_protocols": {
                "health_emergency": {
                    "priority": "critical",
                    "auto_escalate": True,
                    "required_actions": ["call_emergency_services", "notify_primary_caregiver"]
                },
                "fall_detected": {
                    "priority": "critical",
                    "auto_escalate": True,
                    "required_actions": ["assess_response", "call_emergency_services"]
                },
                "medication_missed": {
                    "priority": "high",
                    "auto_escalate": False,
                    "required_actions": ["remind_patient", "notify_caregiver"]
                },
                "safety_anomaly": {
                    "priority": "medium",
                    "auto_escalate": False,
                    "required_actions": ["investigate", "increase_monitoring"]
                }
            }
        }
        
        # Save default configuration
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        self.contact_registry = default_config['contacts']
        self.escalation_rules = default_config['escalation_rules']
        self.response_protocols = default_config['response_protocols']
        
        logger.info("Default configuration created")
    
    def _initialize_default_protocols(self):
        """Initialize default response protocols"""
        # Default response actions
        self.response_actions = {
            'call_emergency_services': self._call_emergency_services,
            'notify_primary_caregiver': self._notify_primary_caregiver,
            'notify_family': self._notify_family,
            'notify_healthcare_provider': self._notify_healthcare_provider,
            'assess_response': self._assess_response,
            'remind_patient': self._remind_patient,
            'investigate': self._investigate_issue,
            'increase_monitoring': self._increase_monitoring,
            'document_incident': self._document_incident,
            'schedule_followup': self._schedule_followup
        }
    
    def receive_alert(self, source_agent: str, alert_data: Dict) -> str:
        """
        Receive an alert from another agent
        
        Args:
            source_agent: Name of the agent sending the alert
            alert_data: Alert data dictionary
            
        Returns:
            Alert ID for tracking
        """
        # Generate unique alert ID
        alert_id = f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{source_agent}"
        
        # Standardize alert data
        standardized_alert = self._standardize_alert(source_agent, alert_data)
        standardized_alert['alert_id'] = alert_id
        standardized_alert['received_at'] = datetime.now().isoformat()
        standardized_alert['status'] = AlertStatus.ACTIVE.value
        
        # Store alert
        self.active_alerts[alert_id] = standardized_alert
        self.alert_history.append(standardized_alert.copy())
        
        # Update statistics
        self.alert_stats[f"{source_agent}_alerts"] += 1
        self.alert_stats[f"{standardized_alert['priority']}_alerts"] += 1
        
        # Process alert
        self._process_alert(alert_id, standardized_alert)
        
        logger.info(f"Alert received from {source_agent}: {alert_id}")
        return alert_id
    
    def _standardize_alert(self, source_agent: str, alert_data: Dict) -> Dict:
        """
        Standardize alert data format
        
        Args:
            source_agent: Source agent name
            alert_data: Raw alert data
            
        Returns:
            Standardized alert dictionary
        """
        # Map source agent alerts to standard format
        alert_mapping = {
            'health_agent': {
                'type': AlertType.HEALTH_EMERGENCY.value,
                'priority_key': 'risk_level',
                'message_key': 'recommendations'
            },
            'safety_agent': {
                'type': AlertType.FALL_DETECTED.value if alert_data.get('fall_detected') else AlertType.SAFETY_ANOMALY.value,
                'priority_key': 'emergency_level',
                'message_key': 'recommendations'
            },
            'reminder_agent': {
                'type': AlertType.MEDICATION_MISSED.value if alert_data.get('reminder_type') == 'Medication' else AlertType.COMPLIANCE_CRITICAL.value,
                'priority_key': 'risk_level',
                'message_key': 'recommendations'
            }
        }
        
        mapping = alert_mapping.get(source_agent, {
            'type': AlertType.SYSTEM_ALERT.value,
            'priority_key': 'priority',
            'message_key': 'message'
        })
        
        # Determine priority
        priority_value = alert_data.get(mapping['priority_key'], 'medium')
        priority = self._map_priority(priority_value)
        
        # Extract message
        message_data = alert_data.get(mapping['message_key'], [])
        if isinstance(message_data, list):
            message = '; '.join(message_data)
        else:
            message = str(message_data)
        
        return {
            'source_agent': source_agent,
            'alert_type': mapping['type'],
            'priority': priority,
            'user_id': alert_data.get('user_id', alert_data.get('device_id', 'unknown')),
            'message': message,
            'raw_data': alert_data,
            'location': alert_data.get('location', 'unknown'),
            'timestamp': alert_data.get('timestamp', datetime.now().isoformat())
        }
    
    def _map_priority(self, priority_value: str) -> str:
        """Map priority value to standard priority"""
        priority_mapping = {
            'critical': 'critical',
            'high': 'high',
            'medium': 'medium',
            'low': 'low'
        }
        return priority_mapping.get(priority_value.lower(), 'medium')
    
    def _process_alert(self, alert_id: str, alert_data: Dict):
        """
        Process an alert according to protocols
        
        Args:
            alert_id: Alert ID
            alert_data: Alert data
        """
        alert_type = alert_data['alert_type']
        priority = alert_data['priority']
        
        # Get response protocol
        protocol = self.response_protocols.get(alert_type, {})
        
        # Execute immediate actions
        required_actions = protocol.get('required_actions', [])
        for action in required_actions:
            if action in self.response_actions:
                try:
                    self.response_actions[action](alert_id, alert_data)
                except Exception as e:
                    logger.error(f"Error executing action {action}: {e}")
        
        # Start escalation timer if needed
        if priority in ['critical', 'high']:
            self._start_escalation_timer(alert_id, alert_data)
        
        # Log alert processing
        logger.info(f"Alert {alert_id} processed with priority {priority}")
    
    def _start_escalation_timer(self, alert_id: str, alert_data: Dict):
        """
        Start escalation timer for alert
        
        Args:
            alert_id: Alert ID
            alert_data: Alert data
        """
        priority = alert_data['priority']
        escalation_rule = self.escalation_rules.get(priority, {})
        
        # Store escalation info
        self.active_alerts[alert_id]['escalation'] = {
            'timer_started': datetime.now().isoformat(),
            'escalation_time': escalation_rule.get('escalation_time', 1800),
            'escalation_contacts': escalation_rule.get('escalation_contacts', []),
            'escalated': False
        }
        
        logger.info(f"Escalation timer started for alert {alert_id}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> Dict:
        """
        Acknowledge an alert
        
        Args:
            alert_id: Alert ID
            acknowledged_by: Person acknowledging the alert
            
        Returns:
            Acknowledgment status
        """
        if alert_id not in self.active_alerts:
            return {'status': 'error', 'message': 'Alert not found'}
        
        alert = self.active_alerts[alert_id]
        alert['status'] = AlertStatus.ACKNOWLEDGED.value
        alert['acknowledged_by'] = acknowledged_by
        alert['acknowledged_at'] = datetime.now().isoformat()
        
        # Stop escalation if active
        if 'escalation' in alert:
            alert['escalation']['escalated'] = True
        
        logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        
        return {'status': 'success', 'message': 'Alert acknowledged'}
    
    def resolve_alert(self, alert_id: str, resolved_by: str, resolution_notes: str = "") -> Dict:
        """
        Resolve an alert
        
        Args:
            alert_id: Alert ID
            resolved_by: Person resolving the alert
            resolution_notes: Notes about resolution
            
        Returns:
            Resolution status
        """
        if alert_id not in self.active_alerts:
            return {'status': 'error', 'message': 'Alert not found'}
        
        alert = self.active_alerts[alert_id]
        alert['status'] = AlertStatus.RESOLVED.value
        alert['resolved_by'] = resolved_by
        alert['resolved_at'] = datetime.now().isoformat()
        alert['resolution_notes'] = resolution_notes
        
        # Move to history and remove from active
        self.alert_history.append(alert.copy())
        del self.active_alerts[alert_id]
        
        logger.info(f"Alert {alert_id} resolved by {resolved_by}")
        
        return {'status': 'success', 'message': 'Alert resolved'}
    
    def get_active_alerts(self, user_id: Optional[str] = None, priority: Optional[str] = None) -> List[Dict]:
        """
        Get active alerts
        
        Args:
            user_id: Filter by user ID
            priority: Filter by priority level
            
        Returns:
            List of active alerts
        """
        alerts = list(self.active_alerts.values())
        
        if user_id:
            alerts = [alert for alert in alerts if alert['user_id'] == user_id]
        
        if priority:
            alerts = [alert for alert in alerts if alert['priority'] == priority]
        
        # Sort by priority and timestamp
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        alerts.sort(key=lambda x: (priority_order.get(x['priority'], 4), x['received_at']))
        
        return alerts
    
    def get_alert_dashboard(self) -> Dict:
        """
        Get alert dashboard summary
        
        Returns:
            Dashboard data
        """
        active_alerts = list(self.active_alerts.values())
        
        # Count by priority
        priority_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for alert in active_alerts:
            priority_counts[alert['priority']] += 1
        
        # Count by type
        type_counts = defaultdict(int)
        for alert in active_alerts:
            type_counts[alert['alert_type']] += 1
        
        # Count by user
        user_counts = defaultdict(int)
        for alert in active_alerts:
            user_counts[alert['user_id']] += 1
        
        # Recent alerts (last 24 hours)
        recent_threshold = datetime.now() - timedelta(hours=24)
        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['received_at']) > recent_threshold
        ]
        
        return {
            'active_alerts_count': len(active_alerts),
            'priority_breakdown': priority_counts,
            'type_breakdown': dict(type_counts),
            'user_breakdown': dict(user_counts),
            'recent_alerts_24h': len(recent_alerts),
            'response_time_avg': self._calculate_average_response_time(),
            'resolution_time_avg': self._calculate_average_resolution_time(),
            'escalation_rate': self._calculate_escalation_rate()
        }
    
    def check_escalations(self):
        """Check for alerts that need escalation"""
        current_time = datetime.now()
        
        for alert_id, alert in self.active_alerts.items():
            if 'escalation' in alert and not alert['escalation']['escalated']:
                timer_started = datetime.fromisoformat(alert['escalation']['timer_started'])
                escalation_time = alert['escalation']['escalation_time']
                
                if (current_time - timer_started).total_seconds() > escalation_time:
                    self._escalate_alert(alert_id, alert)
    
    def _escalate_alert(self, alert_id: str, alert: Dict):
        """
        Escalate an alert
        
        Args:
            alert_id: Alert ID
            alert: Alert data
        """
        alert['escalation']['escalated'] = True
        alert['escalation']['escalated_at'] = datetime.now().isoformat()
        alert['status'] = AlertStatus.ESCALATED.value
        
        # Notify escalation contacts
        escalation_contacts = alert['escalation']['escalation_contacts']
        for contact_id in escalation_contacts:
            self._notify_contact(contact_id, alert_id, alert, is_escalation=True)
        
        logger.warning(f"Alert {alert_id} escalated due to no response")
    
    # Response action methods
    def _call_emergency_services(self, alert_id: str, alert_data: Dict):
        """Call emergency services"""
        logger.critical(f"EMERGENCY: Calling emergency services for alert {alert_id}")
        self._notify_contact('emergency_services', alert_id, alert_data, is_emergency=True)
    
    def _notify_primary_caregiver(self, alert_id: str, alert_data: Dict):
        """Notify primary caregiver"""
        logger.info(f"Notifying primary caregiver for alert {alert_id}")
        self._notify_contact('primary_caregiver', alert_id, alert_data)
    
    def _notify_family(self, alert_id: str, alert_data: Dict):
        """Notify family members"""
        logger.info(f"Notifying family for alert {alert_id}")
        self._notify_contact('family_member', alert_id, alert_data)
    
    def _notify_healthcare_provider(self, alert_id: str, alert_data: Dict):
        """Notify healthcare provider"""
        logger.info(f"Notifying healthcare provider for alert {alert_id}")
        self._notify_contact('healthcare_provider', alert_id, alert_data)
    
    def _assess_response(self, alert_id: str, alert_data: Dict):
        """Assess if patient is responding"""
        logger.info(f"Assessing response for alert {alert_id}")
        # This would integrate with sensors/monitoring systems
        pass
    
    def _remind_patient(self, alert_id: str, alert_data: Dict):
        """Send reminder to patient"""
        logger.info(f"Sending reminder to patient for alert {alert_id}")
        # This would integrate with patient notification systems
        pass
    
    def _investigate_issue(self, alert_id: str, alert_data: Dict):
        """Investigate the issue"""
        logger.info(f"Investigating issue for alert {alert_id}")
        # This would trigger additional monitoring or checks
        pass
    
    def _increase_monitoring(self, alert_id: str, alert_data: Dict):
        """Increase monitoring frequency"""
        logger.info(f"Increasing monitoring for alert {alert_id}")
        # This would adjust monitoring parameters
        pass
    
    def _document_incident(self, alert_id: str, alert_data: Dict):
        """Document the incident"""
        logger.info(f"Documenting incident for alert {alert_id}")
        # This would create incident reports
        pass
    
    def _schedule_followup(self, alert_id: str, alert_data: Dict):
        """Schedule follow-up"""
        logger.info(f"Scheduling follow-up for alert {alert_id}")
        # This would schedule follow-up appointments or checks
        pass
    
    def _notify_contact(self, contact_id: str, alert_id: str, alert_data: Dict, 
                       is_escalation: bool = False, is_emergency: bool = False):
        """
        Notify a contact
        
        Args:
            contact_id: Contact ID
            alert_id: Alert ID
            alert_data: Alert data
            is_escalation: Whether this is an escalation
            is_emergency: Whether this is an emergency
        """
        contact = self.contact_registry.get(contact_id)
        if not contact:
            logger.error(f"Contact {contact_id} not found")
            return
        
        # Create notification message
        urgency = "EMERGENCY" if is_emergency else "ESCALATION" if is_escalation else "ALERT"
        message = f"{urgency}: {alert_data['message']} (Alert ID: {alert_id})"
        
        # Log notification (in real implementation, this would send actual notifications)
        logger.info(f"Notifying {contact['name']} ({contact_id}): {message}")
        
        # Record notification
        if 'notifications' not in alert_data:
            alert_data['notifications'] = []
        
        alert_data['notifications'].append({
            'contact_id': contact_id,
            'contact_name': contact['name'],
            'message': message,
            'sent_at': datetime.now().isoformat(),
            'type': urgency.lower()
        })
    
    def _calculate_average_response_time(self) -> float:
        """Calculate average response time"""
        response_times = []
        for alert in self.alert_history:
            if alert.get('acknowledged_at'):
                received = datetime.fromisoformat(alert['received_at'])
                acknowledged = datetime.fromisoformat(alert['acknowledged_at'])
                response_times.append((acknowledged - received).total_seconds())
        
        return float(np.mean(response_times)) if response_times else 0.0
    
    def _calculate_average_resolution_time(self) -> float:
        """Calculate average resolution time"""
        resolution_times = []
        for alert in self.alert_history:
            if alert.get('resolved_at'):
                received = datetime.fromisoformat(alert['received_at'])
                resolved = datetime.fromisoformat(alert['resolved_at'])
                resolution_times.append((resolved - received).total_seconds())
        
        return float(np.mean(resolution_times)) if resolution_times else 0.0
    
    def _calculate_escalation_rate(self) -> float:
        """Calculate escalation rate"""
        if not self.alert_history:
            return 0
        
        escalated_count = sum(1 for alert in self.alert_history 
                            if alert.get('status') == AlertStatus.ESCALATED.value)
        
        return escalated_count / len(self.alert_history)

# Example usage and integration testing
if __name__ == "__main__":
    # Initialize alert agent
    alert_agent = AlertAgent()
    
    print("=== Alert Agent Demo ===")
    
    # Simulate health emergency alert
    print("\n1. Health Emergency Alert")
    health_alert = {
        'user_id': 'D1001',
        'alert_required': True,
        'risk_level': 'critical',
        'recommendations': ['URGENT: Contact emergency services', 'High heart rate detected'],
        'timestamp': datetime.now().isoformat()
    }
    
    alert_id_1 = alert_agent.receive_alert('health_agent', health_alert)
    print(f"Health alert created: {alert_id_1}")
    
    # Simulate fall detection alert
    print("\n2. Fall Detection Alert")
    safety_alert = {
        'user_id': 'D1002',
        'fall_detected': True,
        'emergency_level': 'high',
        'location': 'Bathroom',
        'recommendations': ['Fall detected in bathroom', 'Check on patient immediately'],
        'timestamp': datetime.now().isoformat()
    }
    
    alert_id_2 = alert_agent.receive_alert('safety_agent', safety_alert)
    print(f"Safety alert created: {alert_id_2}")
    
    # Simulate medication compliance alert
    print("\n3. Medication Compliance Alert")
    reminder_alert = {
        'user_id': 'D1001',
        'reminder_type': 'Medication',
        'risk_level': 'high',
        'recommendations': ['Medication missed for 2 hours', 'Contact patient'],
        'timestamp': datetime.now().isoformat()
    }
    
    alert_id_3 = alert_agent.receive_alert('reminder_agent', reminder_alert)
    print(f"Reminder alert created: {alert_id_3}")
    
    # Get dashboard
    print("\n4. Alert Dashboard")
    dashboard = alert_agent.get_alert_dashboard()
    print("Dashboard Summary:")
    for key, value in dashboard.items():
        print(f"  {key}: {value}")
    
    # Get active alerts
    print("\n5. Active Alerts")
    active_alerts = alert_agent.get_active_alerts()
    print(f"Active alerts count: {len(active_alerts)}")
    for alert in active_alerts:
        print(f"  - {alert['alert_id']}: {alert['priority']} - {alert['message'][:50]}...")
    
    # Acknowledge an alert
    print("\n6. Acknowledge Alert")
    ack_result = alert_agent.acknowledge_alert(alert_id_2, "Primary Caregiver")
    print(f"Acknowledgment result: {ack_result}")
    
    # Resolve an alert
    print("\n7. Resolve Alert")
    resolve_result = alert_agent.resolve_alert(alert_id_3, "Primary Caregiver", "Patient contacted and took medication")
    print(f"Resolution result: {resolve_result}")
    
    # Check escalations
    print("\n8. Check Escalations")
    alert_agent.check_escalations()
    
    print("\n=== Alert Agent Demo Complete ===")