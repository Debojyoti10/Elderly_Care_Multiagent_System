from typing import Dict, List, Callable, Any, Optional
from datetime import datetime
import logging
import threading
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EventBus:
    """
    Event Bus for inter-agent communication in the elderly care system
    
    This class provides a centralized message passing system that allows
    different agents (health, safety, reminder, alert) to communicate
    with each other asynchronously.
    """
    
    def __init__(self):
        """Initialize the event bus"""
        self._subscribers = defaultdict(list)
        self._lock = threading.Lock()
        self._event_history = []
        self.max_history = 1000
        
    def subscribe(self, event_type: str, callback: Callable):
        """
        Subscribe to a specific event type
        
        Args:
            event_type: Type of event to listen for
            callback: Function to call when event occurs
        """
        with self._lock:
            self._subscribers[event_type].append(callback)
            logger.info(f"Subscribed to event type: {event_type}")
    
    def unsubscribe(self, event_type: str, callback: Callable):
        """
        Unsubscribe from a specific event type
        
        Args:
            event_type: Type of event to stop listening for
            callback: Function to remove from subscribers
        """
        with self._lock:
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
                logger.info(f"Unsubscribed from event type: {event_type}")
    
    def publish(self, event_type: str, data: Dict[str, Any], source: Optional[str] = None):
        """
        Publish an event to all subscribers
        
        Args:
            event_type: Type of event being published
            data: Event data
            source: Source agent publishing the event
        """
        event = {
            'type': event_type,
            'data': data,
            'source': source,
            'timestamp': datetime.now().isoformat(),
            'id': len(self._event_history)
        }
        
        # Add to history
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self.max_history:
                self._event_history.pop(0)
        
        # Notify subscribers
        subscribers = self._subscribers.get(event_type, [])
        logger.info(f"Publishing event '{event_type}' from {source} to {len(subscribers)} subscribers")
        
        for callback in subscribers:
            try:
                # Run callback in separate thread to avoid blocking
                thread = threading.Thread(
                    target=self._safe_callback,
                    args=(callback, event),
                    daemon=True
                )
                thread.start()
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    def _safe_callback(self, callback: Callable, event: Dict):
        """
        Safely execute callback with error handling
        
        Args:
            callback: Function to execute
            event: Event data
        """
        try:
            callback(event)
        except Exception as e:
            logger.error(f"Error in event callback: {e}")
    
    def get_event_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """
        Get recent event history
        
        Args:
            event_type: Filter by event type (optional)
            limit: Maximum number of events to return
            
        Returns:
            List of recent events
        """
        with self._lock:
            events = self._event_history
            
            if event_type:
                events = [e for e in events if e['type'] == event_type]
            
            return events[-limit:]
    
    def get_subscribers_count(self, event_type: Optional[str] = None) -> Dict[str, int]:
        """
        Get count of subscribers for each event type
        
        Args:
            event_type: Specific event type to check (optional)
            
        Returns:
            Dictionary with subscriber counts
        """
        with self._lock:
            if event_type:
                return {event_type: len(self._subscribers.get(event_type, []))}
            else:
                return {k: len(v) for k, v in self._subscribers.items()}

# Global event bus instance
event_bus = EventBus()

# Event type constants for consistency across agents
class EventTypes:
    """Constants for event types used throughout the system"""
    
    # Health-related events
    HEALTH_ALERT = "health.alert"
    HEALTH_ANOMALY = "health.anomaly" 
    VITAL_SIGNS_UPDATE = "health.vitals_update"
    HEALTH_TREND_CHANGE = "health.trend_change"
    
    # Safety-related events
    FALL_DETECTED = "safety.fall_detected"
    EMERGENCY_DETECTED = "safety.emergency"
    INACTIVITY_ALERT = "safety.inactivity"
    LOCATION_ALERT = "safety.location_alert"
    
    # Reminder-related events
    MEDICATION_DUE = "reminder.medication_due"
    APPOINTMENT_REMINDER = "reminder.appointment"
    ACTIVITY_REMINDER = "reminder.activity"
    REMINDER_MISSED = "reminder.missed"
    
    # Alert-related events
    CAREGIVER_NOTIFIED = "alert.caregiver_notified"
    EMERGENCY_SERVICES_CALLED = "alert.emergency_called"
    FAMILY_NOTIFIED = "alert.family_notified"
    
    # System events
    AGENT_STATUS_UPDATE = "system.agent_status"
    DEVICE_CONNECTION_CHANGE = "system.device_connection"
    SYSTEM_ERROR = "system.error"

# Utility functions for common event patterns
def publish_health_alert(device_id: str, alert_type: str, severity: str, data: Dict):
    """Convenience function to publish health alerts"""
    event_bus.publish(
        EventTypes.HEALTH_ALERT,
        {
            'device_id': device_id,
            'alert_type': alert_type,
            'severity': severity,
            'details': data
        },
        source="health_agent"
    )

def publish_safety_alert(device_id: str, alert_type: str, location: Optional[str] = None, data: Optional[Dict] = None):
    """Convenience function to publish safety alerts"""
    event_bus.publish(
        EventTypes.FALL_DETECTED if alert_type == "fall" else EventTypes.EMERGENCY_DETECTED,
        {
            'device_id': device_id,
            'alert_type': alert_type,
            'location': location,
            'details': data or {}
        },
        source="safety_agent"
    )

def publish_reminder(device_id: str, reminder_type: str, message: str, due_time: str):
    """Convenience function to publish reminders"""
    event_type_map = {
        'medication': EventTypes.MEDICATION_DUE,
        'appointment': EventTypes.APPOINTMENT_REMINDER,
        'activity': EventTypes.ACTIVITY_REMINDER
    }
    
    event_bus.publish(
        event_type_map.get(reminder_type, EventTypes.ACTIVITY_REMINDER),
        {
            'device_id': device_id,
            'reminder_type': reminder_type,
            'message': message,
            'due_time': due_time
        },
        source="reminder_agent"
    )

# Example event handlers
def log_all_events(event: Dict):
    """Example event handler that logs all events"""
    logger.info(f"Event received: {event['type']} from {event['source']} at {event['timestamp']}")

def health_alert_handler(event: Dict):
    """Example handler for health alerts"""
    data = event['data']
    logger.warning(f"Health Alert: {data['alert_type']} for device {data['device_id']} - Severity: {data['severity']}")

# Subscribe to events for logging (can be disabled in production)
event_bus.subscribe("*", log_all_events)  # Note: This would need special handling for wildcard
event_bus.subscribe(EventTypes.HEALTH_ALERT, health_alert_handler)

if __name__ == "__main__":
    # Test the event bus
    print("Testing Event Bus...")
    
    # Test health alert
    publish_health_alert(
        device_id="D1001",
        alert_type="heart_rate_high",
        severity="medium",
        data={'heart_rate': 120, 'threshold': 100}
    )
    
    # Test reminder
    publish_reminder(
        device_id="D1001",
        reminder_type="medication",
        message="Time to take your morning medication",
        due_time="08:00"
    )
    
    # Get event history
    print(f"\nEvent History: {len(event_bus.get_event_history())} events")
    print(f"Subscriber counts: {event_bus.get_subscribers_count()}")