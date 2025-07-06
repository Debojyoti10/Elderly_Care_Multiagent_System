"""
Agents package for the Elderly Care Multi-Agent System

This package contains specialized agents for:
- Health monitoring and analysis
- Safety monitoring and fall detection
- Medication and activity reminders
- Alert management and escalation
"""

from .health_agent import HealthAgent
from .safety_agent import SafetyAgent
from .reminder_agent import ReminderAgent
from .alert_agent import AlertAgent

__all__ = ['HealthAgent', 'SafetyAgent', 'ReminderAgent', 'AlertAgent']
