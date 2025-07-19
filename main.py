#!/usr/bin/env python3
"""
Elderly Care Multi-Agent System - Multi-Page Streamlit Web Application

This is the main Streamlit application for the elderly care monitoring system.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import random
import json
import smtplib
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from elderly_care_system import ElderlyCareSystem

# Streamlit Configuration
st.set_page_config(
    page_title="Elderly Care Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .alert-critical {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .alert-warning {
        background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .alert-normal {
        background: linear-gradient(135deg, #48dbfb 0%, #0abde3 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'system' not in st.session_state:
    st.session_state.system = None
if 'users' not in st.session_state:
    st.session_state.users = {}
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False
if 'fake_data_enabled' not in st.session_state:
    st.session_state.fake_data_enabled = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'CaregiverSetup'
if 'caregiver_email' not in st.session_state:
    st.session_state.caregiver_email = ''
if 'caregiver_phone' not in st.session_state:
    st.session_state.caregiver_phone = ''
if 'setup_complete' not in st.session_state:
    st.session_state.setup_complete = False
if 'selected_patient' not in st.session_state:
    st.session_state.selected_patient = ''
if 'email_configured' not in st.session_state:
    st.session_state.email_configured = False
if 'sender_email' not in st.session_state:
    st.session_state.sender_email = ''
if 'sender_password' not in st.session_state:
    st.session_state.sender_password = ''

def initialize_system():
    """Initialize the elderly care system"""
    try:
        if st.session_state.system is None:
            with st.spinner("Initializing Elderly Care System..."):
                st.session_state.system = ElderlyCareSystem()
                st.success("✅ System initialized successfully!")
        return True
    except Exception as e:
        st.error(f"❌ System initialization failed: {str(e)}")
        return False

def generate_fake_user_data():
    """Generate fake user data for demonstration"""
    # Make sure system is initialized
    if st.session_state.system is None:
        if not initialize_system():
            st.error("Cannot generate fake data without system initialization")
            return
    
    fake_users = [
        {
            'id': 'U001',
            'name': 'Eleanor Johnson',
            'age': 78,
            'conditions': ['Hypertension', 'Diabetes Type 2'],
            'medications': ['Lisinopril', 'Metformin', 'Aspirin'],
            'emergency_contact': {'name': 'Sarah Johnson', 'phone': '+1-555-0123'},
            'risk_level': 'high'  # More likely to have alerts
        },
        {
            'id': 'U002', 
            'name': 'Robert Chen',
            'age': 82,
            'conditions': ['Heart Disease', 'Arthritis'],
            'medications': ['Atorvastatin', 'Ibuprofen'],
            'emergency_contact': {'name': 'Michael Chen', 'phone': '+1-555-0456'},
            'risk_level': 'medium'
        },
        {
            'id': 'U003',
            'name': 'Margaret Smith',
            'age': 75,
            'conditions': ['Osteoporosis', 'Mild Cognitive Impairment'],
            'medications': ['Alendronate', 'Calcium', 'Vitamin D'],
            'emergency_contact': {'name': 'David Smith', 'phone': '+1-555-0789'},
            'risk_level': 'low'
        }
    ]
    
    for user in fake_users:
        if user['id'] not in st.session_state.users:
            # Register user with system
            user_profile = {
                'name': user['name'],
                'age': user['age'],
                'emergency_contacts': {
                    'primary': user['emergency_contact']
                },
                'medical_conditions': user['conditions'],
                'medications': [{'name': med, 'times': ['08:00', '20:00']} for med in user['medications']],
                'reminder_schedule': {
                    'Medication': {
                        'times': ['08:00', '12:00', '18:00', '20:00'],
                        'message': 'Time to take your medication',
                        'days_of_week': [0, 1, 2, 3, 4, 5, 6]
                    }
                }
            }
            
            try:
                result = st.session_state.system.register_user(user['id'], user_profile)
                if result['status'] == 'success':
                    st.session_state.users[user['id']] = user
                    # Generate some historical alerts for demo
                    generate_historical_alerts(user['id'], user['risk_level'])
                else:
                    st.error(f"Failed to register user {user['name']}: {result.get('message', 'Unknown error')}")
            except Exception as e:
                st.error(f"Error registering user {user['name']}: {str(e)}")
                # Still add user to session state for demo purposes
                st.session_state.users[user['id']] = user

def generate_fake_health_data(user_id: str, scenario: str = 'normal'):
    """Generate fake health data for a user"""
    base_data = {
        'U001': {'hr': 72, 'sys': 130, 'dia': 85, 'glucose': 110, 'o2': 97},
        'U002': {'hr': 68, 'sys': 140, 'dia': 90, 'glucose': 95, 'o2': 96},
        'U003': {'hr': 76, 'sys': 125, 'dia': 82, 'glucose': 105, 'o2': 98}
    }
    
    base = base_data.get(user_id, {'hr': 70, 'sys': 120, 'dia': 80, 'glucose': 100, 'o2': 97})
    
    if scenario == 'normal':
        return {
            'heart_rate': base['hr'] + random.randint(-5, 5),
            'systolic_bp': base['sys'] + random.randint(-10, 10),
            'diastolic_bp': base['dia'] + random.randint(-5, 5),
            'glucose': base['glucose'] + random.randint(-15, 15),
            'oxygen_saturation': base['o2'] + random.randint(-2, 2),
            'timestamp': datetime.now().isoformat()
        }
    elif scenario == 'warning':
        return {
            'heart_rate': base['hr'] + random.randint(20, 35),  # Clearly elevated
            'systolic_bp': base['sys'] + random.randint(25, 40),  # Clearly high
            'diastolic_bp': base['dia'] + random.randint(15, 25),
            'glucose': base['glucose'] + random.randint(60, 100),  # Clearly elevated
            'oxygen_saturation': base['o2'] + random.randint(-4, -2),
            'timestamp': datetime.now().isoformat()
        }
    elif scenario == 'critical':
        return {
            'heart_rate': base['hr'] + random.randint(50, 80),  # Very high
            'systolic_bp': base['sys'] + random.randint(60, 100),  # Dangerously high
            'diastolic_bp': base['dia'] + random.randint(30, 50),
            'glucose': base['glucose'] + random.randint(200, 350),  # Dangerously high
            'oxygen_saturation': base['o2'] + random.randint(-10, -6),  # Critically low
            'timestamp': datetime.now().isoformat()
        }
    else:
        # Default to normal if scenario not recognized
        return {
            'heart_rate': base['hr'] + random.randint(-5, 5),
            'systolic_bp': base['sys'] + random.randint(-10, 10),
            'diastolic_bp': base['dia'] + random.randint(-5, 5),
            'glucose': base['glucose'] + random.randint(-15, 15),
            'oxygen_saturation': base['o2'] + random.randint(-2, 2),
            'timestamp': datetime.now().isoformat()
        }

def generate_historical_health_data(user_id: str, scenario: str = 'normal', days: int = 7):
    """Generate historical health data for charts"""
    history = []
    for i in range(days, 0, -1):
        date = datetime.now() - timedelta(days=i)
        # Generate daily readings with some variation
        for reading in range(3):  # 3 readings per day
            time_offset = random.randint(0, 23)
            timestamp = date + timedelta(hours=time_offset)
            
            # Generate health data
            health_data = generate_fake_health_data(user_id, scenario)
            health_data['timestamp'] = timestamp.isoformat()
            history.append(health_data)
    
    return history

def generate_fake_safety_data(user_id: str, scenario: str = 'normal'):
    """Generate fake safety data for a user"""
    activities = ['Walking', 'Sitting', 'Standing', 'Lying Down']
    locations = ['Living Room', 'Bedroom', 'Kitchen', 'Bathroom']
    
    if scenario == 'normal':
        return {
            'movement_activity': random.choice(activities),
            'location': random.choice(locations),
            'impact_force': 'None',
            'post_fall_inactivity': 0,
            'timestamp': datetime.now().isoformat()
        }
    elif scenario == 'warning':
        return {
            'movement_activity': random.choice(['No Movement', 'Sitting']),
            'location': random.choice(locations),
            'impact_force': 'Low',
            'post_fall_inactivity': random.randint(120, 300),
            'timestamp': datetime.now().isoformat()
        }
    elif scenario == 'critical':
        return {
            'movement_activity': 'No Movement',
            'location': random.choice(['Bathroom', 'Kitchen']),
            'impact_force': 'High',
            'post_fall_inactivity': random.randint(600, 1200),
            'timestamp': datetime.now().isoformat()
        }

def create_health_chart(user_id: str, days: int = 7):
    """Create health monitoring chart"""
    # Get the current scenario for this user
    scenario = 'normal'  # Default
    if 'selected_patient' in st.session_state and st.session_state.selected_patient:
        scenario = st.session_state.get('health_scenario', 'normal')
    
    # Generate historical health data
    history = generate_historical_health_data(user_id, scenario, days)
    df = pd.DataFrame(history)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create plotly chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['heart_rate'],
        mode='lines+markers',
        name='Heart Rate',
        line=dict(color='#ff6b6b', width=3),
        hovertemplate='<b>Heart Rate</b><br>%{y} bpm<br>%{x}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['systolic_bp'],
        mode='lines+markers',
        name='Systolic BP',
        line=dict(color='#4ecdc4', width=3),
        hovertemplate='<b>Systolic BP</b><br>%{y} mmHg<br>%{x}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['diastolic_bp'],
        mode='lines+markers',
        name='Diastolic BP',
        line=dict(color='#45b7d1', width=3),
        hovertemplate='<b>Diastolic BP</b><br>%{y} mmHg<br>%{x}<extra></extra>'
    ))
    
    # Add glucose as secondary y-axis
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['glucose'],
        mode='lines+markers',
        name='Glucose',
        line=dict(color='#feca57', width=3),
        yaxis='y2',
        hovertemplate='<b>Glucose</b><br>%{y} mg/dL<br>%{x}<extra></extra>'
    ))
    
    # Normal ranges
    normal_ranges = {
        'heart_rate': (60, 100),
        'systolic_bp': (90, 140),
        'diastolic_bp': (60, 90),
        'glucose': (70, 140)
    }
    
    # Add normal range bands
    fig.add_hrect(y0=normal_ranges['heart_rate'][0], y1=normal_ranges['heart_rate'][1], 
                  fillcolor="green", opacity=0.1, layer="below", line_width=0)
    
    fig.update_layout(
        title=f'Health Monitoring - Last {days} Days',
        xaxis_title='Time',
        yaxis=dict(
            title='Heart Rate (bpm) / Blood Pressure (mmHg)',
            side='left',
            color='white'
        ),
        yaxis2=dict(
            title='Blood Glucose (mg/dL)',
            side='right',
            overlaying='y',
            color='#feca57'
        ),
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def render_caregiver_setup_page():
    """Step 1: Caregiver info page"""
    st.markdown('<h1 class="main-header">🏥 Elderly Care Monitor - Caregiver Setup</h1>', unsafe_allow_html=True)
    st.markdown("Please provide your contact information to receive emergency alerts.")
    email = st.text_input("Email Address", value=st.session_state.caregiver_email, placeholder="caregiver@example.com")
    phone = st.text_input("Phone Number", value=st.session_state.caregiver_phone, placeholder="+1234567890")
    valid_email = validate_email(email) if email else False
    valid_phone = validate_phone(phone) if phone else False
    if email and not valid_email:
        st.error("Please enter a valid email address")
    if phone and not valid_phone:
        st.error("Please enter a valid phone number (10-15 digits)")
    if st.button("Next", type="primary"):
        if valid_email and valid_phone:
            st.session_state.caregiver_email = email
            st.session_state.caregiver_phone = phone
            st.session_state.current_page = 'PatientSelect'
            st.rerun()
        else:
            st.error("Please provide valid email and phone number before continuing")

def render_patient_select_page():
    """Step 2: Patient selection page"""
    st.markdown('<h1 class="main-header">🏥 Select Patient</h1>', unsafe_allow_html=True)
    st.markdown("Select the patient you want to monitor. Enable demo data if you want to see example patients.")
    
    # Initialize system if not already done
    if not initialize_system():
        st.error("Failed to initialize the system. Please refresh the page.")
        return
    
    if not st.session_state.fake_data_enabled:
        if st.button("Enable Demo Patients", type="primary"):
            st.session_state.fake_data_enabled = True
            if st.session_state.system:  # Make sure system is initialized
                generate_fake_user_data()
            st.rerun()
    
    # Generate fake data if enabled but users are empty
    if st.session_state.fake_data_enabled and not st.session_state.users:
        if st.session_state.system:  # Make sure system is initialized
            generate_fake_user_data()
    
    if st.session_state.users:
        patient_options = list(st.session_state.users.keys())
        patient_names = [st.session_state.users[pid]['name'] for pid in patient_options]
        idx = 0
        if st.session_state.selected_patient in patient_options:
            idx = patient_options.index(st.session_state.selected_patient)
        selected = st.selectbox("Select Patient", options=patient_options, format_func=lambda x: st.session_state.users[x]['name'], index=idx)
        if st.button("View Patient Dashboard", type="primary"):
            st.session_state.selected_patient = selected
            st.session_state.setup_complete = True
            st.session_state.current_page = 'Dashboard'
            st.rerun()
    else:
        st.info("No patients available. Enable demo data to see example patients.")

def render_dashboard_page():
    """Render the main dashboard page with improved flow and working alerts"""
    st.markdown('<h1 class="main-header">🏥 Elderly Care Monitor</h1>', unsafe_allow_html=True)
    if not initialize_system():
        st.stop()
    
    # Navigation buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🏠 Dashboard", type="primary"):
            st.session_state.current_page = 'Dashboard'
            st.rerun()
    with col2:
        if st.button("👥 Users"):
            st.session_state.current_page = 'Users'
            st.rerun()
    with col3:
        if st.button("🚨 Alerts"):
            st.session_state.current_page = 'Alerts'
            st.rerun()
    with col4:
        if st.button("⚙️ Settings"):
            st.session_state.current_page = 'Settings'
            st.rerun()
    
    st.markdown("---")
    
    # Sidebar controls
    st.sidebar.title("🎛️ Control Panel")
    st.sidebar.markdown("### 👨‍⚕️ Caregiver Info")
    st.sidebar.info(f"📧 {st.session_state.caregiver_email}")
    st.sidebar.info(f"📱 {st.session_state.caregiver_phone}")
    
    # Fake data toggle
    fake_data = st.sidebar.toggle("Generate Fake Data", value=st.session_state.fake_data_enabled)
    if fake_data != st.session_state.fake_data_enabled:
        st.session_state.fake_data_enabled = fake_data
        if fake_data:
            generate_fake_user_data()
            st.rerun()
    
    # Patient selection (from setup)
    selected_user = st.session_state.selected_patient if st.session_state.selected_patient else None
    scenario = st.sidebar.selectbox("Health Scenario", options=['normal', 'warning', 'critical'], format_func=lambda x: x.title(), index=0)
    
    if selected_user:
        # Manual health checkup
        if st.sidebar.button("🩺 Run Manual Health Checkup", type="primary"):
            with st.spinner("Running health checkup..."):
                health_data = generate_fake_health_data(selected_user, scenario)
                safety_data = generate_fake_safety_data(selected_user, scenario)
                result = st.session_state.system.monitor_user(selected_user, health_data, safety_data)
                # Generate alerts for warning/critical
                user_name = st.session_state.users[selected_user]['name']
                if scenario in ['warning', 'critical']:
                    send_emergency_alerts(user_name, scenario, health_data, safety_data)
                if result.get('alerts_generated'):
                    st.sidebar.success(f"✅ Health check completed - {len(result['alerts_generated'])} alerts generated")
                else:
                    st.sidebar.info("✅ Health check completed - No alerts")
                st.rerun()
        
        # Real-time monitoring toggle
        monitoring_active = st.sidebar.toggle("Real-time Monitoring", value=st.session_state.monitoring_active)
        if monitoring_active != st.session_state.monitoring_active:
            st.session_state.monitoring_active = monitoring_active
            if monitoring_active:
                st.session_state.system.start_continuous_monitoring()
                st.sidebar.success("✅ Monitoring started")
            else:
                st.session_state.system.stop_continuous_monitoring()
                st.sidebar.info("⏹️ Monitoring stopped")
        
        render_user_dashboard(selected_user, scenario)
    else:
        render_overview_dashboard()

def render_users_page():
    """Render the users management page"""
    st.markdown('<h1 class="main-header">👥 Users Management</h1>', unsafe_allow_html=True)
    
    # Navigation buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🏠 Dashboard"):
            st.session_state.current_page = 'Dashboard'
            st.rerun()
    
    with col2:
        if st.button("👥 Users", type="primary"):
            st.session_state.current_page = 'Users'
            st.rerun()
    
    with col3:
        if st.button("🚨 Alerts"):
            st.session_state.current_page = 'Alerts'
            st.rerun()
    
    with col4:
        if st.button("⚙️ Settings"):
            st.session_state.current_page = 'Settings'
            st.rerun()
    
    st.markdown("---")
    
    if not st.session_state.users:
        st.info("No users registered. Enable fake data in the dashboard to see demo users.")
        return
    
    # Users overview
    st.markdown("### 👥 Registered Users")
    
    users_data = []
    for user_id, user in st.session_state.users.items():
        if st.session_state.system:
            dashboard = st.session_state.system.get_user_dashboard(user_id)
            status = dashboard.get('overall_status', {}).get('level', 'normal')
            alerts = len(st.session_state.system.agents['alert'].get_active_alerts(user_id))
        else:
            status = 'normal'
            alerts = 0
        
        users_data.append({
            'ID': user_id,
            'Name': user['name'],
            'Age': user['age'],
            'Status': status.title(),
            'Active Alerts': alerts,
            'Conditions': len(user['conditions']),
            'Medications': len(user['medications']),
            'Emergency Contact': user['emergency_contact']['name']
        })
    
    df = pd.DataFrame(users_data)
    st.dataframe(df, use_container_width=True)
    
    # User details
    st.markdown("### 📋 User Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_user = st.selectbox(
            "Select user for details",
            options=list(st.session_state.users.keys()),
            format_func=lambda x: st.session_state.users[x]['name']
        )
    
    with col2:
        if st.button("🎯 Generate Demo Alerts"):
            if st.session_state.users:
                for user_id, user_data in st.session_state.users.items():
                    generate_historical_alerts(user_id, user_data['risk_level'])
                st.success("Demo alerts generated for all users!")
            else:
                st.warning("No users found. Please add users first.")
    
    if selected_user:
        user = st.session_state.users[selected_user]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👤 Personal Information")
            st.markdown(f"**Name:** {user['name']}")
            st.markdown(f"**Age:** {user['age']}")
            st.markdown(f"**Risk Level:** {user['risk_level'].title()}")
            st.markdown(f"**ID:** {selected_user}")
            
            st.markdown("#### 📞 Emergency Contact")
            st.markdown(f"**Name:** {user['emergency_contact']['name']}")
            st.markdown(f"**Phone:** {user['emergency_contact']['phone']}")
        
        with col2:
            st.markdown("#### 🏥 Medical Information")
            st.markdown("**Conditions:**")
            for condition in user['conditions']:
                st.markdown(f"• {condition}")
            
            st.markdown("**Medications:**")
            for medication in user['medications']:
                st.markdown(f"• {medication}")
        
        # Generate alerts for this specific user
        if st.button(f"🚨 Generate Alerts for {user['name']}"):
            generate_historical_alerts(selected_user, user['risk_level'])
            st.success(f"Demo alerts generated for {user['name']}!")

def render_alerts_page():
    """Render the alerts management page"""
    st.markdown('<h1 class="main-header">🚨 Alerts Management</h1>', unsafe_allow_html=True)
    
    # Navigation buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🏠 Dashboard"):
            st.session_state.current_page = 'Dashboard'
            st.rerun()
    
    with col2:
        if st.button("👥 Users"):
            st.session_state.current_page = 'Users'
            st.rerun()
    
    with col3:
        if st.button("🚨 Alerts", type="primary"):
            st.session_state.current_page = 'Alerts'
            st.rerun()
    
    with col4:
        if st.button("⚙️ Settings"):
            st.session_state.current_page = 'Settings'
            st.rerun()
    
    st.markdown("---")
    
    # Alert filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        alert_status = st.selectbox(
            "Filter by Status",
            ["All", "Active", "Resolved"],
            index=0
        )
    
    with col2:
        alert_type = st.selectbox(
            "Filter by Type",
            ["All", "Health", "Safety", "Reminder", "Email", "SMS"],
            index=0
        )
    
    with col3:
        alert_severity = st.selectbox(
            "Filter by Severity",
            ["All", "High", "Medium", "Low"],
            index=0
        )
    
    st.markdown("---")
    
    # Show alerts
    if 'sent_alerts' in st.session_state and st.session_state.sent_alerts:
        # Filter alerts based on selection
        filtered_alerts = []
        for alert in st.session_state.sent_alerts:
            # Status filter
            if alert_status != "All":
                if alert_status == "Active" and not alert.get('active', False):
                    continue
                elif alert_status == "Resolved" and alert.get('active', False):
                    continue
            
            # Type filter
            if alert_type != "All":
                alert_type_lower = alert_type.lower()
                if (alert_type_lower == "email" and alert.get('type') != 'email') or \
                   (alert_type_lower == "sms" and alert.get('type') != 'sms') or \
                   (alert_type_lower in ["health", "safety", "reminder"] and alert.get('alert_type') != alert_type_lower):
                    continue
            
            # Severity filter
            if alert_severity != "All" and alert.get('severity', '').lower() != alert_severity.lower():
                continue
            
            filtered_alerts.append(alert)
        
        # Display stats
        total_alerts = len(st.session_state.sent_alerts)
        active_alerts = len([a for a in st.session_state.sent_alerts if a.get('active', False)])
        resolved_alerts = total_alerts - active_alerts
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Alerts", total_alerts)
        with col2:
            st.metric("Active Alerts", active_alerts)
        with col3:
            st.metric("Resolved Alerts", resolved_alerts)
        
        st.markdown("---")
        
        # Display filtered alerts
        if filtered_alerts:
            st.markdown(f"### 📋 Showing {len(filtered_alerts)} alerts")
            
            for alert in filtered_alerts:
                timestamp = datetime.fromisoformat(alert['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
                
                # Determine alert styling
                if alert.get('type') == 'email':
                    icon = "📧"
                    alert_class = "alert-normal"
                elif alert.get('type') == 'sms':
                    icon = "📱"
                    alert_class = "alert-normal"
                else:
                    # System alerts
                    severity = alert.get('severity', 'low').lower()
                    if severity == 'high':
                        icon = "🚨"
                        alert_class = "alert-critical"
                    elif severity == 'medium':
                        icon = "⚠️"
                        alert_class = "alert-warning"
                    else:
                        icon = "ℹ️"
                        alert_class = "alert-normal"
                
                # Status badge
                status_badge = "🟢 Active" if alert.get('active', False) else "🔴 Resolved"
                
                st.markdown(f'<div class="{alert_class}">', unsafe_allow_html=True)
                
                if alert.get('type') in ['email', 'sms']:
                    # Communication alerts
                    st.markdown(f"**{icon} {alert['type'].upper()} Alert** - {timestamp} - {status_badge}")
                    st.markdown(f"**To:** {alert['recipient']}")
                    if alert.get('subject'):
                        st.markdown(f"**Subject:** {alert['subject']}")
                    st.markdown(f"**Message:** {alert['message'][:100]}...")
                    if alert.get('status'):
                        st.markdown(f"**Status:** {alert['status']}")
                else:
                    # System alerts
                    st.markdown(f"**{icon} {alert.get('title', 'System Alert')}** - {timestamp} - {status_badge}")
                    if alert.get('user_name'):
                        st.markdown(f"**User:** {alert['user_name']}")
                    st.markdown(f"**Type:** {alert.get('alert_type', 'system').title()}")
                    st.markdown(f"**Severity:** {alert.get('severity', 'low').title()}")
                    st.markdown(f"**Message:** {alert['message']}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")
        else:
            st.info("No alerts match the selected filters.")
    
    # System alerts from the monitoring system
    if st.session_state.system:
        st.markdown("### � Live System Alerts")
        
        live_alerts = st.session_state.system.agents['alert'].get_active_alerts()
        
        if live_alerts:
            for alert in live_alerts:
                priority_color = {
                    'high': 'alert-critical',
                    'medium': 'alert-warning',
                    'low': 'alert-normal'
                }.get(alert['priority'], 'alert-normal')
                
                st.markdown(f'<div class="{priority_color}">', unsafe_allow_html=True)
                st.markdown(f"**Alert ID:** {alert['alert_id']}")
                st.markdown(f"**User:** {alert['user_id']}")
                st.markdown(f"**Type:** {alert['alert_type']}")
                st.markdown(f"**Priority:** {alert['priority'].upper()}")
                st.markdown(f"**Status:** {alert['status']}")
                st.markdown(f"**Message:** {alert['message']}")
                st.markdown(f"**Received:** {alert['received_at']}")
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")
        else:
            st.info("No active system alerts")

def render_settings_page():
    """Render the settings page"""
    st.markdown('<h1 class="main-header">⚙️ Settings</h1>', unsafe_allow_html=True)
    
    # Navigation buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🏠 Dashboard"):
            st.session_state.current_page = 'Dashboard'
            st.rerun()
    
    with col2:
        if st.button("👥 Users"):
            st.session_state.current_page = 'Users'
            st.rerun()
    
    with col3:
        if st.button("🚨 Alerts"):
            st.session_state.current_page = 'Alerts'
            st.rerun()
    
    with col4:
        if st.button("⚙️ Settings", type="primary"):
            st.session_state.current_page = 'Settings'
            st.rerun()
    
    st.markdown("---")
    
    # Caregiver settings
    st.markdown("### 👨‍⚕️ Caregiver Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_email = st.text_input(
            "Email Address",
            value=st.session_state.caregiver_email,
            help="Email address for receiving alerts"
        )
        
        if new_email and not validate_email(new_email):
            st.error("Please enter a valid email address")
    
    with col2:
        new_phone = st.text_input(
            "Phone Number",
            value=st.session_state.caregiver_phone,
            help="Phone number for SMS alerts"
        )
        
        if new_phone and not validate_phone(new_phone):
            st.error("Please enter a valid phone number")
    
    if st.button("💾 Save Changes"):
        if new_email and new_phone and validate_email(new_email) and validate_phone(new_phone):
            st.session_state.caregiver_email = new_email
            st.session_state.caregiver_phone = new_phone
            st.success("✅ Settings saved successfully!")
        else:
            st.error("Please provide valid email and phone number")
    
    # Test alerts
    st.markdown("### 🧪 Test Alerts")
    
    if st.button("📧 Send Test Email"):
        if st.session_state.caregiver_email:
            send_email_alert(
                st.session_state.caregiver_email,
                "🧪 Test Alert - Elderly Care Monitor",
                "This is a test email from the Elderly Care Monitor system. Your email notifications are working correctly!"
            )
            st.success("Test email sent!")
        else:
            st.error("Please set your email address first")
    
    if st.button("📱 Send Test SMS"):
        if st.session_state.caregiver_phone:
            send_sms_alert(
                st.session_state.caregiver_phone,
                "🧪 Test SMS from Elderly Care Monitor. Your SMS notifications are working correctly!"
            )
            st.success("Test SMS sent!")
        else:
            st.error("Please set your phone number first")
    
    # Email configuration
    st.markdown("### 📧 Email Configuration")
    
    if st.session_state.get('email_configured', False):
        st.success("✅ Email is configured and working!")
        st.info(f"Sending emails from: {st.session_state.get('sender_email', 'Not set')}")
        
        if st.button("🔄 Reconfigure Email"):
            st.session_state.email_configured = False
            st.session_state.sender_email = ""
            st.session_state.sender_password = ""
            st.rerun()
    else:
        st.warning("⚠️ Email not configured. Alerts will be simulated.")
        
        with st.expander("📧 Configure Real Email Sending"):
            st.info("""
            To send real emails, you need to:
            1. Use a Gmail account
            2. Enable 2-factor authentication
            3. Generate an App Password
            4. Enter your credentials below
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                config_email = st.text_input(
                    "Gmail Address",
                    placeholder="youremail@gmail.com",
                    help="Gmail account that will send alerts"
                )
            
            with col2:
                config_password = st.text_input(
                    "Gmail App Password",
                    type="password",
                    placeholder="xxxx xxxx xxxx xxxx",
                    help="16-character app password from Google"
                )
            
            if config_email and config_password:
                if st.button("🧪 Test & Save Email Configuration"):
                    if test_email_config(config_email, config_password, st.session_state.caregiver_email):
                        st.success("✅ Email configuration working!")
                        st.session_state.email_configured = True
                        st.session_state.sender_email = config_email
                        st.session_state.sender_password = config_password
                        st.rerun()
                    else:
                        st.error("❌ Email configuration failed. Check credentials.")
    
    # System settings
    st.markdown("### 🔧 System Settings")
    
    if st.button("🗑️ Clear All Alerts"):
        if 'sent_alerts' in st.session_state:
            st.session_state.sent_alerts = []
            st.success("All alerts cleared!")
    
    if st.button("🔄 Reset System"):
        st.session_state.system = None
        st.session_state.users = {}
        st.session_state.monitoring_active = False
        st.session_state.fake_data_enabled = False
        if 'sent_alerts' in st.session_state:
            st.session_state.sent_alerts = []
        st.success("System reset!")
        st.rerun()

def setup_real_email_sending():
    """Configure real email sending with user's Gmail account"""
    st.markdown("### 📧 Email Configuration")
    st.markdown("""
    To send real emails, you can use Gmail's SMTP. Follow these steps:
    
    1. **Enable 2-Factor Authentication** on your Gmail account
    2. **Create an App Password**:
       - Go to Google Account settings
       - Security → 2-Step Verification → App passwords
       - Generate a password for "Mail"
    3. **Enter your credentials below**:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        sender_email = st.text_input(
            "Your Gmail Address",
            placeholder="youremail@gmail.com",
            help="Gmail account that will send alerts"
        )
    
    with col2:
        sender_password = st.text_input(
            "Gmail App Password",
            type="password",
            placeholder="xxxx xxxx xxxx xxxx",
            help="16-character app password from Google"
        )
    
    if sender_email and sender_password:
        if st.button("🧪 Test Email Configuration"):
            if test_email_config(sender_email, sender_password, st.session_state.caregiver_email):
                st.success("✅ Email configuration working!")
                st.session_state.email_configured = True
                st.session_state.sender_email = sender_email
                st.session_state.sender_password = sender_password
            else:
                st.error("❌ Email configuration failed. Check credentials.")
    
    return sender_email, sender_password

def test_email_config(sender_email: str, sender_password: str, recipient_email: str) -> bool:
    """Test email configuration by sending a test email"""
    try:
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = "🧪 Elderly Care Monitor - Email Test"
        
        test_message = f"""
Hello!

This is a test email from the Elderly Care Monitor system.

If you're receiving this email, your email configuration is working correctly!

Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Best regards,
Elderly Care Monitor System
        """
        
        msg.attach(MIMEText(test_message, 'plain'))
        
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        
        return True
    except Exception as e:
        print(f"Email test failed: {e}")
        return False

def render_main_app():
    """Render the main application with improved step-by-step flow"""
    if st.session_state.current_page == 'CaregiverSetup':
        render_caregiver_setup_page()
    elif st.session_state.current_page == 'PatientSelect':
        render_patient_select_page()
    elif st.session_state.current_page == 'Dashboard':
        if not st.session_state.selected_patient:
            st.session_state.current_page = 'PatientSelect'
            st.rerun()
        render_dashboard_page()
    elif st.session_state.current_page == 'Users':
        render_users_page()
    elif st.session_state.current_page == 'Alerts':
        render_alerts_page()
    elif st.session_state.current_page == 'Settings':
        render_settings_page()
    else:
        render_caregiver_setup_page()

def render_user_dashboard(user_id: str, scenario: str):
    """Render dashboard for a specific user"""
    user = st.session_state.users[user_id]
    
    # User header
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"### 👤 {user['name']}")
        st.markdown(f"**Age:** {user['age']} | **ID:** {user_id}")
    
    with col2:
        st.markdown("### 📞 Emergency Contact")
        st.markdown(f"**{user['emergency_contact']['name']}**")
        st.markdown(f"{user['emergency_contact']['phone']}")
    
    with col3:
        # Overall status indicator based on current scenario
        if scenario == 'critical':
            st.markdown('<div class="alert-critical"><h3>🚨 CRITICAL</h3></div>', unsafe_allow_html=True)
        elif scenario == 'warning':
            st.markdown('<div class="alert-warning"><h3>⚠️ WARNING</h3></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-normal"><h3>✅ NORMAL</h3></div>', unsafe_allow_html=True)
    
    # Current metrics
    st.markdown("### 📊 Current Metrics")
    
    # Generate current data
    current_health = generate_fake_health_data(user_id, scenario)
    current_safety = generate_fake_safety_data(user_id, scenario)
    
    # Ensure data is not None
    if current_health is None:
        current_health = {
            'heart_rate': 70,
            'systolic_bp': 120,
            'diastolic_bp': 80,
            'glucose': 100,
            'oxygen_saturation': 97
        }
    
    if current_safety is None:
        current_safety = {
            'location': 'Living Room',
            'movement_activity': 'Walking'
        }
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        hr_color = "normal" if 60 <= current_health['heart_rate'] <= 100 else "inverse"
        st.metric(
            "❤️ Heart Rate",
            f"{current_health['heart_rate']} bpm",
            delta=f"{random.randint(-5, 5)} bpm",
            delta_color=hr_color
        )
    
    with col2:
        bp_color = "normal" if current_health['systolic_bp'] <= 140 else "inverse"
        st.metric(
            "🩺 Blood Pressure",
            f"{current_health['systolic_bp']}/{current_health['diastolic_bp']}",
            delta=f"{random.randint(-10, 10)} mmHg",
            delta_color=bp_color
        )
    
    with col3:
        glucose_color = "normal" if current_health['glucose'] <= 140 else "inverse"
        st.metric(
            "🍯 Glucose",
            f"{current_health['glucose']} mg/dL",
            delta=f"{random.randint(-15, 15)} mg/dL",
            delta_color=glucose_color
        )
    
    with col4:
        o2_color = "normal" if current_health['oxygen_saturation'] >= 95 else "inverse"
        st.metric(
            "🫁 Oxygen Saturation",
            f"{current_health['oxygen_saturation']}%",
            delta=f"{random.randint(-2, 2)}%",
            delta_color=o2_color
        )
    
    with col5:
        st.metric(
            "📍 Location",
            current_safety['location'],
            delta=current_safety['movement_activity']
        )
    
    # Health trends chart
    st.markdown("### 📈 Health Trends")
    health_chart = create_health_chart(user_id)
    st.plotly_chart(health_chart, use_container_width=True)
    
    # Alerts and reminders section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚨 Active Alerts")
        if 'sent_alerts' in st.session_state and st.session_state.sent_alerts:
            user_alerts = [alert for alert in st.session_state.sent_alerts 
                          if alert.get('user_id') == user_id and alert.get('active', False)]
            if user_alerts:
                for alert in user_alerts[-5:]:  # Show last 5 alerts
                    severity = alert.get('severity', 'low')
                    if severity == 'high':
                        icon = "🔴"
                    elif severity == 'medium':
                        icon = "🟡"
                    else:
                        icon = "🟢"
                    st.markdown(f"{icon} **{alert.get('alert_type', 'alert').title()}** - {alert['message'][:50]}...")
            else:
                st.info("No active alerts")
        else:
            st.info("No active alerts")
    
    with col2:
        st.markdown("### 💊 Today's Medications")
        user = st.session_state.users[user_id]
        if user['medications']:
            for medication in user['medications']:
                st.markdown(f"💊 **{medication}** - Take with meals")
        else:
            st.info("No medications scheduled")
        
        # Show scenario-based health status
        st.markdown("### 📊 Current Health Status")
        if scenario == 'critical':
            st.error("🚨 **CRITICAL** - Immediate medical attention required!")
            st.markdown("- Heart rate critically elevated")
            st.markdown("- Blood pressure dangerously high") 
            st.markdown("- Blood glucose severely elevated")
        elif scenario == 'warning':
            st.warning("⚠️ **WARNING** - Health metrics need attention")
            st.markdown("- Heart rate above normal range")
            st.markdown("- Blood pressure elevated")
            st.markdown("- Blood glucose trending high")
        else:
            st.success("✅ **NORMAL** - All metrics within healthy ranges")
            st.markdown("- Heart rate normal")
            st.markdown("- Blood pressure stable")
            st.markdown("- Blood glucose controlled")

def render_overview_dashboard():
    """Render overview dashboard when no patient is selected"""
    st.markdown("### 🏠 System Overview")
    
    if not st.session_state.users:
        st.info("👋 Welcome to the Elderly Care Monitor! Enable 'Generate Fake Data' in the sidebar to see demo users.")
        
        # Quick start guide
        st.markdown("### 🚀 Quick Start Guide")
        st.markdown("""
        1. **Enable Fake Data** - Toggle the switch in the sidebar to load demo users
        2. **Select a User** - Choose a user from the dropdown to view their dashboard
        3. **Monitor Health** - Use different scenarios (normal, warning, critical) to simulate health events
        4. **View Real-time Data** - Enable real-time monitoring to see continuous updates
        5. **Check Alerts** - Monitor the alerts panel for any health concerns
        """)
    else:
        # System statistics
        total_users = len(st.session_state.users)
        total_alerts = 0
        critical_users = 0
        
        for user_id in st.session_state.users:
            if st.session_state.system:
                alerts = st.session_state.system.agents['alert'].get_active_alerts(user_id)
                total_alerts += len(alerts)
                
                dashboard = st.session_state.system.get_user_dashboard(user_id)
                if 'overall_status' in dashboard and dashboard['overall_status']['level'] == 'critical':
                    critical_users += 1
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("👥 Total Users", total_users)
        
        with col2:
            st.metric("🚨 Active Alerts", total_alerts)
        
        with col3:
            st.metric("🔴 Critical Users", critical_users)
        
        with col4:
            st.metric("⚡ System Status", "Online", delta="Active")

def validate_email(email: str) -> bool:
    """Validate email address format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_phone(phone: str) -> bool:
    """Validate phone number format"""
    # Remove all non-digit characters
    digits = re.sub(r'\D', '', phone)
    # Check if it's a valid length (10-15 digits)
    return len(digits) >= 10 and len(digits) <= 15

def send_email_alert(recipient_email: str, subject: str, message: str) -> bool:
    """Send real email alert using Gmail SMTP"""
    try:
        # Check if email is configured
        if ('sender_email' not in st.session_state or 
            'sender_password' not in st.session_state or
            not st.session_state.get('email_configured', False)):
            
            # Use demo mode - simulate email sending
            print(f"📧 EMAIL ALERT (DEMO MODE)!")
            print(f"To: {recipient_email}")
            print(f"Subject: {subject}")
            print(f"Message Preview: {message[:100]}...")
            print("-" * 50)
            
            # Store the alert in session state for display
            if 'sent_alerts' not in st.session_state:
                st.session_state.sent_alerts = []
            
            st.session_state.sent_alerts.append({
                'type': 'email',
                'recipient': recipient_email,
                'subject': subject,
                'message': message,
                'timestamp': datetime.now().isoformat(),
                'status': 'demo_sent'
            })
            
            st.toast(f"📧 Email simulated to {recipient_email}!", icon="✅")
            return True
        
        # Real email sending
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        
        sender_email = st.session_state.sender_email
        sender_password = st.session_state.sender_password
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        
        # Add body to email
        msg.attach(MIMEText(message, 'plain'))
        
        # Create SMTP session
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        
        print(f"📧 REAL EMAIL SENT!")
        print(f"From: {sender_email}")
        print(f"To: {recipient_email}")
        print(f"Subject: {subject}")
        print("-" * 50)
        
        # Store the alert in session state for display
        if 'sent_alerts' not in st.session_state:
            st.session_state.sent_alerts = []
        
        st.session_state.sent_alerts.append({
            'type': 'email',
            'recipient': recipient_email,
            'subject': subject,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'status': 'sent'
        })
        
        st.toast(f"📧 Real email sent to {recipient_email}!", icon="✅")
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        error_msg = "Email authentication failed. Please check your Gmail credentials and app password."
        print(f"SMTP Authentication Error: {e}")
        
        # Store failed alert
        if 'sent_alerts' not in st.session_state:
            st.session_state.sent_alerts = []
        
        st.session_state.sent_alerts.append({
            'type': 'email',
            'recipient': recipient_email,
            'subject': subject,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'status': 'failed',
            'error': error_msg
        })
        
        st.toast(f"❌ {error_msg}", icon="🚨")
        return False
        
    except Exception as e:
        error_msg = f"Failed to send email: {str(e)}"
        print(f"Email sending error: {e}")
        
        # Store failed alert
        if 'sent_alerts' not in st.session_state:
            st.session_state.sent_alerts = []
        
        st.session_state.sent_alerts.append({
            'type': 'email',
            'recipient': recipient_email,
            'subject': subject,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'status': 'failed',
            'error': error_msg
        })
        
        st.toast(f"❌ {error_msg}", icon="🚨")
        return False

def send_sms_alert(recipient_phone: str, message: str) -> bool:
    """Send SMS alert (simulation for demo)"""
    try:
        # In a real application, you would use a service like Twilio
        # For demo purposes, we'll simulate the SMS sending
        print(f"📱 SMS ALERT SENT!")
        print(f"To: {recipient_phone}")
        print(f"Message: {message}")
        print("-" * 50)
        
        # Store the alert in session state for display
        if 'sent_alerts' not in st.session_state:
            st.session_state.sent_alerts = []
        
        st.session_state.sent_alerts.append({
            'type': 'sms',
            'recipient': recipient_phone,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
        
        return True
    except Exception as e:
        print(f"Failed to send SMS: {e}")
        return False

def send_emergency_alerts(user_name: str, alert_type: str, health_data: Optional[Dict], safety_data: Optional[Dict]):
    """Send emergency alerts via email and SMS"""
    if not st.session_state.caregiver_email or not st.session_state.caregiver_phone:
        return
    
    # Ensure we have data to work with
    if health_data is None:
        health_data = {}
    if safety_data is None:
        safety_data = {}
    
    # Create alert message
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if alert_type == 'critical':
        subject = f"🚨 CRITICAL ALERT - {user_name}"
        message = f"""
CRITICAL HEALTH ALERT

Patient: {user_name}
Time: {timestamp}
Status: CRITICAL - Immediate attention required

Health Metrics:
- Heart Rate: {health_data.get('heart_rate', 'N/A')} bpm
- Blood Pressure: {health_data.get('systolic_bp', 'N/A')}/{health_data.get('diastolic_bp', 'N/A')} mmHg
- Glucose: {health_data.get('glucose', 'N/A')} mg/dL
- Oxygen Saturation: {health_data.get('oxygen_saturation', 'N/A')}%

Safety Status:
- Location: {safety_data.get('location', 'N/A')}
- Activity: {safety_data.get('movement_activity', 'N/A')}
- Impact Force: {safety_data.get('impact_force', 'N/A')}

IMMEDIATE ACTION REQUIRED!
Please check on the patient immediately.
        """
    elif alert_type == 'warning':
        subject = f"⚠️ WARNING ALERT - {user_name}"
        message = f"""
HEALTH WARNING ALERT

Patient: {user_name}
Time: {timestamp}
Status: WARNING - Attention needed

Health Metrics:
- Heart Rate: {health_data.get('heart_rate', 'N/A')} bpm
- Blood Pressure: {health_data.get('systolic_bp', 'N/A')}/{health_data.get('diastolic_bp', 'N/A')} mmHg
- Glucose: {health_data.get('glucose', 'N/A')} mg/dL
- Oxygen Saturation: {health_data.get('oxygen_saturation', 'N/A')}%

Please check on the patient when convenient.
        """
    else:
        return  # No alert needed for normal status
    
    # Send email alert
    send_email_alert(st.session_state.caregiver_email, subject, message)
    
    # Send SMS alert (shorter message)
    sms_message = f"ALERT: {user_name} - {alert_type.upper()} health status detected at {timestamp}. Please check immediately."
    send_sms_alert(st.session_state.caregiver_phone, sms_message)

def generate_historical_alerts(user_id: str, risk_level: str):
    """Generate historical alerts for demo purposes"""
    if 'sent_alerts' not in st.session_state:
        st.session_state.sent_alerts = []
    
    # Get user info
    user = st.session_state.users.get(user_id, {})
    user_name = user.get('name', f'User {user_id}')
    
    # Generate alerts based on risk level
    current_time = datetime.now()
    
    # Create a mix of active and inactive alerts
    alert_scenarios = []
    
    if risk_level == 'high':
        # High risk users get more alerts
        alert_scenarios = [
            # Recent active alerts
            {
                'type': 'health',
                'severity': 'high',
                'title': f'🚨 Critical Health Alert - {user_name}',
                'message': f'Abnormal heart rate detected: 45 BPM (critically low). Immediate medical attention required.',
                'hours_ago': 0.5,
                'active': True
            },
            {
                'type': 'safety',
                'severity': 'high', 
                'title': f'⚠️ Fall Detection Alert - {user_name}',
                'message': f'Potential fall detected in living room. No movement detected for 10 minutes.',
                'hours_ago': 2,
                'active': True
            },
            {
                'type': 'health',
                'severity': 'medium',
                'title': f'📊 Blood Pressure Alert - {user_name}',
                'message': f'Blood pressure reading: 180/95 mmHg (hypertensive). Monitor closely.',
                'hours_ago': 4,
                'active': True
            },
            # Older inactive alerts
            {
                'type': 'reminder',
                'severity': 'low',
                'title': f'💊 Medication Reminder - {user_name}',
                'message': f'Evening medication reminder acknowledged.',
                'hours_ago': 24,
                'active': False
            },
            {
                'type': 'safety',
                'severity': 'medium',
                'title': f'🏠 Safety Check - {user_name}',
                'message': f'Unusual activity pattern detected. Issue resolved.',
                'hours_ago': 48,
                'active': False
            }
        ]
    elif risk_level == 'medium':
        # Medium risk users get moderate alerts
        alert_scenarios = [
            {
                'type': 'health',
                'severity': 'medium',
                'title': f'📈 Health Trend Alert - {user_name}',
                'message': f'Blood glucose levels trending upward: 140 mg/dL. Consider dietary review.',
                'hours_ago': 1,
                'active': True
            },
            {
                'type': 'reminder',
                'severity': 'low',
                'title': f'💊 Medication Reminder - {user_name}',
                'message': f'Morning medication reminder - please take medications.',
                'hours_ago': 6,
                'active': True
            },
            {
                'type': 'safety',
                'severity': 'low',
                'title': f'🚶 Activity Alert - {user_name}',
                'message': f'Low activity detected today. Encourage light exercise.',
                'hours_ago': 12,
                'active': True
            },
            # Inactive alerts
            {
                'type': 'health',
                'severity': 'low',
                'title': f'🌡️ Temperature Alert - {user_name}',
                'message': f'Slightly elevated temperature: 99.2°F. Monitored and resolved.',
                'hours_ago': 36,
                'active': False
            },
            {
                'type': 'reminder',
                'severity': 'low',
                'title': f'💊 Medication Reminder - {user_name}',
                'message': f'Evening medication reminder completed.',
                'hours_ago': 72,
                'active': False
            }
        ]
    else:  # low risk
        # Low risk users get fewer alerts
        alert_scenarios = [
            {
                'type': 'reminder',
                'severity': 'low',
                'title': f'💊 Medication Reminder - {user_name}',
                'message': f'Daily medication reminder - vitamins and supplements.',
                'hours_ago': 2,
                'active': True
            },
            {
                'type': 'health',
                'severity': 'low',
                'title': f'📊 Weekly Health Summary - {user_name}',
                'message': f'All health metrics within normal ranges. Keep up the good work!',
                'hours_ago': 24,
                'active': False
            },
            {
                'type': 'safety',
                'severity': 'low',
                'title': f'🏠 Routine Check - {user_name}',
                'message': f'Daily safety check completed successfully.',
                'hours_ago': 48,
                'active': False
            }
        ]
    
    # Create alert records
    for scenario in alert_scenarios:
        alert_time = current_time - timedelta(hours=scenario['hours_ago'])
        
        # Create alert record
        alert = {
            'type': 'system',
            'user_id': user_id,
            'user_name': user_name,
            'alert_type': scenario['type'],
            'severity': scenario['severity'],
            'title': scenario['title'],
            'message': scenario['message'],
            'timestamp': alert_time.isoformat(),
            'active': scenario['active'],
            'status': 'active' if scenario['active'] else 'resolved'
        }
        
        st.session_state.sent_alerts.append(alert)
    
    # Sort alerts by timestamp (newest first)
    st.session_state.sent_alerts.sort(key=lambda x: x['timestamp'], reverse=True)

def generate_health_alert(user_id: str, alert_type: str, severity: str, health_data: Optional[dict]):
    """Generate a health alert based on monitoring data"""
    if 'sent_alerts' not in st.session_state:
        st.session_state.sent_alerts = []
    
    if not health_data:
        health_data = {}
    
    user = st.session_state.users.get(user_id, {})
    user_name = user.get('name', f'User {user_id}')
    
    # Create alert message based on type and severity
    if alert_type == 'heart_rate':
        if severity == 'high':
            title = f"🚨 Critical Heart Rate Alert - {user_name}"
            message = f"Heart rate: {health_data.get('heart_rate', 'N/A')} BPM. Critical level detected!"
        elif severity == 'medium':
            title = f"⚠️ Heart Rate Warning - {user_name}"
            message = f"Heart rate: {health_data.get('heart_rate', 'N/A')} BPM. Above normal range."
        else:
            title = f"ℹ️ Heart Rate Notice - {user_name}"
            message = f"Heart rate: {health_data.get('heart_rate', 'N/A')} BPM. Slightly elevated."
    
    elif alert_type == 'blood_pressure':
        if severity == 'high':
            title = f"🚨 Critical Blood Pressure Alert - {user_name}"
            message = f"Blood pressure: {health_data.get('systolic_bp', 'N/A')}/{health_data.get('diastolic_bp', 'N/A')} mmHg. Hypertensive crisis!"
        elif severity == 'medium':
            title = f"⚠️ Blood Pressure Warning - {user_name}"
            message = f"Blood pressure: {health_data.get('systolic_bp', 'N/A')}/{health_data.get('diastolic_bp', 'N/A')} mmHg. High blood pressure."
        else:
            title = f"ℹ️ Blood Pressure Notice - {user_name}"
            message = f"Blood pressure: {health_data.get('systolic_bp', 'N/A')}/{health_data.get('diastolic_bp', 'N/A')} mmHg. Slightly elevated."
    
    elif alert_type == 'glucose':
        if severity == 'high':
            title = f"🚨 Critical Glucose Alert - {user_name}"
            message = f"Blood glucose: {health_data.get('glucose', 'N/A')} mg/dL. Dangerously high!"
        elif severity == 'medium':
            title = f"⚠️ Glucose Warning - {user_name}"
            message = f"Blood glucose: {health_data.get('glucose', 'N/A')} mg/dL. Above target range."
        else:
            title = f"ℹ️ Glucose Notice - {user_name}"
            message = f"Blood glucose: {health_data.get('glucose', 'N/A')} mg/dL. Slightly elevated."
    
    elif alert_type == 'oxygen_saturation':
        if severity == 'high':
            title = f"🚨 Critical Oxygen Alert - {user_name}"
            message = f"Oxygen saturation: {health_data.get('oxygen_saturation', 'N/A')}%. Critically low!"
        elif severity == 'medium':
            title = f"⚠️ Oxygen Warning - {user_name}"
            message = f"Oxygen saturation: {health_data.get('oxygen_saturation', 'N/A')}%. Below normal range."
        else:
            title = f"ℹ️ Oxygen Notice - {user_name}"
            message = f"Oxygen saturation: {health_data.get('oxygen_saturation', 'N/A')}%. Slightly low."
    
    else:
        title = f"🔔 Health Alert - {user_name}"
        message = f"Health monitoring alert: {alert_type.replace('_', ' ').title()}"
    
    # Create alert record
    alert = {
        'type': 'system',
        'user_id': user_id,
        'user_name': user_name,
        'alert_type': 'health',
        'severity': severity,
        'title': title,
        'message': message,
        'timestamp': datetime.now().isoformat(),
        'active': True,
        'status': 'active'
    }
    
    st.session_state.sent_alerts.append(alert)
    
    # Send emergency alert if severity is high
    if severity == 'high':
        send_emergency_alerts(user_name, 'critical', health_data, {})
    elif severity == 'medium':
        send_emergency_alerts(user_name, 'warning', health_data, {})

def main():
    """Main Streamlit application"""
    render_main_app()

if __name__ == "__main__":
    main()