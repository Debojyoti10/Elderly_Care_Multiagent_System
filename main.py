#!/usr/bin/env python3
"""
Elderly Care Multi-Agent System - Streamlit Web Application

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
    fake_users = [
        {
            'id': 'U001',
            'name': 'Eleanor Johnson',
            'age': 78,
            'conditions': ['Hypertension', 'Diabetes Type 2'],
            'medications': ['Lisinopril', 'Metformin', 'Aspirin'],
            'emergency_contact': {'name': 'Sarah Johnson', 'phone': '+1-555-0123'}
        },
        {
            'id': 'U002', 
            'name': 'Robert Chen',
            'age': 82,
            'conditions': ['Heart Disease', 'Arthritis'],
            'medications': ['Atorvastatin', 'Ibuprofen'],
            'emergency_contact': {'name': 'Michael Chen', 'phone': '+1-555-0456'}
        },
        {
            'id': 'U003',
            'name': 'Margaret Smith',
            'age': 75,
            'conditions': ['Osteoporosis', 'Mild Cognitive Impairment'],
            'medications': ['Alendronate', 'Calcium', 'Vitamin D'],
            'emergency_contact': {'name': 'David Smith', 'phone': '+1-555-0789'}
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
            
            if st.session_state.system:
                result = st.session_state.system.register_user(user['id'], user_profile)
                if result['status'] == 'success':
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
            'heart_rate': base['hr'] + random.randint(15, 25),
            'systolic_bp': base['sys'] + random.randint(20, 35),
            'diastolic_bp': base['dia'] + random.randint(10, 15),
            'glucose': base['glucose'] + random.randint(50, 80),
            'oxygen_saturation': base['o2'] + random.randint(-5, -2),
            'timestamp': datetime.now().isoformat()
        }
    elif scenario == 'critical':
        return {
            'heart_rate': base['hr'] + random.randint(40, 60),
            'systolic_bp': base['sys'] + random.randint(50, 80),
            'diastolic_bp': base['dia'] + random.randint(20, 30),
            'glucose': base['glucose'] + random.randint(150, 250),
            'oxygen_saturation': base['o2'] + random.randint(-8, -5),
            'timestamp': datetime.now().isoformat()
        }

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
    # Generate fake historical data
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                         end=datetime.now(), freq='H')
    
    base_hr = 70 + random.randint(-5, 5)
    base_bp = 120 + random.randint(-10, 10)
    
    health_data = []
    for date in dates:
        health_data.append({
            'timestamp': date,
            'heart_rate': base_hr + random.randint(-10, 15),
            'systolic_bp': base_bp + random.randint(-15, 20),
            'diastolic_bp': 80 + random.randint(-10, 15)
        })
    
    df = pd.DataFrame(health_data)
    
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
    
    fig.update_layout(
        title=f'Health Monitoring - {user_id}',
        xaxis_title='Time',
        yaxis_title='Value',
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

def render_dashboard():
    """Render the main dashboard"""
    st.markdown('<h1 class="main-header">🏥 Elderly Care Monitor</h1>', unsafe_allow_html=True)
    
    if not initialize_system():
        st.stop()
    
    # Generate fake data if enabled
    if st.session_state.fake_data_enabled:
        generate_fake_user_data()
    
    # Sidebar controls
    st.sidebar.title("🎛️ Control Panel")
    
    # Fake data toggle
    fake_data = st.sidebar.toggle("Generate Fake Data", value=st.session_state.fake_data_enabled)
    if fake_data != st.session_state.fake_data_enabled:
        st.session_state.fake_data_enabled = fake_data
        if fake_data:
            generate_fake_user_data()
            st.rerun()
    
    # User selection
    selected_user = None
    scenario = 'normal'  # Default scenario
    
    if st.session_state.users:
        selected_user = st.sidebar.selectbox(
            "Select User",
            options=list(st.session_state.users.keys()),
            format_func=lambda x: st.session_state.users[x]['name']
        )
    else:
        selected_user = None
        st.sidebar.info("No users registered. Enable fake data to see demo users.")
    
    # Monitoring controls
    st.sidebar.subheader("📊 Monitoring")
    
    if selected_user:
        # Scenario selection for fake data
        scenario = st.sidebar.selectbox(
            "Health Scenario",
            options=['normal', 'warning', 'critical'],
            format_func=lambda x: x.title()
        )
        
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
        
        # Manual monitoring button
        if st.sidebar.button("📊 Run Manual Check", type="primary"):
            with st.spinner("Monitoring user..."):
                health_data = generate_fake_health_data(selected_user, scenario)
                safety_data = generate_fake_safety_data(selected_user, scenario)
                
                result = st.session_state.system.monitor_user(
                    selected_user, health_data, safety_data
                )
                
                if result['status'] == 'monitored':
                    st.sidebar.success(f"✅ Monitoring completed")
                    if result.get('alerts_generated'):
                        st.sidebar.warning(f"⚠️ {len(result['alerts_generated'])} alerts generated")
                else:
                    st.sidebar.error("❌ Monitoring failed")
    
    # Main dashboard area
    if selected_user:
        render_user_dashboard(selected_user, scenario)
    else:
        render_overview_dashboard()

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
        # Overall status indicator
        dashboard = st.session_state.system.get_user_dashboard(user_id)
        if 'overall_status' in dashboard:
            status = dashboard['overall_status']['level']
            if status == 'critical':
                st.markdown('<div class="alert-critical"><h3>🚨 CRITICAL</h3></div>', unsafe_allow_html=True)
            elif status == 'warning':
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
    
    # Alerts and reminders
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚨 Active Alerts")
        alerts = st.session_state.system.agents['alert'].get_active_alerts(user_id)
        
        if alerts:
            for alert in alerts[-5:]:  # Show last 5 alerts
                priority_icon = "🔴" if alert['priority'] == 'high' else "🟡" if alert['priority'] == 'medium' else "🟢"
                st.markdown(f"{priority_icon} **{alert['alert_type']}** - {alert['message'][:50]}...")
        else:
            st.info("No active alerts")
    
    with col2:
        st.markdown("### 💊 Today's Reminders")
        reminders = st.session_state.system.agents['reminder'].get_daily_reminders(user_id)
        
        if reminders:
            for reminder in reminders:
                time_icon = "⏰" if reminder['priority'] == 'high' else "🕐"
                st.markdown(f"{time_icon} **{reminder['scheduled_time']}** - {reminder['message']}")
        else:
            st.info("No reminders scheduled")
    
    # Medical information
    st.markdown("### 🏥 Medical Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Medical Conditions:**")
        for condition in user['conditions']:
            st.markdown(f"• {condition}")
    
    with col2:
        st.markdown("**Current Medications:**")
        for medication in user['medications']:
            st.markdown(f"• {medication}")

def render_overview_dashboard():
    """Render overview dashboard when no user is selected"""
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
        
        # Users overview
        st.markdown("### 👥 Registered Users")
        
        users_data = []
        for user_id, user in st.session_state.users.items():
            dashboard = st.session_state.system.get_user_dashboard(user_id)
            status = dashboard.get('overall_status', {}).get('level', 'normal')
            alerts = len(st.session_state.system.agents['alert'].get_active_alerts(user_id))
            
            users_data.append({
                'ID': user_id,
                'Name': user['name'],
                'Age': user['age'],
                'Status': status.title(),
                'Active Alerts': alerts,
                'Conditions': len(user['conditions']),
                'Medications': len(user['medications'])
            })
        
        df = pd.DataFrame(users_data)
        st.dataframe(df, use_container_width=True)

def main():
    """Main Streamlit application"""
    render_dashboard()

if __name__ == "__main__":
    main()