"""
Demo script for the Multi-Page Elderly Care Monitor

This script demonstrates the key features of the application:
1. Email and phone setup
2. Multi-page navigation
3. Emergency alert system
4. Fake data generation
"""

import streamlit as st

def show_demo_instructions():
    """Show instructions for using the demo"""
    st.markdown("""
    # 🎯 Demo Instructions
    
    ## 1. Setup Phase
    - Enter your **real email address** (you'll receive actual test alerts)
    - Enter your **phone number** (for SMS simulation)
    - Click "Complete Setup" to proceed
    
    ## 2. Dashboard Navigation
    - **🏠 Dashboard**: Main monitoring interface
    - **👥 Users**: User management and details
    - **🚨 Alerts**: View sent alerts and system alerts
    - **⚙️ Settings**: Update contact info and test alerts
    
    ## 3. Generate Fake Data
    - Toggle "Generate Fake Data" in the sidebar
    - This creates 3 demo users: Eleanor, Robert, and Margaret
    
    ## 4. Test Emergency Alerts
    - Select a user from the dropdown
    - Choose "Critical" scenario
    - Click "Run Manual Check"
    - Check your email for the alert!
    
    ## 5. Features to Explore
    - **Real-time monitoring**: Toggle continuous monitoring
    - **Health scenarios**: Test normal, warning, and critical states
    - **Interactive charts**: View health trends over time
    - **Alert management**: View all sent alerts in the Alerts page
    - **Settings**: Test email and SMS functionality
    
    ## 📧 What You'll Receive
    When you generate a critical alert, you'll receive:
    - **Email**: Detailed health report with metrics
    - **SMS**: Quick alert notification (simulated)
    
    ## 🔧 Technical Demo
    - All alerts are logged and viewable in the Alerts page
    - Email sending is simulated but shows real format
    - SMS alerts are simulated with realistic messages
    - Data is generated realistically based on medical conditions
    """)

if __name__ == "__main__":
    show_demo_instructions()
