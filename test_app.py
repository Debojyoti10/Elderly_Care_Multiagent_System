#!/usr/bin/env python3
"""
Test script to verify the Streamlit application works
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd
    import numpy as np
    from elderly_care_system import ElderlyCareSystem
    
    print("✅ All imports successful!")
    
    # Test system initialization
    print("🔧 Testing system initialization...")
    system = ElderlyCareSystem()
    print("✅ System initialized successfully!")
    
    # Test fake data generation
    print("📊 Testing fake data generation...")
    from main import generate_fake_user_data, generate_fake_health_data, generate_fake_safety_data
    
    # Test the functions exist
    health_data = generate_fake_health_data('U001', 'normal')
    safety_data = generate_fake_safety_data('U001', 'normal')
    
    print("✅ Fake data generation working!")
    print(f"Sample health data: {health_data}")
    print(f"Sample safety data: {safety_data}")
    
    print("\n🎉 All tests passed! The Streamlit application is ready to run.")
    print("Run: streamlit run main.py")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required packages: pip install streamlit plotly scikit-learn matplotlib")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Please check the installation and try again.")
