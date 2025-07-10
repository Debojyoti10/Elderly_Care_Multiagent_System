#!/usr/bin/env python3
"""
Streamlit App Runner for Elderly Care System
"""

import os
import sys
import subprocess

def run_streamlit():
    """Run the Streamlit application"""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_py = os.path.join(script_dir, 'main.py')
    
    # Check if main.py exists
    if not os.path.exists(main_py):
        print(f"Error: {main_py} not found")
        sys.exit(1)
    
    # Run streamlit
    try:
        print("🚀 Starting Elderly Care Monitor...")
        print("📊 The application will open in your default web browser")
        print("🔗 URL: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the application")
        print("-" * 50)
        
        cmd = [sys.executable, '-m', 'streamlit', 'run', main_py, '--server.port=8501']
        subprocess.run(cmd, cwd=script_dir)
        
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_streamlit()
