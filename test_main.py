#!/usr/bin/env python3
"""
Simple test script to verify main.py imports and basic functionality
"""

try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Test imports
    from main import validate_email, validate_phone
    
    # Test validation functions
    print("Testing validation functions...")
    
    # Test email validation
    assert validate_email("test@example.com") == True
    assert validate_email("invalid-email") == False
    print("✅ Email validation works")
    
    # Test phone validation
    assert validate_phone("+1234567890") == True
    assert validate_phone("1234567890") == True
    assert validate_phone("123") == False
    print("✅ Phone validation works")
    
    print("🎉 All basic tests passed! The main.py file is working correctly.")
    print("✨ You can now run the Streamlit app with: streamlit run main.py")

except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
