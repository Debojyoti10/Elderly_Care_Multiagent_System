#!/usr/bin/env python3
"""
Test script for email functionality
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

def test_email_sending():
    """Test email sending functionality"""
    print("🧪 Testing Email Sending Functionality")
    print("=" * 50)
    
    # Get user input
    sender_email = input("Enter your Gmail address: ")
    sender_password = input("Enter your Gmail app password: ")
    recipient_email = input("Enter recipient email: ")
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = "🧪 Test Email - Elderly Care Monitor"
        
        test_message = f"""
Hello!

This is a test email from the Elderly Care Monitor system.

If you're receiving this email, your email configuration is working correctly!

Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Best regards,
Elderly Care Monitor System
        """
        
        msg.attach(MIMEText(test_message, 'plain'))
        
        # Send email
        print("📧 Sending email...")
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        
        print("✅ Email sent successfully!")
        print(f"From: {sender_email}")
        print(f"To: {recipient_email}")
        
    except smtplib.SMTPAuthenticationError as e:
        print(f"❌ Authentication failed: {e}")
        print("Make sure you're using a Gmail app password, not your regular password.")
        
    except Exception as e:
        print(f"❌ Error sending email: {e}")

if __name__ == "__main__":
    test_email_sending()
