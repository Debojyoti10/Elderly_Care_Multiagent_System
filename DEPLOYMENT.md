# Elderly Care Monitor - Multi-Page Web Application

## 🚀 New Features

### Multi-Page Architecture
- **Setup Page**: Initial caregiver contact information setup
- **Dashboard Page**: Real-time health monitoring
- **Users Page**: User management and details
- **Alerts Page**: View sent alerts and system notifications
- **Settings Page**: Update contact info and test alerts

### Emergency Alert System
- **Email Alerts**: Detailed health reports sent to caregiver email
- **SMS Alerts**: Quick notifications sent to caregiver phone
- **Real-time Notifications**: Instant alerts for critical health events
- **Alert History**: View all sent alerts with timestamps

### Enhanced User Experience
- **Mandatory Setup**: Email and phone verification before access
- **Navigation**: Easy page switching with button navigation
- **Contact Management**: Update caregiver info anytime
- **Test Alerts**: Send test emails and SMS to verify setup

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python run_app.py
```

Or directly with Streamlit:
```bash
streamlit run main.py
```

### 3. Access the Application
- Open your browser and go to: `http://localhost:8501`
- The application will automatically load with a modern, responsive interface

## 📊 Features

### Real-time Monitoring Dashboard
- **Multi-user Support**: Monitor multiple elderly users simultaneously
- **Health Metrics**: Heart rate, blood pressure, glucose, oxygen saturation
- **Safety Monitoring**: Fall detection, location tracking, activity monitoring
- **Alert System**: Priority-based alerts with escalation
- **Medical Information**: Conditions, medications, emergency contacts

### Interactive Controls
- **Fake Data Generation**: Toggle to generate demo users and data
- **Scenario Simulation**: Test normal, warning, and critical health scenarios
- **Real-time Updates**: Enable continuous monitoring with live data updates
- **Manual Monitoring**: Run individual health checks on demand

### Visual Analytics
- **Health Trends**: Interactive charts showing vital signs over time
- **Status Indicators**: Color-coded health status (Normal, Warning, Critical)
- **Alert Management**: View and manage active alerts
- **Reminder System**: Daily medication and activity reminders

## 🎨 UI/UX Features

### Modern Design
- **Gradient Backgrounds**: Beautiful gradient color schemes
- **Responsive Layout**: Works on desktop, tablet, and mobile
- **Interactive Charts**: Powered by Plotly for smooth interactions
- **Status Cards**: Color-coded metric cards for quick status overview

### User Experience
- **Intuitive Navigation**: Sidebar controls for easy access
- **Real-time Updates**: Live data refreshing every few seconds
- **Error Handling**: Graceful error messages and recovery
- **Loading States**: Smooth loading animations and progress indicators

## 🔧 Configuration

### System Configuration
The system uses `config/system_config.json` for configuration:
- Monitoring intervals
- Alert thresholds
- Model persistence settings
- Data paths

### Fake Data Generation
The system includes comprehensive fake data generation:
- **3 Demo Users**: Pre-configured with different health profiles
- **Dynamic Health Data**: Realistic vital signs with variations
- **Safety Scenarios**: Fall detection and emergency situations
- **Historical Data**: Generated time-series data for trend analysis

## 📱 Usage Examples

### Demo Mode
1. Enable "Generate Fake Data" in the sidebar
2. Select a user from the dropdown
3. Choose different health scenarios (normal, warning, critical)
4. Click "Run Manual Check" to simulate monitoring
5. View real-time updates in the dashboard

### Real-time Monitoring
1. Select a user
2. Toggle "Real-time Monitoring" to start continuous monitoring
3. Watch as health metrics update automatically
4. Monitor the alerts panel for any health concerns

### Emergency Simulation
1. Select a user
2. Change scenario to "Critical"
3. Run manual check to simulate emergency
4. Observe how the system generates alerts and updates status

## 🚀 Deployment Options

### Local Development
```bash
# Clone the repository
git clone <repository-url>
cd elderly_care

# Install dependencies
pip install -r requirements.txt

# Run the application
python run_app.py
```

### Cloud Deployment (Streamlit Cloud)
1. Push your code to GitHub
2. Connect to Streamlit Cloud (https://streamlit.io/cloud)
3. Deploy directly from your GitHub repository
4. Access your application via the provided URL

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📊 Sample Data

The system generates realistic fake data including:

### Health Metrics
- Heart Rate: 60-100 bpm (normal), up to 150+ (critical)
- Blood Pressure: 120/80 (normal), up to 200/120 (critical)
- Glucose: 80-140 mg/dL (normal), up to 400+ (critical)
- Oxygen Saturation: 95-100% (normal), below 85% (critical)

### Safety Data
- Movement: Walking, Sitting, Standing, No Movement
- Locations: Living Room, Bedroom, Kitchen, Bathroom
- Fall Detection: Impact force levels and inactivity periods
- Emergency Scenarios: High impact events with extended inactivity

### User Profiles
- **Eleanor Johnson (78)**: Hypertension, Diabetes
- **Robert Chen (82)**: Heart Disease, Arthritis
- **Margaret Smith (75)**: Osteoporosis, Mild Cognitive Impairment

## 🔧 Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed
2. **Port Conflicts**: Change port using `--server.port=8502`
3. **Model Loading**: Check if model files exist in `models/` directory
4. **Data Issues**: Clear browser cache and restart application

### Performance Tips
- Use fake data mode for demonstration
- Limit historical data range for better performance
- Monitor system resources during real-time monitoring

## 🎯 Next Steps

### Enhancements
- Add more sophisticated AI models
- Implement WebSocket for real-time updates
- Add user authentication and authorization
- Include more health metrics and sensors
- Add mobile app integration

### Production Deployment
- Set up proper database for data persistence
- Implement logging and monitoring
- Add backup and recovery systems
- Configure SSL/TLS for secure connections
- Set up load balancing for multiple users
