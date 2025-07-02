# Elderly Care Multi-Agent AI System

A comprehensive AI-powered multi-agent system designed to assist elderly individuals living independently through real-time health monitoring, anomaly detection, and intelligent alerting.

## 🎯 Project Overview

This system addresses the challenge of ensuring the well-being of elderly individuals through:
- **Real-time Health Monitoring**: Continuous tracking of vital signs (heart rate, blood pressure, glucose, oxygen saturation)
- **Intelligent Anomaly Detection**: Machine learning-based identification of health anomalies
- **Multi-Agent Architecture**: Coordinated agents for health, safety, reminders, and alerts
- **Predictive Analytics**: Early warning system for potential health issues
- **Event-Driven Communication**: Seamless inter-agent communication through event bus

## 🏗️ System Architecture

### Agents
- **Health Agent**: ML-powered health monitoring and anomaly detection
- **Safety Agent**: Fall detection and emergency monitoring
- **Reminder Agent**: Medication and appointment reminders
- **Alert Agent**: Caregiver and emergency service notifications

### Key Components
- **Event Bus**: Central communication hub for all agents
- **Machine Learning Models**: Isolation Forest for anomaly detection, Random Forest for alert prediction
- **Data Processing**: Real-time vital signs processing and feature engineering

## 📊 Features

### Health Monitoring
- ✅ Real-time vital signs analysis
- ✅ Anomaly detection using Isolation Forest
- ✅ Risk level assessment (low, medium, high, critical)
- ✅ Personalized health recommendations
- ✅ Historical trend analysis
- ✅ Threshold violation tracking

### Machine Learning
- ✅ Supervised learning for alert prediction
- ✅ Unsupervised anomaly detection
- ✅ Feature engineering with derived health metrics
- ✅ Model persistence and loading
- ✅ Performance evaluation and metrics

### Event System
- ✅ Publish-subscribe event architecture
- ✅ Real-time inter-agent communication
- ✅ Event history and logging
- ✅ Type-safe event definitions

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Required packages (see requirements.txt)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/elderly-care-multiagent-system.git
   cd elderly-care-multiagent-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the data**
   - Ensure `data/health_monitoring.csv` contains your health monitoring data
   - The system expects columns: Device-ID/User-ID, Timestamp, Heart Rate, Blood Pressure, etc.

### Usage

1. **Train the Health Agent**
   ```python
   from agents.health_agent import HealthAgent
   
   # Initialize and train the agent
   agent = HealthAgent(data_path="data/health_monitoring.csv")
   agent.train_models()
   ```

2. **Make Health Predictions**
   ```python
   # Sample health data
   health_data = {
       'heart_rate': 85,
       'systolic_bp': 130,
       'diastolic_bp': 85,
       'glucose': 110,
       'oxygen_saturation': 97,
       'timestamp': '2025-07-02 14:30:00'
   }
   
   # Get prediction
   result = agent.predict_health_status(health_data)
   print(f"Alert Required: {result['alert_required']}")
   print(f"Risk Level: {result['risk_level']}")
   ```

3. **Run Comprehensive Tests**
   ```bash
   python test_comprehensive_health_agent.py
   ```

## 📁 Project Structure

```
elderly_care/
├── agents/
│   ├── health_agent.py      # ML-powered health monitoring agent
│   ├── alert_agent.py       # Alert and notification management
│   ├── reminder_agent.py    # Medication and appointment reminders
│   └── safety_agent.py      # Fall detection and safety monitoring
├── data/
│   ├── health_monitoring.csv    # Health monitoring dataset
│   ├── daily_reminder.csv       # Daily activity data
│   └── safety_monitoring.csv    # Safety monitoring data
├── models/                  # Trained ML models storage
├── event_bus.py            # Inter-agent communication system
├── main.py                 # Main application entry point
├── test_comprehensive_health_agent.py  # Comprehensive testing suite
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔬 Testing

The project includes a comprehensive test suite that validates:
- ✅ Import functionality
- ✅ Agent initialization
- ✅ Data loading and preprocessing
- ✅ Feature preparation
- ✅ Model training and evaluation
- ✅ Prediction accuracy
- ✅ Health insights generation
- ✅ Model persistence
- ✅ Event bus integration
- ✅ Error handling

Run tests with:
```bash
python test_comprehensive_health_agent.py
```

## 📈 Machine Learning Details

### Health Agent ML Pipeline
1. **Data Preprocessing**
   - Blood pressure parsing (systolic/diastolic)
   - Time-based feature extraction
   - Deviation calculations from normal ranges
   - Threshold violation counting

2. **Feature Engineering**
   - Pulse pressure calculation
   - Mean arterial pressure (MAP)
   - Heart rate deviation percentage
   - Glucose and SpO2 deviation metrics
   - Temporal features (hour, day of week)

3. **Models**
   - **Isolation Forest**: Anomaly detection with 10% contamination rate
   - **Random Forest**: Alert prediction with 100 estimators
   - **StandardScaler**: Feature normalization

4. **Evaluation Metrics**
   - Classification report for alert prediction
   - Feature importance analysis
   - Anomaly detection rate
   - Cross-validation scores

## 🚨 Alert System

### Risk Levels
- **Critical**: Alert probability > 0.8 OR 3+ violations OR anomaly score < -0.5
- **High**: Alert probability > 0.6 OR 2+ violations OR anomaly score < -0.3
- **Medium**: Alert probability > 0.4 OR 1+ violations OR anomaly score < -0.1
- **Low**: All parameters within normal ranges

### Recommendations Engine
The system provides personalized recommendations based on:
- Heart rate anomalies (bradycardia/tachycardia)
- Blood pressure issues (hypertension/hypotension)
- Glucose level concerns (hyperglycemia/hypoglycemia)
- Oxygen saturation problems
- Risk level assessment

## 🔧 Configuration

### Normal Health Ranges (Elderly)
- Heart Rate: 60-100 bpm
- Systolic BP: 90-140 mmHg
- Diastolic BP: 60-90 mmHg
- Glucose: 70-140 mg/dL
- Oxygen Saturation: 95-100%

### Model Parameters
- Isolation Forest contamination: 0.1
- Random Forest estimators: 100
- Train/test split: 80/20
- Random state: 42 (reproducible results)

## 📊 Dataset Format

The health monitoring dataset should include:
- Device-ID/User-ID
- Timestamp
- Heart Rate
- Blood Pressure (format: "120/80 mmHg")
- Glucose Levels
- Oxygen Saturation (SpO₂%)
- Threshold flags (Yes/No)
- Alert Triggered (Yes/No)
- Caregiver Notified (Yes/No)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Scikit-learn for machine learning algorithms
- Pandas for data manipulation
- NumPy for numerical computations
- Healthcare professionals for domain expertise

## 📞 Support

For questions or support, please open an issue in the GitHub repository.

---

**Note**: This system is designed for monitoring and alerting purposes. It should complement, not replace, professional medical care and regular health checkups.
