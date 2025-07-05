import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime, timedelta, time
import pickle
import os
import logging
from typing import Dict, List, Tuple, Optional, Union
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReminderAgent:
    """
    Advanced Reminder Agent for Elderly Care
    
    This agent manages medication reminders, activity schedules, and compliance tracking
    for elderly individuals. It uses machine learning to optimize reminder timing and
    predict compliance patterns.
    """
    
    def __init__(self, data_path: Optional[str] = None, model_save_path: str = "models"):
        """
        Initialize the Reminder Agent
        
        Args:
            data_path: Path to the daily reminder CSV file
            model_save_path: Directory to save trained models
        """
        self.data_path = data_path or "data/daily_reminder.csv"
        self.model_save_path = model_save_path
        self.scaler = StandardScaler()
        self.compliance_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        self.reminder_optimizer = RandomForestClassifier(n_estimators=100, random_state=42)
        self.reminder_type_encoder = LabelEncoder()
        self.feature_columns = []
        self.is_trained = False
        
        # Reminder configuration
        self.reminder_types = {
            'Medication': {
                'priority': 1,  # Highest priority
                'default_times': ['08:00', '12:00', '18:00'],  # Common medication times
                'retry_interval': 30,  # minutes
                'max_retries': 3,
                'urgency_level': 'critical'
            },
            'Hydration': {
                'priority': 2,
                'default_times': ['09:00', '13:00', '17:00', '21:00'],
                'retry_interval': 60,
                'max_retries': 2,
                'urgency_level': 'high'
            },
            'Exercise': {
                'priority': 3,
                'default_times': ['10:00', '16:00'],
                'retry_interval': 120,
                'max_retries': 2,
                'urgency_level': 'medium'
            },
            'Appointment': {
                'priority': 1,  # High priority for appointments
                'default_times': ['09:00'],  # Usually scheduled individually
                'retry_interval': 15,  # Shorter retry for appointments
                'max_retries': 5,
                'urgency_level': 'critical'
            }
        }
        
        # User profiles and preferences
        self.user_profiles = {}
        self.active_reminders = {}
        self.compliance_history = defaultdict(dict)
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_save_path, exist_ok=True)
    
    def load_and_preprocess_data(self) -> pd.DataFrame:
        """
        Load and preprocess the daily reminder dataset
        
        Returns:
            Preprocessed DataFrame
        """
        try:
            # Load the dataset
            df = pd.read_csv(self.data_path)
            logger.info(f"Loaded reminder dataset with {len(df)} records")
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Convert timestamp to datetime
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df['Scheduled Time'] = pd.to_datetime(df['Scheduled Time'], format='%H:%M:%S').dt.time
            
            # Convert Yes/No columns to binary
            yes_no_columns = ['Reminder Sent (Yes/No)', 'Acknowledged (Yes/No)']
            for col in yes_no_columns:
                df[col] = (df[col] == 'Yes').astype(int)
            
            # Create time-based features
            df['Hour'] = df['Timestamp'].dt.hour
            df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
            df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
            df['IsWeekday'] = (df['DayOfWeek'] < 5).astype(int)
            
            # Extract scheduled hour
            df['Scheduled_Hour'] = df['Scheduled Time'].apply(lambda x: x.hour)
            df['Scheduled_Minute'] = df['Scheduled Time'].apply(lambda x: x.minute)
            
            # Create time period features
            df['Is_Morning'] = (df['Scheduled_Hour'] >= 6) & (df['Scheduled_Hour'] < 12)
            df['Is_Afternoon'] = (df['Scheduled_Hour'] >= 12) & (df['Scheduled_Hour'] < 18)
            df['Is_Evening'] = (df['Scheduled_Hour'] >= 18) & (df['Scheduled_Hour'] < 22)
            df['Is_Night'] = (df['Scheduled_Hour'] >= 22) | (df['Scheduled_Hour'] < 6)
            
            # Calculate time difference between actual and scheduled time
            df['Scheduled_DateTime'] = pd.to_datetime(
                df['Timestamp'].dt.date.astype(str) + ' ' + df['Scheduled Time'].astype(str)
            )
            df['Time_Diff_Minutes'] = (df['Timestamp'] - df['Scheduled_DateTime']).dt.total_seconds() / 60
            
            # Create reminder timing features
            df['Is_Early'] = (df['Time_Diff_Minutes'] < -30).astype(int)
            df['Is_OnTime'] = (df['Time_Diff_Minutes'].abs() <= 30).astype(int)
            df['Is_Late'] = (df['Time_Diff_Minutes'] > 30).astype(int)
            
            # Encode reminder types
            df['Reminder_Type_Encoded'] = self.reminder_type_encoder.fit_transform(df['Reminder Type'])
            
            # Create reminder type specific features
            df['Is_Medication'] = (df['Reminder Type'] == 'Medication').astype(int)
            df['Is_Exercise'] = (df['Reminder Type'] == 'Exercise').astype(int)
            df['Is_Hydration'] = (df['Reminder Type'] == 'Hydration').astype(int)
            df['Is_Appointment'] = (df['Reminder Type'] == 'Appointment').astype(int)
            
            # Create compliance features by user
            df['User_Compliance_Rate'] = df.groupby('Device-ID/User-ID')['Acknowledged (Yes/No)'].transform('mean')
            df['User_Response_Rate'] = df.groupby('Device-ID/User-ID')['Reminder Sent (Yes/No)'].transform('mean')
            
            # Create reminder type compliance by user
            df['User_Type_Compliance'] = df.groupby(['Device-ID/User-ID', 'Reminder Type'])['Acknowledged (Yes/No)'].transform('mean')
            
            # Time-based compliance patterns
            df['Hour_Compliance'] = df.groupby('Scheduled_Hour')['Acknowledged (Yes/No)'].transform('mean')
            df['Weekday_Compliance'] = df.groupby('DayOfWeek')['Acknowledged (Yes/No)'].transform('mean')
            
            # Create success/failure features
            df['Reminder_Success'] = (df['Reminder Sent (Yes/No)'] == 1) & (df['Acknowledged (Yes/No)'] == 1)
            df['Reminder_Failure'] = (df['Reminder Sent (Yes/No)'] == 1) & (df['Acknowledged (Yes/No)'] == 0)
            df['Reminder_Not_Sent'] = (df['Reminder Sent (Yes/No)'] == 0)
            
            logger.info(f"Data preprocessing completed. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare features for machine learning models
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Feature matrix, compliance target, and reminder success target
        """
        # Select relevant features for training
        feature_columns = [
            'Hour', 'DayOfWeek', 'IsWeekend', 'IsWeekday',
            'Scheduled_Hour', 'Scheduled_Minute', 'Is_Morning', 'Is_Afternoon', 
            'Is_Evening', 'Is_Night', 'Time_Diff_Minutes', 'Is_Early', 
            'Is_OnTime', 'Is_Late', 'Reminder_Type_Encoded', 'Is_Medication', 
            'Is_Exercise', 'Is_Hydration', 'Is_Appointment', 'User_Compliance_Rate',
            'User_Response_Rate', 'User_Type_Compliance', 'Hour_Compliance',
            'Weekday_Compliance'
        ]
        
        self.feature_columns = feature_columns
        X = df[feature_columns].values
        
        # Target variables
        y_compliance = df['Acknowledged (Yes/No)'].values
        y_reminder_sent = df['Reminder Sent (Yes/No)'].values
        
        return X, np.array(y_compliance), np.array(y_reminder_sent)
    
    def train_models(self, df: Optional[pd.DataFrame] = None):
        """
        Train compliance prediction and reminder optimization models
        
        Args:
            df: Preprocessed DataFrame (optional, will load if not provided)
        """
        if df is None:
            df = self.load_and_preprocess_data()
        
        X, y_compliance, y_reminder_sent = self.prepare_features(df)
        
        # Split data for training and testing
        X_train, X_test, y_comp_train, y_comp_test, y_rem_train, y_rem_test = train_test_split(
            X, y_compliance, y_reminder_sent, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train compliance prediction model
        logger.info("Training compliance prediction model...")
        self.compliance_predictor.fit(X_train_scaled, y_comp_train)
        
        # Train reminder optimization model
        logger.info("Training reminder optimization model...")
        self.reminder_optimizer.fit(X_train_scaled, y_rem_train)
        
        # Evaluate models
        self._evaluate_models(X_test_scaled, y_comp_test, y_rem_test)
        
        # Save models
        self._save_models()
        
        self.is_trained = True
        logger.info("Reminder model training completed successfully!")
    
    def _evaluate_models(self, X_test: np.ndarray, y_comp_test: np.ndarray, y_rem_test: np.ndarray):
        """
        Evaluate the trained models
        
        Args:
            X_test: Test features
            y_comp_test: Test compliance targets
            y_rem_test: Test reminder targets
        """
        # Evaluate compliance prediction
        y_comp_pred = self.compliance_predictor.predict(X_test)
        
        logger.info("Compliance Prediction Model Performance:")
        logger.info(f"\n{classification_report(y_comp_test, y_comp_pred)}")
        
        # Evaluate reminder optimization
        y_rem_pred = self.reminder_optimizer.predict(X_test)
        
        logger.info("Reminder Optimization Model Performance:")
        logger.info(f"\n{classification_report(y_rem_test, y_rem_pred)}")
        
        # Feature importance for compliance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.compliance_predictor.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 10 Most Important Features for Compliance:")
        logger.info(f"\n{feature_importance.head(10)}")
    
    def create_reminder_schedule(self, user_id: str, reminder_config: Dict) -> Dict:
        """
        Create a personalized reminder schedule for a user
        
        Args:
            user_id: User/Device ID
            reminder_config: Configuration for reminders
            
        Returns:
            Dictionary with scheduled reminders
        """
        schedule_data = {
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'reminders': []
        }
        
        for reminder_type, details in reminder_config.items():
            if reminder_type in self.reminder_types:
                for scheduled_time in details.get('times', self.reminder_types[reminder_type]['default_times']):
                    reminder = {
                        'type': reminder_type,
                        'scheduled_time': scheduled_time,
                        'message': details.get('message', f"Time for your {reminder_type.lower()}"),
                        'priority': self.reminder_types[reminder_type]['priority'],
                        'urgency_level': self.reminder_types[reminder_type]['urgency_level'],
                        'retry_interval': self.reminder_types[reminder_type]['retry_interval'],
                        'max_retries': self.reminder_types[reminder_type]['max_retries'],
                        'days_of_week': details.get('days_of_week', [0, 1, 2, 3, 4, 5, 6]),  # All days by default
                        'is_active': True
                    }
                    schedule_data['reminders'].append(reminder)
        
        self.user_profiles[user_id] = schedule_data
        return schedule_data
    
    def predict_compliance(self, user_id: str, reminder_type: str, scheduled_time: str) -> Dict:
        """
        Predict compliance for a specific reminder
        
        Args:
            user_id: User/Device ID
            reminder_type: Type of reminder
            scheduled_time: Scheduled time in HH:MM format
            
        Returns:
            Dictionary with compliance prediction
        """
        if not self.is_trained:
            self._load_models()
        
        try:
            # Create feature vector for prediction
            current_time = datetime.now()
            scheduled_datetime = datetime.strptime(scheduled_time, '%H:%M').time()
            
            # Get user historical data if available
            user_compliance_rate = self.compliance_history.get(user_id, {}).get('overall_rate', 0.5)
            user_response_rate = self.compliance_history.get(user_id, {}).get('response_rate', 0.5)
            user_type_compliance = self.compliance_history.get(user_id, {}).get(reminder_type, 0.5)
            
            # Handle LabelEncoder transform safely
            try:
                reminder_type_encoded = self.reminder_type_encoder.transform([reminder_type])
                # Convert numpy array to int
                reminder_type_encoded = int(np.array(reminder_type_encoded).flatten()[0])
            except (ValueError, AttributeError, IndexError):
                reminder_type_encoded = 0  # Default value if transformation fails
            
            # Create input features
            input_features = pd.DataFrame([{
                'Hour': current_time.hour,
                'DayOfWeek': current_time.weekday(),
                'IsWeekend': int(current_time.weekday() >= 5),
                'IsWeekday': int(current_time.weekday() < 5),
                'Scheduled_Hour': scheduled_datetime.hour,
                'Scheduled_Minute': scheduled_datetime.minute,
                'Is_Morning': int(6 <= scheduled_datetime.hour < 12),
                'Is_Afternoon': int(12 <= scheduled_datetime.hour < 18),
                'Is_Evening': int(18 <= scheduled_datetime.hour < 22),
                'Is_Night': int(scheduled_datetime.hour >= 22 or scheduled_datetime.hour < 6),
                'Time_Diff_Minutes': 0,  # Assuming on-time
                'Is_Early': 0,
                'Is_OnTime': 1,
                'Is_Late': 0,
                'Reminder_Type_Encoded': reminder_type_encoded,
                'Is_Medication': int(reminder_type == 'Medication'),
                'Is_Exercise': int(reminder_type == 'Exercise'),
                'Is_Hydration': int(reminder_type == 'Hydration'),
                'Is_Appointment': int(reminder_type == 'Appointment'),
                'User_Compliance_Rate': user_compliance_rate,
                'User_Response_Rate': user_response_rate,
                'User_Type_Compliance': user_type_compliance,
                'Hour_Compliance': 0.5,  # Default value
                'Weekday_Compliance': 0.5  # Default value
            }])
            
            # Scale features
            X_scaled = self.scaler.transform(input_features[self.feature_columns])
            
            # Make predictions
            compliance_prob = self.compliance_predictor.predict_proba(X_scaled)[0]
            compliance_prediction = self.compliance_predictor.predict(X_scaled)[0]
            
            # Assess risk level
            risk_level = self._assess_compliance_risk(compliance_prob[1], reminder_type)
            
            # Generate recommendations
            recommendations = self._generate_compliance_recommendations(
                user_id, reminder_type, compliance_prob[1], risk_level
            )
            
            return {
                'user_id': user_id,
                'reminder_type': reminder_type,
                'scheduled_time': scheduled_time,
                'compliance_predicted': bool(compliance_prediction),
                'compliance_probability': float(compliance_prob[1]),
                'risk_level': risk_level,
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error predicting compliance: {e}")
            return {
                'error': str(e),
                'user_id': user_id,
                'reminder_type': reminder_type,
                'risk_level': 'high',  # Safe default
                'timestamp': datetime.now().isoformat()
            }
    
    def send_reminder(self, user_id: str, reminder_type: str, message: str, 
                     priority: int = 1, retry_count: int = 0) -> Dict:
        """
        Send a reminder to a user
        
        Args:
            user_id: User/Device ID
            reminder_type: Type of reminder
            message: Reminder message
            priority: Priority level (1 = highest)
            retry_count: Number of retry attempts
            
        Returns:
            Dictionary with reminder status
        """
        reminder_id = f"{user_id}_{reminder_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        reminder_data = {
            'reminder_id': reminder_id,
            'user_id': user_id,
            'reminder_type': reminder_type,
            'message': message,
            'priority': priority,
            'sent_at': datetime.now().isoformat(),
            'retry_count': retry_count,
            'status': 'sent',
            'acknowledged': False
        }
        
        # Store active reminder
        self.active_reminders[reminder_id] = reminder_data
        
        # Log reminder
        logger.info(f"Reminder sent to {user_id}: {message}")
        
        return reminder_data
    
    def acknowledge_reminder(self, reminder_id: str, user_id: str) -> Dict:
        """
        Acknowledge a reminder
        
        Args:
            reminder_id: Reminder ID
            user_id: User/Device ID
            
        Returns:
            Dictionary with acknowledgment status
        """
        if reminder_id in self.active_reminders:
            reminder = self.active_reminders[reminder_id]
            reminder['acknowledged'] = True
            reminder['acknowledged_at'] = datetime.now().isoformat()
            reminder['status'] = 'acknowledged'
            
            # Update compliance history
            self._update_compliance_history(user_id, reminder['reminder_type'], True)
            
            logger.info(f"Reminder {reminder_id} acknowledged by {user_id}")
            return {'status': 'success', 'message': 'Reminder acknowledged'}
        else:
            return {'status': 'error', 'message': 'Reminder not found'}
    
    def get_daily_reminders(self, user_id: str, date: Optional[str] = None) -> List[Dict]:
        """
        Get all reminders for a user for a specific date
        
        Args:
            user_id: User/Device ID
            date: Date in YYYY-MM-DD format (default: today)
            
        Returns:
            List of reminders for the date
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        target_date = datetime.strptime(date, '%Y-%m-%d')
        weekday = target_date.weekday()
        
        daily_reminders = []
        
        if user_id in self.user_profiles:
            for reminder in self.user_profiles[user_id]['reminders']:
                if (reminder['is_active'] and 
                    weekday in reminder['days_of_week']):
                    
                    # Predict compliance for this reminder
                    compliance_pred = self.predict_compliance(
                        user_id, reminder['type'], reminder['scheduled_time']
                    )
                    
                    daily_reminder = {
                        'type': reminder['type'],
                        'scheduled_time': reminder['scheduled_time'],
                        'message': reminder['message'],
                        'priority': reminder['priority'],
                        'urgency_level': reminder['urgency_level'],
                        'compliance_probability': compliance_pred.get('compliance_probability', 0.5),
                        'risk_level': compliance_pred.get('risk_level', 'medium'),
                        'recommendations': compliance_pred.get('recommendations', [])
                    }
                    daily_reminders.append(daily_reminder)
        
        # Sort by priority and scheduled time
        daily_reminders.sort(key=lambda x: (x['priority'], x['scheduled_time']))
        
        return daily_reminders
    
    def get_compliance_insights(self, user_id: Optional[str] = None, days: int = 30) -> Dict:
        """
        Get compliance insights and trends
        
        Args:
            user_id: User/Device ID (optional, for all users if None)
            days: Number of days to analyze
            
        Returns:
            Dictionary with compliance insights
        """
        try:
            df = self.load_and_preprocess_data()
            
            if user_id:
                df = df[df['Device-ID/User-ID'] == user_id]
            
            # Filter by date range
            end_date = df['Timestamp'].max()
            start_date = end_date - timedelta(days=days)
            df_recent = df[df['Timestamp'] >= start_date]
            
            if len(df_recent) == 0:
                return {"error": "No recent data found"}
            
            # Calculate compliance metrics
            insights = {
                'user_id': user_id,
                'analysis_period': f"{days} days",
                'total_reminders': len(df_recent),
                'reminders_sent': df_recent['Reminder Sent (Yes/No)'].sum(),
                'reminders_acknowledged': df_recent['Acknowledged (Yes/No)'].sum(),
                'overall_compliance_rate': df_recent['Acknowledged (Yes/No)'].mean(),
                'reminder_success_rate': df_recent['Reminder Sent (Yes/No)'].mean(),
                
                # By reminder type
                'compliance_by_type': {
                    reminder_type: {
                        'total': len(df_type),
                        'compliance_rate': df_type['Acknowledged (Yes/No)'].mean(),
                        'sent_rate': df_type['Reminder Sent (Yes/No)'].mean()
                    }
                    for reminder_type, df_type in df_recent.groupby('Reminder Type')
                },
                
                # By time of day
                'compliance_by_hour': df_recent.groupby('Scheduled_Hour')['Acknowledged (Yes/No)'].mean().to_dict(),
                
                # By day of week
                'compliance_by_weekday': df_recent.groupby('DayOfWeek')['Acknowledged (Yes/No)'].mean().to_dict(),
                
                # Trends
                'best_compliance_hour': df_recent.groupby('Scheduled_Hour')['Acknowledged (Yes/No)'].mean().idxmax(),
                'worst_compliance_hour': df_recent.groupby('Scheduled_Hour')['Acknowledged (Yes/No)'].mean().idxmin(),
                'best_compliance_type': df_recent.groupby('Reminder Type')['Acknowledged (Yes/No)'].mean().idxmax(),
                'worst_compliance_type': df_recent.groupby('Reminder Type')['Acknowledged (Yes/No)'].mean().idxmin()
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating compliance insights: {e}")
            return {"error": str(e)}
    
    def _assess_compliance_risk(self, compliance_prob: float, reminder_type: str) -> str:
        """
        Assess compliance risk level
        
        Args:
            compliance_prob: Compliance probability
            reminder_type: Type of reminder
            
        Returns:
            Risk level string
        """
        # Higher risk for critical reminders
        if reminder_type in ['Medication', 'Appointment']:
            if compliance_prob < 0.3:
                return 'critical'
            elif compliance_prob < 0.6:
                return 'high'
            elif compliance_prob < 0.8:
                return 'medium'
            else:
                return 'low'
        else:
            if compliance_prob < 0.2:
                return 'high'
            elif compliance_prob < 0.5:
                return 'medium'
            else:
                return 'low'
    
    def _generate_compliance_recommendations(self, user_id: str, reminder_type: str, 
                                           compliance_prob: float, risk_level: str) -> List[str]:
        """
        Generate recommendations to improve compliance
        
        Args:
            user_id: User/Device ID
            reminder_type: Type of reminder
            compliance_prob: Compliance probability
            risk_level: Risk level
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if risk_level == 'critical':
            recommendations.append("URGENT: Contact caregiver immediately for medication support")
            recommendations.append("Consider automated pill dispenser or medication management service")
        
        elif risk_level == 'high':
            recommendations.append("Schedule additional check-ins with caregiver")
            recommendations.append("Consider adjusting reminder timing or frequency")
        
        # Type-specific recommendations
        if reminder_type == 'Medication':
            if compliance_prob < 0.5:
                recommendations.append("Consider setting up pill organizer or medication reminders")
                recommendations.append("Review with doctor if medication timing can be adjusted")
        
        elif reminder_type == 'Exercise':
            if compliance_prob < 0.5:
                recommendations.append("Start with shorter, easier exercise sessions")
                recommendations.append("Consider group activities or walking with a friend")
        
        elif reminder_type == 'Hydration':
            if compliance_prob < 0.5:
                recommendations.append("Keep water bottle visible and accessible")
                recommendations.append("Set more frequent, smaller hydration reminders")
        
        elif reminder_type == 'Appointment':
            if compliance_prob < 0.5:
                recommendations.append("Add appointment to calendar with multiple reminders")
                recommendations.append("Arrange transportation in advance")
        
        return recommendations
    
    def _update_compliance_history(self, user_id: str, reminder_type: str, acknowledged: bool):
        """
        Update compliance history for a user
        
        Args:
            user_id: User/Device ID
            reminder_type: Type of reminder
            acknowledged: Whether reminder was acknowledged
        """
        if user_id not in self.compliance_history:
            self.compliance_history[user_id] = {
                'overall_rate': 0.5,
                'response_rate': 0.5,
                'total_reminders': 0,
                'acknowledged_count': 0
            }
        
        user_history = self.compliance_history[user_id]
        user_history['total_reminders'] += 1
        
        if acknowledged:
            user_history['acknowledged_count'] += 1
        
        # Update rates
        user_history['overall_rate'] = user_history['acknowledged_count'] / user_history['total_reminders']
        
        # Update type-specific compliance
        if reminder_type not in user_history:
            user_history[reminder_type] = 0.5
        
        # Simple moving average for type-specific compliance
        current_rate = user_history[reminder_type]
        user_history[reminder_type] = (current_rate * 0.8) + (float(acknowledged) * 0.2)
    
    def _save_models(self):
        """Save trained models to disk"""
        model_files = {
            'scaler': self.scaler,
            'compliance_predictor': self.compliance_predictor,
            'reminder_optimizer': self.reminder_optimizer,
            'reminder_type_encoder': self.reminder_type_encoder,
            'feature_columns': self.feature_columns,
            'reminder_types': self.reminder_types,
            'user_profiles': self.user_profiles,
            'compliance_history': dict(self.compliance_history)
        }
        
        for name, model in model_files.items():
            with open(f"{self.model_save_path}/reminder_{name}.pkl", 'wb') as f:
                pickle.dump(model, f)
        
        logger.info(f"Reminder models saved to {self.model_save_path}/")
    
    def _load_models(self):
        """Load trained models from disk"""
        try:
            model_files = ['scaler', 'compliance_predictor', 'reminder_optimizer', 
                          'reminder_type_encoder', 'feature_columns', 'reminder_types',
                          'user_profiles', 'compliance_history']
            
            for name in model_files:
                with open(f"{self.model_save_path}/reminder_{name}.pkl", 'rb') as f:
                    if name == 'compliance_history':
                        self.compliance_history = defaultdict(dict, pickle.load(f))
                    else:
                        setattr(self, name, pickle.load(f))
            
            self.is_trained = True
            logger.info("Reminder models loaded successfully!")
            
        except FileNotFoundError:
            logger.warning("No saved reminder models found. Please train the models first.")
            self.is_trained = False

# Example usage and testing
if __name__ == "__main__":
    # Initialize reminder agent
    agent = ReminderAgent(data_path="data/daily_reminder.csv")
    
    # Train the models
    print("Training reminder models...")
    agent.train_models()
    
    # Create a user profile with reminders
    print("\n=== Creating User Profile ===")
    user_id = "D1001"
    reminder_config = {
        'Medication': {
            'times': ['08:00', '12:00', '20:00'],
            'message': 'Time to take your medication',
            'days_of_week': [0, 1, 2, 3, 4, 5, 6]  # Every day
        },
        'Exercise': {
            'times': ['10:00', '16:00'],
            'message': 'Time for your daily exercise',
            'days_of_week': [0, 1, 2, 3, 4]  # Weekdays only
        },
        'Hydration': {
            'times': ['09:00', '13:00', '17:00', '21:00'],
            'message': 'Remember to drink water',
            'days_of_week': [0, 1, 2, 3, 4, 5, 6]  # Every day
        }
    }
    
    schedule = agent.create_reminder_schedule(user_id, reminder_config)
    print(f"Created schedule with {len(schedule['reminders'])} reminders")
    
    # Test compliance prediction
    print("\n=== Testing Compliance Prediction ===")
    compliance_pred = agent.predict_compliance(user_id, 'Medication', '08:00')
    print("Medication Compliance Prediction:")
    for key, value in compliance_pred.items():
        print(f"  {key}: {value}")
    
    # Get daily reminders
    print("\n=== Daily Reminders ===")
    daily_reminders = agent.get_daily_reminders(user_id)
    print(f"Today's reminders for {user_id}:")
    for reminder in daily_reminders:
        print(f"  {reminder['scheduled_time']} - {reminder['type']}: {reminder['message']}")
        print(f"    Compliance Probability: {reminder['compliance_probability']:.2f}")
        print(f"    Risk Level: {reminder['risk_level']}")
    
    # Test sending and acknowledging a reminder
    print("\n=== Testing Reminder Flow ===")
    reminder_data = agent.send_reminder(user_id, 'Medication', 'Time to take your morning medication', priority=1)
    print(f"Sent reminder: {reminder_data['reminder_id']}")
    
    # Acknowledge the reminder
    ack_result = agent.acknowledge_reminder(reminder_data['reminder_id'], user_id)
    print(f"Acknowledgment result: {ack_result}")
    
    # Get compliance insights
    print("\n=== Compliance Insights ===")
    insights = agent.get_compliance_insights(user_id, days=30)
    print("Compliance Insights:")
    for key, value in insights.items():
        if key not in ['compliance_by_type', 'compliance_by_hour', 'compliance_by_weekday']:
            print(f"  {key}: {value}")
    
    print("\nCompliance by Type:")
    for reminder_type, stats in insights['compliance_by_type'].items():
        print(f"  {reminder_type}: {stats['compliance_rate']:.2f} compliance rate")