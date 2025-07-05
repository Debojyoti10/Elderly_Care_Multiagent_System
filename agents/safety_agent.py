import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime, timedelta
import pickle
import os
import logging
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SafetyAgent:
    """
    Advanced Safety Monitoring Agent for Elderly Care
    
    This agent focuses on fall detection, home safety monitoring, and emergency response
    coordination. It uses machine learning to detect dangerous situations and predict
    safety risks for elderly individuals.
    """
    
    def __init__(self, data_path: Optional[str] = None, model_save_path: str = "models"):
        """
        Initialize the Safety Agent
        
        Args:
            data_path: Path to the safety monitoring CSV file
            model_save_path: Directory to save trained models
        """
        self.data_path = data_path or "data/safety_monitoring.csv"
        self.model_save_path = model_save_path
        self.scaler = StandardScaler()
        self.fall_detector = RandomForestClassifier(n_estimators=100, random_state=42)
        self.risk_assessor = IsolationForest(contamination=0.1, random_state=42)
        self.location_encoder = LabelEncoder()
        self.activity_encoder = LabelEncoder()
        self.feature_columns = []
        self.is_trained = False
        
        # Safety thresholds and risk factors
        self.risk_factors = {
            'high_risk_locations': ['Bathroom', 'Kitchen'],
            'high_risk_activities': ['No Movement', 'Lying'],
            'critical_inactivity_threshold': 300,  # 5 minutes
            'impact_force_weights': {'Low': 1, 'Medium': 2, 'High': 3},
            'time_based_risks': {
                'night_hours': (22, 6),  # 10 PM to 6 AM
                'high_risk_hours': [1, 2, 3, 4, 5]  # Early morning hours
            }
        }
        
        # Emergency response configuration
        self.emergency_protocols = {
            'immediate_response': {
                'conditions': ['High impact fall', 'Long inactivity after fall'],
                'actions': ['Call emergency services', 'Notify primary caregiver', 'Alert neighbors']
            },
            'urgent_response': {
                'conditions': ['Medium impact fall', 'Bathroom fall', 'Night fall'],
                'actions': ['Notify primary caregiver', 'Schedule wellness check']
            },
            'monitoring_response': {
                'conditions': ['Low impact fall', 'Anomaly detected'],
                'actions': ['Increase monitoring', 'Log for review']
            }
        }
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_save_path, exist_ok=True)
    
    def load_and_preprocess_data(self) -> pd.DataFrame:
        """
        Load and preprocess the safety monitoring dataset
        
        Returns:
            Preprocessed DataFrame
        """
        try:
            # Load the dataset
            df = pd.read_csv(self.data_path)
            logger.info(f"Loaded safety dataset with {len(df)} records")
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Convert timestamp to datetime
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            # Handle missing values and clean data
            df['Impact Force Level'] = df['Impact Force Level'].fillna('None')
            df['Post-Fall Inactivity Duration (Seconds)'] = pd.to_numeric(
                df['Post-Fall Inactivity Duration (Seconds)'], errors='coerce'
            ).fillna(0)
            
            # Convert Yes/No columns to binary
            yes_no_columns = ['Fall Detected (Yes/No)', 'Alert Triggered (Yes/No)', 'Caregiver Notified (Yes/No)']
            for col in yes_no_columns:
                df[col] = (df[col] == 'Yes').astype(int)
            
            # Create time-based features
            df['Hour'] = df['Timestamp'].dt.hour
            df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
            df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
            df['IsNightTime'] = ((df['Hour'] >= 22) | (df['Hour'] <= 6)).astype(int)
            df['IsHighRiskHour'] = df['Hour'].isin(self.risk_factors['time_based_risks']['high_risk_hours']).astype(int)
            
            # Location-based risk features
            df['IsHighRiskLocation'] = df['Location'].isin(self.risk_factors['high_risk_locations']).astype(int)
            df['IsBathroom'] = (df['Location'] == 'Bathroom').astype(int)
            df['IsKitchen'] = (df['Location'] == 'Kitchen').astype(int)
            
            # Activity-based risk features
            df['IsHighRiskActivity'] = df['Movement Activity'].isin(self.risk_factors['high_risk_activities']).astype(int)
            df['IsNoMovement'] = (df['Movement Activity'] == 'No Movement').astype(int)
            df['IsLying'] = (df['Movement Activity'] == 'Lying').astype(int)
            
            # Impact force numerical encoding
            df['Impact Force Numeric'] = df['Impact Force Level'].map({
                'None': 0, 'Low': 1, 'Medium': 2, 'High': 3
            }).fillna(0)
            
            # Inactivity risk assessment
            df['Is_Long_Inactivity'] = (df['Post-Fall Inactivity Duration (Seconds)'] >= 
                                      self.risk_factors['critical_inactivity_threshold']).astype(int)
            
            # Composite risk scores
            df['Location_Risk_Score'] = (df['IsHighRiskLocation'] * 2 + 
                                       df['IsBathroom'] * 1.5 + 
                                       df['IsKitchen'] * 1.2)
            
            df['Time_Risk_Score'] = (df['IsNightTime'] * 2 + 
                                   df['IsHighRiskHour'] * 1.5 + 
                                   df['IsWeekend'] * 0.5)
            
            df['Activity_Risk_Score'] = (df['IsHighRiskActivity'] * 2 + 
                                       df['IsNoMovement'] * 1.5 + 
                                       df['IsLying'] * 1.2)
            
            # Overall risk score
            df['Overall_Risk_Score'] = (df['Location_Risk_Score'] + 
                                      df['Time_Risk_Score'] + 
                                      df['Activity_Risk_Score'] + 
                                      df['Impact Force Numeric'] + 
                                      df['Is_Long_Inactivity'] * 3)
            
            # Encode categorical variables
            df['Location_Encoded'] = self.location_encoder.fit_transform(df['Location'])
            df['Activity_Encoded'] = self.activity_encoder.fit_transform(df['Movement Activity'])
            
            logger.info(f"Data preprocessing completed. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for machine learning models
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            Feature matrix and target array
        """
        # Select relevant features for training
        feature_columns = [
            'Hour', 'DayOfWeek', 'IsWeekend', 'IsNightTime', 'IsHighRiskHour',
            'Location_Encoded', 'Activity_Encoded', 'IsHighRiskLocation', 
            'IsBathroom', 'IsKitchen', 'IsHighRiskActivity', 'IsNoMovement', 
            'IsLying', 'Impact Force Numeric', 'Post-Fall Inactivity Duration (Seconds)',
            'Is_Long_Inactivity', 'Location_Risk_Score', 'Time_Risk_Score',
            'Activity_Risk_Score', 'Overall_Risk_Score'
        ]
        
        self.feature_columns = feature_columns
        X = df[feature_columns].values
        y = df['Fall Detected (Yes/No)'].values
        
        return X, np.array(y)
    
    def train_models(self, df: Optional[pd.DataFrame] = None):
        """
        Train fall detection and safety risk assessment models
        
        Args:
            df: Preprocessed DataFrame (optional, will load if not provided)
        """
        if df is None:
            df = self.load_and_preprocess_data()
        
        X, y = self.prepare_features(df)
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train fall detection model
        logger.info("Training fall detection model...")
        self.fall_detector.fit(X_train_scaled, y_train)
        
        # Train risk assessment model (anomaly detection)
        logger.info("Training safety risk assessment model...")
        self.risk_assessor.fit(X_train_scaled)
        
        # Evaluate models
        self._evaluate_models(X_test_scaled, y_test)
        
        # Save models
        self._save_models()
        
        self.is_trained = True
        logger.info("Safety model training completed successfully!")
    
    def _evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Evaluate the trained models
        
        Args:
            X_test: Test features
            y_test: Test targets
        """
        # Evaluate fall detection
        y_pred = self.fall_detector.predict(X_test)
        
        logger.info("Fall Detection Model Performance:")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.fall_detector.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 10 Most Important Safety Features:")
        logger.info(f"\n{feature_importance.head(10)}")
        
        # Risk assessment evaluation
        risk_scores = self.risk_assessor.decision_function(X_test)
        anomalies = self.risk_assessor.predict(X_test)
        
        anomaly_rate = (anomalies == -1).mean()
        logger.info(f"Safety risk anomaly detection rate: {anomaly_rate:.3f}")
    
    def assess_safety_status(self, safety_data: Dict) -> Dict:
        """
        Assess safety status and fall risk for new data
        
        Args:
            safety_data: Dictionary containing safety monitoring data
            
        Returns:
            Dictionary with safety assessment and recommendations
        """
        if not self.is_trained:
            self._load_models()
        
        try:
            # Convert input to DataFrame for consistent preprocessing
            df_input = pd.DataFrame([safety_data])
            
            # Add time-based features if timestamp provided
            if 'timestamp' in safety_data:
                timestamp = pd.to_datetime(safety_data['timestamp'])
                df_input['Hour'] = timestamp.hour
                df_input['DayOfWeek'] = timestamp.dayofweek
                df_input['IsWeekend'] = int(timestamp.dayofweek >= 5)
                df_input['IsNightTime'] = int((timestamp.hour >= 22) or (timestamp.hour <= 6))
                df_input['IsHighRiskHour'] = int(timestamp.hour in self.risk_factors['time_based_risks']['high_risk_hours'])
            else:
                current_time = datetime.now()
                df_input['Hour'] = current_time.hour
                df_input['DayOfWeek'] = current_time.weekday()
                df_input['IsWeekend'] = int(current_time.weekday() >= 5)
                df_input['IsNightTime'] = int((current_time.hour >= 22) or (current_time.hour <= 6))
                df_input['IsHighRiskHour'] = int(current_time.hour in self.risk_factors['time_based_risks']['high_risk_hours'])
            
            # Location features
            location = safety_data.get('location', 'Living Room')
            try:
                location_encoded = self.location_encoder.transform([location])
                df_input['Location_Encoded'] = np.asarray(location_encoded).item()
            except ValueError:
                # Handle unseen location by using the most common one
                df_input['Location_Encoded'] = 0
            df_input['IsHighRiskLocation'] = int(location in self.risk_factors['high_risk_locations'])
            df_input['IsBathroom'] = int(location == 'Bathroom')
            df_input['IsKitchen'] = int(location == 'Kitchen')
            
            # Activity features
            activity = safety_data.get('movement_activity', 'Walking')
            try:
                activity_encoded = self.activity_encoder.transform([activity])
                df_input['Activity_Encoded'] = np.asarray(activity_encoded).item()
            except ValueError:
                # Handle unseen activity by using the most common one
                df_input['Activity_Encoded'] = 0
            df_input['IsHighRiskActivity'] = int(activity in self.risk_factors['high_risk_activities'])
            df_input['IsNoMovement'] = int(activity == 'No Movement')
            df_input['IsLying'] = int(activity == 'Lying')
            
            # Impact and inactivity features
            impact_force = safety_data.get('impact_force', 'None')
            df_input['Impact Force Numeric'] = self.risk_factors['impact_force_weights'].get(impact_force, 0)
            
            inactivity_duration = safety_data.get('post_fall_inactivity', 0)
            df_input['Post-Fall Inactivity Duration (Seconds)'] = inactivity_duration
            df_input['Is_Long_Inactivity'] = int(inactivity_duration >= self.risk_factors['critical_inactivity_threshold'])
            
            # Calculate risk scores
            df_input['Location_Risk_Score'] = (df_input['IsHighRiskLocation'] * 2 + 
                                             df_input['IsBathroom'] * 1.5 + 
                                             df_input['IsKitchen'] * 1.2)
            
            df_input['Time_Risk_Score'] = (df_input['IsNightTime'] * 2 + 
                                         df_input['IsHighRiskHour'] * 1.5 + 
                                         df_input['IsWeekend'] * 0.5)
            
            df_input['Activity_Risk_Score'] = (df_input['IsHighRiskActivity'] * 2 + 
                                             df_input['IsNoMovement'] * 1.5 + 
                                             df_input['IsLying'] * 1.2)
            
            df_input['Overall_Risk_Score'] = (df_input['Location_Risk_Score'] + 
                                            df_input['Time_Risk_Score'] + 
                                            df_input['Activity_Risk_Score'] + 
                                            df_input['Impact Force Numeric'] + 
                                            df_input['Is_Long_Inactivity'] * 3)
            
            # Prepare features
            X = df_input[self.feature_columns].values
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            fall_probability = self.fall_detector.predict_proba(X_scaled)[0]
            fall_prediction = self.fall_detector.predict(X_scaled)[0]
            risk_score = self.risk_assessor.decision_function(X_scaled)[0]
            is_anomaly = self.risk_assessor.predict(X_scaled)[0] == -1
            
            # Assess emergency level
            emergency_level = self._assess_emergency_level(
                fall_probability[1], 
                impact_force, 
                inactivity_duration, 
                location, 
                df_input['IsNightTime'].iloc[0]
            )
            
            # Generate safety recommendations
            recommendations = self._generate_safety_recommendations(
                safety_data, 
                fall_probability[1], 
                emergency_level
            )
            
            # Determine required actions
            required_actions = self._determine_required_actions(
                emergency_level, 
                fall_prediction, 
                is_anomaly
            )
            
            return {
                'fall_detected': bool(fall_prediction),
                'fall_probability': float(fall_probability[1]),
                'emergency_level': emergency_level,
                'is_safety_anomaly': bool(is_anomaly),
                'risk_score': float(risk_score),
                'overall_risk_score': float(df_input['Overall_Risk_Score'].iloc[0]),
                'location_risk': float(df_input['Location_Risk_Score'].iloc[0]),
                'time_risk': float(df_input['Time_Risk_Score'].iloc[0]),
                'activity_risk': float(df_input['Activity_Risk_Score'].iloc[0]),
                'recommendations': recommendations,
                'required_actions': required_actions,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in safety assessment: {e}")
            return {
                'error': str(e),
                'emergency_level': 'critical',  # Safe default
                'fall_detected': True,
                'required_actions': ['immediate_medical_attention'],
                'timestamp': datetime.now().isoformat()
            }
    
    def _assess_emergency_level(self, fall_prob: float, impact_force: str, 
                               inactivity_duration: int, location: str, is_night: bool) -> str:
        """
        Assess the emergency level based on multiple factors
        
        Args:
            fall_prob: Fall probability
            impact_force: Impact force level
            inactivity_duration: Duration of inactivity after potential fall
            location: Location of incident
            is_night: Whether it's nighttime
            
        Returns:
            Emergency level string
        """
        # Critical conditions
        if (impact_force == 'High' or 
            inactivity_duration >= 600 or  # 10 minutes
            (fall_prob > 0.8 and location == 'Bathroom')):
            return 'critical'
        
        # High emergency conditions
        if (impact_force == 'Medium' or 
            inactivity_duration >= 300 or  # 5 minutes
            fall_prob > 0.7 or
            (fall_prob > 0.5 and is_night)):
            return 'high'
        
        # Medium emergency conditions
        if (impact_force == 'Low' or 
            inactivity_duration >= 60 or  # 1 minute
            fall_prob > 0.4 or
            location in self.risk_factors['high_risk_locations']):
            return 'medium'
        
        return 'low'
    
    def _generate_safety_recommendations(self, safety_data: Dict, 
                                       fall_prob: float, emergency_level: str) -> List[str]:
        """
        Generate safety recommendations based on current status
        
        Args:
            safety_data: Current safety monitoring data
            fall_prob: Fall probability
            emergency_level: Assessed emergency level
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Location-specific recommendations
        location = safety_data.get('location', 'Unknown')
        if location == 'Bathroom':
            recommendations.append("Install bathroom safety rails and non-slip mats")
            recommendations.append("Ensure adequate lighting in bathroom")
        elif location == 'Kitchen':
            recommendations.append("Clear walkways of clutter and obstacles")
            recommendations.append("Use non-slip mats near sink and stove areas")
        
        # Activity-specific recommendations
        activity = safety_data.get('movement_activity', 'Unknown')
        if activity == 'No Movement':
            recommendations.append("Extended inactivity detected - wellness check recommended")
        elif activity == 'Lying':
            recommendations.append("Monitor for signs of difficulty getting up")
        
        # Time-based recommendations
        if safety_data.get('timestamp'):
            timestamp = pd.to_datetime(safety_data['timestamp'])
            if timestamp.hour >= 22 or timestamp.hour <= 6:
                recommendations.append("Use nightlights for safe navigation in darkness")
                recommendations.append("Keep path to bathroom well-lit and clear")
        
        # Emergency level recommendations
        if emergency_level == 'critical':
            recommendations.append("IMMEDIATE ACTION REQUIRED - Contact emergency services")
        elif emergency_level == 'high':
            recommendations.append("Urgent attention needed - Contact caregiver immediately")
        elif emergency_level == 'medium':
            recommendations.append("Increased monitoring recommended")
        
        # Fall probability recommendations
        if fall_prob > 0.6:
            recommendations.append("High fall risk - consider mobility assistance")
            recommendations.append("Review medication for side effects affecting balance")
        
        return recommendations
    
    def _determine_required_actions(self, emergency_level: str, 
                                  fall_detected: bool, is_anomaly: bool) -> List[str]:
        """
        Determine required actions based on assessment
        
        Args:
            emergency_level: Emergency level
            fall_detected: Whether a fall was detected
            is_anomaly: Whether an anomaly was detected
            
        Returns:
            List of required actions
        """
        actions = []
        
        if emergency_level == 'critical':
            actions.extend([
                'call_emergency_services',
                'notify_primary_caregiver',
                'alert_backup_contacts',
                'activate_emergency_protocol'
            ])
        elif emergency_level == 'high':
            actions.extend([
                'notify_primary_caregiver',
                'schedule_immediate_check',
                'increase_monitoring_frequency'
            ])
        elif emergency_level == 'medium':
            actions.extend([
                'notify_caregiver',
                'schedule_wellness_check',
                'log_incident'
            ])
        else:
            actions.extend([
                'log_for_review',
                'continue_monitoring'
            ])
        
        if fall_detected:
            actions.append('document_fall_incident')
        
        if is_anomaly:
            actions.append('investigate_anomaly')
        
        return actions
    
    def get_safety_insights(self, device_id: Optional[str] = None, days: int = 7) -> Dict:
        """
        Get safety insights and trends for a specific device/user
        
        Args:
            device_id: Device ID to analyze
            days: Number of days to look back
            
        Returns:
            Dictionary with safety insights
        """
        try:
            df = self.load_and_preprocess_data()
            
            if device_id:
                df = df[df['Device-ID/User-ID'] == device_id]
            
            # Filter by date range
            end_date = df['Timestamp'].max()
            start_date = end_date - timedelta(days=days)
            df_recent = df[df['Timestamp'] >= start_date]
            
            if len(df_recent) == 0:
                return {"error": "No recent data found"}
            
            # Calculate safety metrics
            insights = {
                'device_id': device_id,
                'analysis_period': f"{days} days",
                'total_monitoring_events': len(df_recent),
                'falls_detected': df_recent['Fall Detected (Yes/No)'].sum(),
                'fall_rate': df_recent['Fall Detected (Yes/No)'].mean(),
                'alerts_triggered': df_recent['Alert Triggered (Yes/No)'].sum(),
                'avg_risk_score': df_recent['Overall_Risk_Score'].mean(),
                'most_common_location': df_recent['Location'].mode().iloc[0] if not df_recent['Location'].mode().empty else 'Unknown',
                'most_common_activity': df_recent['Movement Activity'].mode().iloc[0] if not df_recent['Movement Activity'].mode().empty else 'Unknown',
                'high_risk_times': df_recent[df_recent['IsHighRiskHour'] == 1]['Hour'].tolist(),
                'bathroom_incidents': len(df_recent[df_recent['IsBathroom'] == 1]),
                'night_incidents': len(df_recent[df_recent['IsNightTime'] == 1]),
                'avg_inactivity_duration': df_recent['Post-Fall Inactivity Duration (Seconds)'].mean()
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating safety insights: {e}")
            return {"error": str(e)}
    
    def _save_models(self):
        """Save trained models to disk"""
        model_files = {
            'scaler': self.scaler,
            'fall_detector': self.fall_detector,
            'risk_assessor': self.risk_assessor,
            'location_encoder': self.location_encoder,
            'activity_encoder': self.activity_encoder,
            'feature_columns': self.feature_columns,
            'risk_factors': self.risk_factors
        }
        
        for name, model in model_files.items():
            with open(f"{self.model_save_path}/safety_{name}.pkl", 'wb') as f:
                pickle.dump(model, f)
        
        logger.info(f"Safety models saved to {self.model_save_path}/")
    
    def _load_models(self):
        """Load trained models from disk"""
        try:
            model_files = ['scaler', 'fall_detector', 'risk_assessor', 
                          'location_encoder', 'activity_encoder', 'feature_columns', 'risk_factors']
            
            for name in model_files:
                with open(f"{self.model_save_path}/safety_{name}.pkl", 'rb') as f:
                    setattr(self, name, pickle.load(f))
            
            self.is_trained = True
            logger.info("Safety models loaded successfully!")
            
        except FileNotFoundError:
            logger.warning("No saved safety models found. Please train the models first.")
            self.is_trained = False

# Example usage and testing
if __name__ == "__main__":
    # Initialize safety agent
    agent = SafetyAgent(data_path="data/safety_monitoring.csv")
    
    # Train the models
    print("Training safety monitoring models...")
    agent.train_models()
    
    # Test with sample data - Normal activity
    print("\n=== Testing Normal Activity ===")
    sample_normal = {
        'movement_activity': 'Walking',
        'location': 'Living Room',
        'impact_force': 'None',
        'post_fall_inactivity': 0,
        'timestamp': '2025-07-03 14:30:00'
    }
    
    result = agent.assess_safety_status(sample_normal)
    print("Normal Activity Assessment:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    # Test with sample data - High risk scenario
    print("\n=== Testing High Risk Scenario ===")
    sample_high_risk = {
        'movement_activity': 'No Movement',
        'location': 'Bathroom',
        'impact_force': 'Medium',
        'post_fall_inactivity': 400,
        'timestamp': '2025-07-03 02:30:00'  # Night time
    }
    
    result = agent.assess_safety_status(sample_high_risk)
    print("High Risk Assessment:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    # Test with critical emergency
    print("\n=== Testing Critical Emergency ===")
    sample_critical = {
        'movement_activity': 'No Movement',
        'location': 'Kitchen',
        'impact_force': 'High',
        'post_fall_inactivity': 650,
        'timestamp': '2025-07-03 03:15:00'
    }
    
    result = agent.assess_safety_status(sample_critical)
    print("Critical Emergency Assessment:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    # Get safety insights
    print("\n=== Safety Insights ===")
    insights = agent.get_safety_insights(days=30)
    print("Safety Insights:")
    for key, value in insights.items():
        print(f"  {key}: {value}")