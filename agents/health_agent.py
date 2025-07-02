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

class HealthAgent:
    """
    Advanced Health Monitoring Agent for Elderly Care
    
    This agent learns from historical health data to identify anomalies and flag
    potential health risks for elderly individuals. It uses machine learning
    techniques to detect patterns and predict health alerts.
    """
    
    def __init__(self, data_path: Optional[str] = None, model_save_path: str = "models"):
        """
        Initialize the Health Agent
        
        Args:
            data_path: Path to the health monitoring CSV file
            model_save_path: Directory to save trained models
        """
        self.data_path = data_path or "data/health_monitoring.csv"
        self.model_save_path = model_save_path
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.alert_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_columns = []
        self.is_trained = False
        
        # Health thresholds (normal ranges for elderly)
        self.normal_ranges = {
            'heart_rate': (60, 100),
            'systolic_bp': (90, 140),
            'diastolic_bp': (60, 90),
            'glucose': (70, 140),
            'oxygen_saturation': (95, 100)
        }
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_save_path, exist_ok=True)
        
    def load_and_preprocess_data(self) -> pd.DataFrame:
        """
        Load and preprocess the health monitoring dataset
        
        Returns:
            Preprocessed DataFrame
        """
        try:
            # Load the dataset
            df = pd.read_csv(self.data_path)
            logger.info(f"Loaded dataset with {len(df)} records")
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Convert timestamp to datetime
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            
            # Extract blood pressure values
            bp_split = df['Blood Pressure'].str.extract(r'(\d+)/(\d+)')
            df['Systolic_BP'] = pd.to_numeric(bp_split[0])
            df['Diastolic_BP'] = pd.to_numeric(bp_split[1])
            
            # Convert Yes/No columns to binary
            yes_no_columns = [col for col in df.columns if 'Yes/No' in col or 'Threshold' in col]
            for col in yes_no_columns:
                df[col] = (df[col] == 'Yes').astype(int)
            
            # Create time-based features
            df['Hour'] = df['Timestamp'].dt.hour
            df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
            df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
            
            # Create derived features
            df['Pulse_Pressure'] = df['Systolic_BP'] - df['Diastolic_BP']
            df['MAP'] = (df['Systolic_BP'] + 2 * df['Diastolic_BP']) / 3  # Mean Arterial Pressure
            
            # Calculate deviation from normal ranges
            df['HR_Deviation'] = self._calculate_deviation(df['Heart Rate'], 'heart_rate')
            df['Glucose_Deviation'] = self._calculate_deviation(df['Glucose Levels'], 'glucose')
            df['SpO2_Deviation'] = self._calculate_deviation(df['Oxygen Saturation (SpO₂%)'], 'oxygen_saturation')
            
            # Count total threshold violations per record
            threshold_cols = [col for col in df.columns if 'Threshold' in col]
            df['Total_Violations'] = df[threshold_cols].sum(axis=1)
            
            logger.info(f"Data preprocessing completed. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _calculate_deviation(self, values: pd.Series, metric_type: str) -> pd.Series:
        """
        Calculate deviation from normal range as a percentage
        
        Args:
            values: Series of values
            metric_type: Type of metric for threshold lookup
            
        Returns:
            Series of deviation percentages
        """
        min_val, max_val = self.normal_ranges[metric_type]
        
        # Calculate deviation
        deviation = np.where(
            values < min_val, (min_val - values) / min_val * 100,
            np.where(values > max_val, (values - max_val) / max_val * 100, 0)
        )
        
        return pd.Series(deviation)
    
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
            'Heart Rate', 'Systolic_BP', 'Diastolic_BP', 'Glucose Levels',
            'Oxygen Saturation (SpO₂%)', 'Hour', 'DayOfWeek', 'IsWeekend',
            'Pulse_Pressure', 'MAP', 'HR_Deviation', 'Glucose_Deviation',
            'SpO2_Deviation', 'Total_Violations'
        ]
        
        self.feature_columns = feature_columns
        X = df[feature_columns].values
        y = df['Alert Triggered (Yes/No)'].values
        
        return X, np.array(y)
    
    def train_models(self, df: Optional[pd.DataFrame] = None):
        """
        Train anomaly detection and alert prediction models
        
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
        
        # Train anomaly detection model
        logger.info("Training anomaly detection model...")
        self.anomaly_detector.fit(X_train_scaled)
        
        # Train alert prediction model
        logger.info("Training alert prediction model...")
        self.alert_predictor.fit(X_train_scaled, y_train)
        
        # Evaluate models
        self._evaluate_models(X_test_scaled, y_test)
        
        # Save models
        self._save_models()
        
        self.is_trained = True
        logger.info("Model training completed successfully!")
    
    def _evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Evaluate the trained models
        
        Args:
            X_test: Test features
            y_test: Test targets
        """
        # Evaluate alert prediction
        y_pred = self.alert_predictor.predict(X_test)
        
        logger.info("Alert Prediction Model Performance:")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.alert_predictor.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 10 Most Important Features:")
        logger.info(f"\n{feature_importance.head(10)}")
        
        # Anomaly detection evaluation
        anomaly_scores = self.anomaly_detector.decision_function(X_test)
        anomalies = self.anomaly_detector.predict(X_test)
        
        anomaly_rate = (anomalies == -1).mean()
        logger.info(f"Anomaly detection rate: {anomaly_rate:.3f}")
    
    def predict_health_status(self, health_data: Dict) -> Dict:
        """
        Predict health status and potential alerts for new data
        
        Args:
            health_data: Dictionary containing health measurements
            
        Returns:
            Dictionary with predictions and risk assessment
        """
        if not self.is_trained:
            self._load_models()
        
        try:
            # Convert input to DataFrame for consistent preprocessing
            df_input = pd.DataFrame([health_data])
            
            # Add time-based features if timestamp provided
            if 'timestamp' in health_data:
                timestamp = pd.to_datetime(health_data['timestamp'])
                df_input['Hour'] = timestamp.hour
                df_input['DayOfWeek'] = timestamp.dayofweek
                df_input['IsWeekend'] = int(timestamp.dayofweek >= 5)
            else:
                current_time = datetime.now()
                df_input['Hour'] = current_time.hour
                df_input['DayOfWeek'] = current_time.weekday()
                df_input['IsWeekend'] = int(current_time.weekday() >= 5)
            
            # Extract blood pressure if provided as string
            if 'blood_pressure' in health_data and isinstance(health_data['blood_pressure'], str):
                bp_parts = health_data['blood_pressure'].split('/')
                df_input['Systolic_BP'] = int(bp_parts[0])
                df_input['Diastolic_BP'] = int(bp_parts[1])
            else:
                df_input['Systolic_BP'] = health_data.get('systolic_bp', 120)
                df_input['Diastolic_BP'] = health_data.get('diastolic_bp', 80)
            
            # Calculate derived features
            df_input['Pulse_Pressure'] = df_input['Systolic_BP'] - df_input['Diastolic_BP']
            df_input['MAP'] = (df_input['Systolic_BP'] + 2 * df_input['Diastolic_BP']) / 3
            
            # Calculate deviations
            df_input['HR_Deviation'] = self._calculate_deviation(
                pd.Series([health_data.get('heart_rate', 70)]), 'heart_rate'
            )[0]
            df_input['Glucose_Deviation'] = self._calculate_deviation(
                pd.Series([health_data.get('glucose', 100)]), 'glucose'
            )[0]
            df_input['SpO2_Deviation'] = self._calculate_deviation(
                pd.Series([health_data.get('oxygen_saturation', 98)]), 'oxygen_saturation'
            )[0]
            
            # Map input fields to expected feature names
            feature_mapping = {
                'heart_rate': 'Heart Rate',
                'glucose': 'Glucose Levels',
                'oxygen_saturation': 'Oxygen Saturation (SpO₂%)'
            }
            
            for input_key, feature_name in feature_mapping.items():
                if input_key in health_data:
                    df_input[feature_name] = health_data[input_key]
            
            # Count threshold violations
            violations = 0
            if health_data.get('heart_rate', 70) < 60 or health_data.get('heart_rate', 70) > 100:
                violations += 1
            if df_input['Systolic_BP'].iloc[0] < 90 or df_input['Systolic_BP'].iloc[0] > 140:
                violations += 1
            if df_input['Diastolic_BP'].iloc[0] < 60 or df_input['Diastolic_BP'].iloc[0] > 90:
                violations += 1
            if health_data.get('glucose', 100) < 70 or health_data.get('glucose', 100) > 140:
                violations += 1
            if health_data.get('oxygen_saturation', 98) < 95:
                violations += 1
            
            df_input['Total_Violations'] = violations
            
            # Prepare features
            X = df_input[self.feature_columns].values
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            alert_probability = self.alert_predictor.predict_proba(X_scaled)[0]
            alert_prediction = self.alert_predictor.predict(X_scaled)[0]
            anomaly_score = self.anomaly_detector.decision_function(X_scaled)[0]
            is_anomaly = self.anomaly_detector.predict(X_scaled)[0] == -1
            
            # Risk assessment
            risk_level = self._assess_risk_level(alert_probability[1], anomaly_score, violations)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(health_data, violations, risk_level)
            
            return {
                'alert_required': bool(alert_prediction),
                'alert_probability': float(alert_probability[1]),
                'is_anomaly': bool(is_anomaly),
                'anomaly_score': float(anomaly_score),
                'risk_level': risk_level,
                'threshold_violations': int(violations),
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return {
                'error': str(e),
                'alert_required': True,  # Safe default
                'risk_level': 'high',
                'timestamp': datetime.now().isoformat()
            }
    
    def _assess_risk_level(self, alert_prob: float, anomaly_score: float, violations: int) -> str:
        """
        Assess overall risk level based on multiple factors
        
        Args:
            alert_prob: Probability of alert
            anomaly_score: Anomaly detection score
            violations: Number of threshold violations
            
        Returns:
            Risk level string
        """
        if alert_prob > 0.8 or violations >= 3 or anomaly_score < -0.5:
            return 'critical'
        elif alert_prob > 0.6 or violations >= 2 or anomaly_score < -0.3:
            return 'high'
        elif alert_prob > 0.4 or violations >= 1 or anomaly_score < -0.1:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendations(self, health_data: Dict, violations: int, risk_level: str) -> List[str]:
        """
        Generate health recommendations based on current status
        
        Args:
            health_data: Current health measurements
            violations: Number of threshold violations
            risk_level: Assessed risk level
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Heart rate recommendations
        hr = health_data.get('heart_rate', 70)
        if hr < 60:
            recommendations.append("Heart rate is below normal. Consider consulting a physician about bradycardia.")
        elif hr > 100:
            recommendations.append("Heart rate is elevated. Consider rest and hydration. Seek medical attention if persistent.")
        
        # Blood pressure recommendations
        systolic = health_data.get('systolic_bp', 120)
        diastolic = health_data.get('diastolic_bp', 80)
        
        if systolic > 140 or diastolic > 90:
            recommendations.append("Blood pressure is elevated. Monitor closely and consider medication review.")
        elif systolic < 90:
            recommendations.append("Blood pressure is low. Stay hydrated and avoid sudden position changes.")
        
        # Glucose recommendations
        glucose = health_data.get('glucose', 100)
        if glucose > 140:
            recommendations.append("Blood glucose is elevated. Monitor diet and consider diabetes management.")
        elif glucose < 70:
            recommendations.append("Blood glucose is low. Consider consuming a quick-acting carbohydrate.")
        
        # Oxygen saturation recommendations
        spo2 = health_data.get('oxygen_saturation', 98)
        if spo2 < 95:
            recommendations.append("Oxygen saturation is low. Ensure proper breathing and consider oxygen therapy.")
        
        # General recommendations based on risk level
        if risk_level == 'critical':
            recommendations.append("URGENT: Multiple critical parameters detected. Seek immediate medical attention.")
        elif risk_level == 'high':
            recommendations.append("High risk detected. Contact healthcare provider within 24 hours.")
        elif risk_level == 'medium':
            recommendations.append("Some concerns detected. Monitor closely and schedule routine check-up.")
        
        return recommendations
    
    def _save_models(self):
        """Save trained models to disk"""
        model_files = {
            'scaler': self.scaler,
            'anomaly_detector': self.anomaly_detector,
            'alert_predictor': self.alert_predictor,
            'feature_columns': self.feature_columns,
            'normal_ranges': self.normal_ranges
        }
        
        for name, model in model_files.items():
            with open(f"{self.model_save_path}/{name}.pkl", 'wb') as f:
                pickle.dump(model, f)
        
        logger.info(f"Models saved to {self.model_save_path}/")
    
    def _load_models(self):
        """Load trained models from disk"""
        try:
            model_files = ['scaler', 'anomaly_detector', 'alert_predictor', 'feature_columns', 'normal_ranges']
            
            for name in model_files:
                with open(f"{self.model_save_path}/{name}.pkl", 'rb') as f:
                    setattr(self, name, pickle.load(f))
            
            self.is_trained = True
            logger.info("Models loaded successfully!")
            
        except FileNotFoundError:
            logger.warning("No saved models found. Please train the models first.")
            self.is_trained = False
    
    def get_health_insights(self, device_id: Optional[str] = None, days: int = 7) -> Dict:
        """
        Get health insights and trends for a specific device/user
        
        Args:
            device_id: Device ID to analyze
            days: Number of days to look back
            
        Returns:
            Dictionary with health insights
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
            
            # Calculate trends
            insights = {
                'device_id': device_id,
                'analysis_period': f"{days} days",
                'total_readings': len(df_recent),
                'alert_rate': df_recent['Alert Triggered (Yes/No)'].mean(),
                'avg_heart_rate': df_recent['Heart Rate'].mean(),
                'avg_systolic_bp': df_recent['Systolic_BP'].mean(),
                'avg_diastolic_bp': df_recent['Diastolic_BP'].mean(),
                'avg_glucose': df_recent['Glucose Levels'].mean(),
                'avg_oxygen_sat': df_recent['Oxygen Saturation (SpO₂%)'].mean(),
                'total_violations': df_recent['Total_Violations'].sum(),
                'most_common_violation_time': df_recent.groupby('Hour')['Total_Violations'].sum().idxmax()
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {"error": str(e)}

# Example usage and testing
if __name__ == "__main__":
    # Initialize health agent
    agent = HealthAgent(data_path="data/health_monitoring.csv")
    
    # Train the models
    print("Training health monitoring models...")
    agent.train_models()
    
    # Test with sample data
    sample_health_data = {
        'heart_rate': 85,
        'systolic_bp': 130,
        'diastolic_bp': 85,
        'glucose': 110,
        'oxygen_saturation': 97,
        'timestamp': '2025-07-02 14:30:00'
    }
    
    print("\nTesting with sample health data...")
    result = agent.predict_health_status(sample_health_data)
    print("Prediction Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    # Get health insights
    print("\nGetting health insights...")
    insights = agent.get_health_insights(days=30)
    print("Health Insights:")
    for key, value in insights.items():
        print(f"  {key}: {value}")