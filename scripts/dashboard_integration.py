"""
Dashboard Integration Functions for Student Performance ML Models

This module provides functions to integrate the trained ML models with the 
Student Dashboard application, enabling real-time predictions and insights.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
from typing import Dict, List, Tuple, Any

class StudentPerformancePredictor:
    """
    Main class for student performance prediction and persona classification.
    Integrates with the dashboard to provide real-time ML insights.
    """
    
    def __init__(self):
        self.model = None
        self.clustering_model = None
        self.scaler = None
        self.clustering_scaler = None
        self.feature_columns = ['comprehension', 'attention', 'focus', 'retention', 'engagement_time']
        self.cognitive_columns = ['comprehension', 'attention', 'focus', 'retention']
        self.persona_mapping = {
            0: "High Performers",
            1: "Strong Achievers", 
            2: "Developing Learners",
            3: "Support Needed"
        }
        self.is_trained = False
    
    def train_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train both prediction and clustering models on the provided data.
        
        Args:
            data (pd.DataFrame): Student data with required columns
            
        Returns:
            Dict[str, Any]: Training results and performance metrics
        """
        # Prepare features and target
        X = data[self.feature_columns]
        y = data['assessment_score']
        X_cognitive = data[self.cognitive_columns]
        
        # Train prediction model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        # Train clustering model
        self.clustering_scaler = StandardScaler()
        X_cognitive_scaled = self.clustering_scaler.fit_transform(X_cognitive)
        
        self.clustering_model = KMeans(n_clusters=4, random_state=42, n_init=10)
        cluster_labels = self.clustering_model.fit_predict(X_cognitive_scaled)
        
        # Calculate performance metrics
        train_score = self.model.score(X, y)
        feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        
        # Analyze clusters
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = cluster_labels
        cluster_summary = data_with_clusters.groupby('cluster')[
            self.cognitive_columns + ['assessment_score', 'engagement_time']
        ].mean()
        
        self.is_trained = True
        
        return {
            'model_r2_score': train_score,
            'feature_importance': feature_importance,
            'cluster_summary': cluster_summary.to_dict(),
            'total_students': len(data),
            'personas_identified': len(self.persona_mapping)
        }
    
    def predict_performance(self, comprehension: float, attention: float, 
                          focus: float, retention: float, engagement_time: float) -> Dict[str, Any]:
        """
        Predict student assessment score and learning persona.
        
        Args:
            comprehension (float): Comprehension score (0-100)
            attention (float): Attention score (0-100)
            focus (float): Focus score (0-100)
            retention (float): Retention score (0-100)
            engagement_time (float): Weekly engagement time in hours
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")
        
        # Prepare input data
        input_data = np.array([[comprehension, attention, focus, retention, engagement_time]])
        cognitive_data = np.array([[comprehension, attention, focus, retention]])
        
        # Predict assessment score
        predicted_score = self.model.predict(input_data)[0]
        
        # Predict learning persona
        cognitive_scaled = self.clustering_scaler.transform(cognitive_data)
        persona_id = self.clustering_model.predict(cognitive_scaled)[0]
        persona_name = self.persona_mapping.get(persona_id, f"Persona {persona_id}")
        
        # Calculate confidence
        distances = self.clustering_model.transform(cognitive_scaled)[0]
        confidence = 1 / (1 + distances[persona_id])
        
        # Generate risk assessment
        risk_level = self._assess_risk(predicted_score, comprehension, attention, focus, retention)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            predicted_score, persona_id, comprehension, attention, focus, retention
        )
        
        return {
            'predicted_score': round(predicted_score, 1),
            'persona_id': int(persona_id),
            'persona_name': persona_name,
            'confidence': round(confidence, 3),
            'risk_level': risk_level,
            'recommendations': recommendations,
            'cognitive_profile': {
                'comprehension': comprehension,
                'attention': attention,
                'focus': focus,
                'retention': retention
            }
        }
    
    def analyze_class_performance(self, students_data: List[Dict]) -> Dict[str, Any]:
        """
        Analyze performance patterns for a group of students.
        
        Args:
            students_data (List[Dict]): List of student data dictionaries
            
        Returns:
            Dict[str, Any]: Class analysis results
        """
        if not students_data:
            return {'error': 'No student data provided'}
        
        # Convert to DataFrame
        df = pd.DataFrame(students_data)
        
        # Calculate class statistics
        class_stats = {
            'total_students': len(df),
            'avg_assessment_score': df['assessment_score'].mean(),
            'score_std': df['assessment_score'].std(),
            'avg_engagement_time': df['engagement_time'].mean(),
            'performance_distribution': {
                'high_performers': len(df[df['assessment_score'] >= 85]),
                'average_performers': len(df[(df['assessment_score'] >= 65) & (df['assessment_score'] < 85)]),
                'needs_support': len(df[df['assessment_score'] < 65])
            }
        }
        
        # Identify correlations
        correlations = {}
        for skill in self.cognitive_columns:
            if skill in df.columns:
                correlations[skill] = df[skill].corr(df['assessment_score'])
        
        # Predict personas for all students
        persona_distribution = {}
        at_risk_students = []
        
        for _, student in df.iterrows():
            try:
                prediction = self.predict_performance(
                    student['comprehension'], student['attention'],
                    student['focus'], student['retention'], student['engagement_time']
                )
                
                persona = prediction['persona_name']
                persona_distribution[persona] = persona_distribution.get(persona, 0) + 1
                
                if prediction['risk_level'] == 'High':
                    at_risk_students.append({
                        'student_id': student.get('student_id', 'Unknown'),
                        'name': student.get('name', 'Unknown'),
                        'predicted_score': prediction['predicted_score'],
                        'risk_factors': prediction['recommendations'][:2]  # Top 2 recommendations
                    })
                    
            except Exception as e:
                continue
        
        return {
            'class_statistics': class_stats,
            'skill_correlations': correlations,
            'persona_distribution': persona_distribution,
            'at_risk_students': at_risk_students,
            'insights': self._generate_class_insights(class_stats, correlations, persona_distribution)
        }
    
    def _assess_risk(self, predicted_score: float, comprehension: float, 
                    attention: float, focus: float, retention: float) -> str:
        """Assess student risk level based on predicted performance and cognitive skills."""
        risk_factors = 0
        
        if predicted_score < 60:
            risk_factors += 2
        elif predicted_score < 70:
            risk_factors += 1
            
        if comprehension < 50:
            risk_factors += 1
        if attention < 50:
            risk_factors += 1
        if focus < 50:
            risk_factors += 1
        if retention < 50:
            risk_factors += 1
            
        if risk_factors >= 3:
            return "High"
        elif risk_factors >= 1:
            return "Medium"
        else:
            return "Low"
    
    def _generate_recommendations(self, predicted_score: float, persona_id: int,
                                comprehension: float, attention: float, 
                                focus: float, retention: float) -> List[str]:
        """Generate personalized recommendations based on student profile."""
        recommendations = []
        
        # Score-based recommendations
        if predicted_score < 60:
            recommendations.extend([
                "Provide additional foundational support and scaffolding",
                "Consider one-on-one tutoring sessions",
                "Break down complex tasks into smaller, manageable steps"
            ])
        elif predicted_score < 75:
            recommendations.extend([
                "Focus on consistency and regular practice",
                "Provide moderate challenges to promote growth",
                "Implement regular progress check-ins"
            ])
        else:
            recommendations.extend([
                "Offer advanced challenges and enrichment activities",
                "Consider peer tutoring or leadership opportunities",
                "Encourage independent research projects"
            ])
        
        # Skill-specific recommendations
        if attention < 60:
            recommendations.append("Implement attention-building exercises and mindfulness practices")
        if focus < 60:
            recommendations.append("Use shorter learning sessions with frequent breaks")
        if comprehension < 60:
            recommendations.append("Provide visual aids and multiple explanation methods")
        if retention < 60:
            recommendations.append("Implement spaced repetition and memory techniques")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _generate_class_insights(self, stats: Dict, correlations: Dict, 
                               personas: Dict) -> List[str]:
        """Generate insights for class-level analysis."""
        insights = []
        
        # Performance insights
        if stats['avg_assessment_score'] > 80:
            insights.append("Class shows strong overall performance with high achievement levels")
        elif stats['avg_assessment_score'] < 65:
            insights.append("Class may benefit from additional support and intervention strategies")
        
        # Correlation insights
        strongest_predictor = max(correlations.items(), key=lambda x: abs(x[1]))
        insights.append(f"{strongest_predictor[0].title()} shows the strongest correlation with performance ({strongest_predictor[1]:.2f})")
        
        # Persona insights
        dominant_persona = max(personas.items(), key=lambda x: x[1])
        insights.append(f"Most students fall into the '{dominant_persona[0]}' category ({dominant_persona[1]} students)")
        
        # Risk insights
        needs_support = stats['performance_distribution']['needs_support']
        if needs_support > 0:
            insights.append(f"{needs_support} students may need additional support to improve performance")
        
        return insights

# Utility functions for dashboard integration
def generate_synthetic_data(n_students: int = 100) -> pd.DataFrame:
    """Generate synthetic student data for testing and demonstration."""
    np.random.seed(42)
    
    # Generate student IDs and names
    student_ids = range(1, n_students + 1)
    names = [f"Student {i}" for i in student_ids]
    classes = np.random.choice(['9A', '9B', '10A', '10B', '11A', '11B'], n_students)
    
    # Generate correlated cognitive skills
    base_ability = np.random.normal(70, 15, n_students)
    base_ability = np.clip(base_ability, 30, 95)
    
    comprehension = base_ability + np.random.normal(0, 8, n_students)
    attention = base_ability + np.random.normal(0, 10, n_students)
    focus = 0.7 * attention + 0.3 * base_ability + np.random.normal(0, 6, n_students)
    retention = 0.6 * comprehension + 0.4 * base_ability + np.random.normal(0, 7, n_students)
    
    # Clip values to realistic ranges
    comprehension = np.clip(comprehension, 20, 100)
    attention = np.clip(attention, 15, 100)
    focus = np.clip(focus, 15, 100)
    retention = np.clip(retention, 20, 100)
    
    # Generate assessment scores
    assessment_score = (0.3 * comprehension + 0.25 * attention + 0.2 * focus + 0.25 * retention + 
                       np.random.normal(0, 8, n_students))
    assessment_score = np.clip(assessment_score, 30, 100)
    
    # Generate engagement time
    base_engagement = 15 + (assessment_score - 50) * 0.2
    engagement_time = base_engagement + np.random.normal(0, 5, n_students)
    engagement_time = np.clip(engagement_time, 5, 40)
    
    return pd.DataFrame({
        'student_id': student_ids,
        'name': names,
        'class': classes,
        'comprehension': np.round(comprehension, 1),
        'attention': np.round(attention, 1),
        'focus': np.round(focus, 1),
        'retention': np.round(retention, 1),
        'assessment_score': np.round(assessment_score, 1),
        'engagement_time': np.round(engagement_time, 1)
    })

def export_model_config() -> Dict[str, Any]:
    """Export model configuration for dashboard integration."""
    return {
        'feature_columns': ['comprehension', 'attention', 'focus', 'retention', 'engagement_time'],
        'cognitive_columns': ['comprehension', 'attention', 'focus', 'retention'],
        'persona_mapping': {
            0: "High Performers",
            1: "Strong Achievers", 
            2: "Developing Learners",
            3: "Support Needed"
        },
        'score_ranges': {
            'excellent': (85, 100),
            'good': (70, 84),
            'average': (55, 69),
            'needs_improvement': (0, 54)
        },
        'risk_thresholds': {
            'high_risk': 60,
            'medium_risk': 70,
            'low_risk': 85
        }
    }

# Example usage and testing
if __name__ == "__main__":
    # Create predictor instance
    predictor = StudentPerformancePredictor()
    
    # Generate sample data
    sample_data = generate_synthetic_data(200)
    
    # Train models
    training_results = predictor.train_models(sample_data)
    print("Training Results:", training_results)
    
    # Test prediction
    test_prediction = predictor.predict_performance(75, 80, 70, 85, 20)
    print("Test Prediction:", test_prediction)
    
    # Test class analysis
    class_analysis = predictor.analyze_class_performance(sample_data.to_dict('records'))
    print("Class Analysis:", class_analysis['insights'])
