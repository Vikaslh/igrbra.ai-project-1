"""
Comprehensive Machine Learning Functions for Student Performance Analysis
This module contains all the functions needed for training, testing, and deploying ML models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

class StudentPerformanceMLPipeline:
    """
    Comprehensive ML pipeline for student performance analysis and prediction.
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scalers = {}
        self.encoders = {}
        self.selected_features = []
        self.results = {}
        
    def generate_synthetic_dataset(self, n_samples=5000, random_state=42):
        """
        Generate comprehensive synthetic student dataset.
        
        Parameters:
        n_samples (int): Number of samples to generate
        random_state (int): Random seed for reproducibility
        
        Returns:
        pd.DataFrame: Generated dataset
        """
        np.random.seed(random_state)
        
        # Generate base features with realistic distributions
        data = {
            'student_id': range(1, n_samples + 1),
            'age': np.random.normal(16, 2, n_samples).clip(13, 20),
            'grade_level': np.random.choice([9, 10, 11, 12], n_samples, p=[0.25, 0.25, 0.25, 0.25]),
            'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.45, 0.45, 0.1]),
            'socioeconomic_status': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.3, 0.5, 0.2]),
            
            # Cognitive skills (0-100 scale)
            'comprehension': np.random.beta(2, 2, n_samples) * 100,
            'attention_span': np.random.beta(2, 2, n_samples) * 100,
            'memory_retention': np.random.beta(2, 2, n_samples) * 100,
            'problem_solving': np.random.beta(2, 2, n_samples) * 100,
            'critical_thinking': np.random.beta(2, 2, n_samples) * 100,
            
            # Study habits and environment
            'study_hours_per_week': np.random.gamma(2, 5, n_samples).clip(0, 50),
            'sleep_hours_per_night': np.random.normal(7.5, 1.5, n_samples).clip(4, 12),
            'extracurricular_activities': np.random.poisson(2, n_samples).clip(0, 8),
            'family_support': np.random.beta(3, 2, n_samples) * 10,
            'teacher_rating': np.random.beta(3, 2, n_samples) * 10,
            
            # Technology and resources
            'computer_access': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
            'internet_quality': np.random.choice(['Poor', 'Fair', 'Good', 'Excellent'], n_samples, p=[0.1, 0.2, 0.4, 0.3]),
            'learning_resources': np.random.beta(2, 2, n_samples) * 10,
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic assessment scores with multiple influences
        assessment_base = (
            0.3 * df['comprehension'] +
            0.25 * df['problem_solving'] +
            0.2 * df['critical_thinking'] +
            0.15 * df['attention_span'] +
            0.1 * df['memory_retention'] +
            0.05 * df['study_hours_per_week'] +
            0.03 * df['family_support'] * 10 +
            0.02 * df['teacher_rating'] * 10
        )
        
        # Add demographic effects
        ses_effect = df['socioeconomic_status'].map({'Low': -5, 'Medium': 0, 'High': 5})
        grade_effect = (df['grade_level'] - 9) * 2
        
        # Add realistic noise
        noise = np.random.normal(0, 8, n_samples)
        df['assessment_score'] = (assessment_base + ses_effect + grade_effect + noise).clip(0, 100)
        
        # Create engagement metrics
        df['class_participation'] = (df['attention_span'] * 0.6 + df['family_support'] * 4 + np.random.normal(0, 10, n_samples)).clip(0, 100)
        df['homework_completion'] = (df['study_hours_per_week'] * 1.5 + df['family_support'] * 3 + np.random.normal(0, 8, n_samples)).clip(0, 100)
        
        return df
    
    def preprocess_data(self, df):
        """
        Comprehensive data preprocessing pipeline.
        
        Parameters:
        df (pd.DataFrame): Input dataframe
        
        Returns:
        pd.DataFrame: Preprocessed dataframe
        """
        df_processed = df.copy()
        
        # Handle categorical variables
        ordinal_mappings = {
            'socioeconomic_status': {'Low': 0, 'Medium': 1, 'High': 2},
            'internet_quality': {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
        }
        
        for col, mapping in ordinal_mappings.items():
            df_processed[col] = df_processed[col].map(mapping)
            self.encoders[col] = mapping
        
        # One-hot encoding for nominal variables
        nominal_cols = ['gender']
        for col in nominal_cols:
            dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
            df_processed = pd.concat([df_processed, dummies], axis=1)
            df_processed.drop(col, axis=1, inplace=True)
            self.encoders[col] = list(dummies.columns)
        
        # Outlier treatment using IQR method
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols.remove('student_id')
        
        for col in numerical_cols:
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
        
        # Feature scaling
        feature_cols = [col for col in df_processed.columns if col not in ['student_id', 'assessment_score']]
        scaler = StandardScaler()
        df_processed[feature_cols] = scaler.fit_transform(df_processed[feature_cols])
        self.scalers['features'] = scaler
        
        return df_processed
    
    def engineer_features(self, df_processed):
        """
        Create engineered features based on domain knowledge.
        
        Parameters:
        df_processed (pd.DataFrame): Preprocessed dataframe
        
        Returns:
        pd.DataFrame: Dataframe with engineered features
        """
        df_engineered = df_processed.copy()
        
        # Cognitive composite score
        cognitive_cols = ['comprehension', 'attention_span', 'memory_retention', 'problem_solving', 'critical_thinking']
        df_engineered['cognitive_composite'] = df_engineered[cognitive_cols].mean(axis=1)
        
        # Study efficiency
        df_engineered['study_efficiency'] = df_engineered['assessment_score'] / (df_engineered['study_hours_per_week'] + 1)
        
        # Work-life balance
        df_engineered['work_life_balance'] = (df_engineered['sleep_hours_per_night'] * 0.4 + 
                                             (10 - df_engineered['extracurricular_activities']) * 0.3 + 
                                             (50 - df_engineered['study_hours_per_week']) * 0.3)
        
        # Support system strength
        df_engineered['support_system'] = (df_engineered['family_support'] * 0.6 + 
                                          df_engineered['teacher_rating'] * 0.4)
        
        # Technology advantage
        df_engineered['tech_advantage'] = (df_engineered['computer_access'] * 0.5 + 
                                          df_engineered['internet_quality'] * 0.3 + 
                                          df_engineered['learning_resources'] * 0.2)
        
        # Interaction features
        df_engineered['comprehension_x_attention'] = df_engineered['comprehension'] * df_engineered['attention_span']
        df_engineered['study_hours_x_family_support'] = df_engineered['study_hours_per_week'] * df_engineered['family_support']
        
        # Age-grade alignment
        expected_age = df_engineered['grade_level'] + 5
        df_engineered['age_grade_alignment'] = 1 - abs(df_engineered['age'] - expected_age) / 4
        
        return df_engineered
    
    def select_features(self, X, y, k=15):
        """
        Select top k features using statistical methods.
        
        Parameters:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        k (int): Number of features to select
        
        Returns:
        list: Selected feature names
        """
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        self.selected_features = selected_features
        return selected_features
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """
        Train multiple ML models and compare performance.
        
        Parameters:
        X_train, y_train: Training data
        X_test, y_test: Test data
        
        Returns:
        dict: Model performance results
        """
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            self.models[name] = model
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred_test': y_pred_test
            }
        
        self.results = results
        
        # Identify best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
        self.best_model = self.models[best_model_name]
        
        return results, best_model_name
    
    def optimize_hyperparameters(self, X_train, y_train, model_name):
        """
        Optimize hyperparameters for the best model.
        
        Parameters:
        X_train, y_train: Training data
        model_name (str): Name of the model to optimize
        
        Returns:
        sklearn model: Optimized model
        """
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestRegressor(random_state=42)
            
        elif model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
            base_model = GradientBoostingRegressor(random_state=42)
            
        else:
            # Default to Random Forest
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
            base_model = RandomForestRegressor(random_state=42)
        
        # Perform grid search
        grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        self.best_model = grid_search.best_estimator_
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    def perform_clustering_analysis(self, df, features_for_clustering):
        """
        Perform student clustering analysis.
        
        Parameters:
        df (pd.DataFrame): Original dataframe
        features_for_clustering (list): Features to use for clustering
        
        Returns:
        tuple: (optimal_k, cluster_labels, cluster_characteristics)
        """
        clustering_data = df[features_for_clustering]
        
        # Determine optimal number of clusters
        silhouette_scores = []
        k_range = range(2, 11)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(clustering_data)
            silhouette_scores.append(silhouette_score(clustering_data, cluster_labels))
        
        # Choose optimal k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        # Final clustering
        final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = final_kmeans.fit_predict(clustering_data)
        
        # Analyze cluster characteristics
        df_clustered = df.copy()
        df_clustered['cluster'] = cluster_labels
        cluster_characteristics = df_clustered.groupby('cluster')[features_for_clustering].mean()
        
        return optimal_k, cluster_labels, cluster_characteristics
    
    def generate_visualizations(self, df, results, best_model_name, X_test, y_test):
        """
        Generate comprehensive visualizations for the analysis.
        
        Parameters:
        df: Original dataframe
        results: Model results dictionary
        best_model_name: Name of best performing model
        X_test, y_test: Test data
        """
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Model Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # R² comparison
        model_names = list(results.keys())
        test_r2_scores = [results[name]['test_r2'] for name in model_names]
        
        axes[0, 0].bar(model_names, test_r2_scores, color='skyblue', alpha=0.8)
        axes[0, 0].set_title('Model R² Score Comparison')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE comparison
        test_rmse_scores = [results[name]['test_rmse'] for name in model_names]
        axes[0, 1].bar(model_names, test_rmse_scores, color='lightcoral', alpha=0.8)
        axes[0, 1].set_title('Model RMSE Comparison (Lower is Better)')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Actual vs Predicted for best model
        best_predictions = results[best_model_name]['y_pred_test']
        axes[1, 0].scatter(y_test, best_predictions, alpha=0.6, color='blue')
        axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1, 0].set_title(f'Actual vs Predicted - {best_model_name}')
        axes[1, 0].set_xlabel('Actual Assessment Score')
        axes[1, 0].set_ylabel('Predicted Assessment Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_test - best_predictions
        axes[1, 1].scatter(best_predictions, residuals, alpha=0.6, color='green')
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_title(f'Residuals Plot - {best_model_name}')
        axes[1, 1].set_xlabel('Predicted Assessment Score')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 2. Feature Importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.selected_features,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(data=feature_importance.head(15), x='importance', y='feature', palette='viridis')
            plt.title(f'Top 15 Feature Importances - {best_model_name}', fontsize=14, fontweight='bold')
            plt.xlabel('Feature Importance')
            plt.ylabel('Features')
            plt.tight_layout()
            plt.show()
    
    def save_model_artifacts(self, filepath_prefix='student_performance'):
        """
        Save all model artifacts for deployment.
        
        Parameters:
        filepath_prefix (str): Prefix for saved files
        """
        # Save the best model
        joblib.dump(self.best_model, f'{filepath_prefix}_model.pkl')
        
        # Save preprocessing components
        joblib.dump(self.scalers, f'{filepath_prefix}_scalers.pkl')
        joblib.dump(self.encoders, f'{filepath_prefix}_encoders.pkl')
        
        # Save selected features
        with open(f'{filepath_prefix}_features.json', 'w') as f:
            json.dump(self.selected_features, f)
        
        # Save results
        serializable_results = {}
        for model_name, metrics in self.results.items():
            serializable_results[model_name] = {
                k: float(v) if isinstance(v, (np.float64, np.float32)) else v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in metrics.items()
            }
        
        with open(f'{filepath_prefix}_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"✅ Model artifacts saved with prefix '{filepath_prefix}'")
    
    def predict_student_performance(self, student_data):
        """
        Predict performance for new student data.
        
        Parameters:
        student_data (dict or pd.DataFrame): Student features
        
        Returns:
        float: Predicted assessment score
        """
        if isinstance(student_data, dict):
            student_data = pd.DataFrame([student_data])
        
        # Apply same preprocessing
        # Note: In production, you'd need to handle this more carefully
        processed_data = student_data[self.selected_features]
        
        # Make prediction
        prediction = self.best_model.predict(processed_data)
        
        return prediction[0] if len(prediction) == 1 else prediction

# Utility functions for analysis
def calculate_prediction_intervals(model, X, n_bootstrap=100, confidence=0.95):
    """
    Calculate prediction intervals using bootstrap sampling.
    
    Parameters:
    model: Trained model
    X: Feature matrix
    n_bootstrap (int): Number of bootstrap samples
    confidence (float): Confidence level
    
    Returns:
    tuple: (mean_predictions, lower_bounds, upper_bounds)
    """
    predictions = []
    n_samples = len(X)
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_bootstrap = X.iloc[indices]
        
        # Make predictions
        pred = model.predict(X_bootstrap)
        predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Calculate intervals
    alpha = 1 - confidence
    lower = np.percentile(predictions, (alpha/2) * 100, axis=0)
    upper = np.percentile(predictions, (1 - alpha/2) * 100, axis=0)
    mean_pred = np.mean(predictions, axis=0)
    
    return mean_pred, lower, upper

def generate_learning_curves(model, X, y, cv=5):
    """
    Generate learning curves for model evaluation.
    
    Parameters:
    model: ML model
    X: Feature matrix
    y: Target variable
    cv (int): Cross-validation folds
    
    Returns:
    tuple: (train_sizes, train_scores, validation_scores)
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), scoring='r2'
    )
    
    return train_sizes, train_scores, val_scores

def create_comprehensive_report(pipeline, df, results, best_model_name):
    """
    Generate a comprehensive analysis report.
    
    Parameters:
    pipeline: ML pipeline object
    df: Original dataframe
    results: Model results
    best_model_name: Name of best model
    
    Returns:
    dict: Comprehensive report
    """
    report = {
        'dataset_overview': {
            'total_students': len(df),
            'features_analyzed': len(pipeline.selected_features),
            'avg_assessment_score': float(df['assessment_score'].mean()),
            'score_std': float(df['assessment_score'].std()),
            'score_range': [float(df['assessment_score'].min()), float(df['assessment_score'].max())]
        },
        'best_model': {
            'algorithm': best_model_name,
            'test_r2': float(results[best_model_name]['test_r2']),
            'test_rmse': float(results[best_model_name]['test_rmse']),
            'cv_score': float(results[best_model_name]['cv_mean'])
        },
        'feature_importance': None,
        'recommendations': []
    }
    
    # Add feature importance if available
    if hasattr(pipeline.best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': pipeline.selected_features,
            'importance': pipeline.best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        report['feature_importance'] = feature_importance.head(10).to_dict('records')
    
    # Generate recommendations based on analysis
    key_correlations = {
        'Study Hours': df['study_hours_per_week'].corr(df['assessment_score']),
        'Family Support': df['family_support'].corr(df['assessment_score']),
        'Sleep Hours': df['sleep_hours_per_night'].corr(df['assessment_score']),
        'Comprehension': df['comprehension'].corr(df['assessment_score'])
    }
    
    recommendations = []
    
    if key_correlations['Family Support'] > 0.3:
        recommendations.append("Implement family engagement programs to boost student support systems")
    
    if key_correlations['Sleep Hours'] > 0.2:
        recommendations.append("Educate students about the importance of adequate sleep for academic performance")
    
    if key_correlations['Study Hours'] > 0.2:
        recommendations.append("Provide study skills training and time management workshops")
    
    # Check for at-risk students
    at_risk_percentage = len(df[df['assessment_score'] < 40]) / len(df) * 100
    if at_risk_percentage > 10:
        recommendations.append(f"Develop targeted intervention programs for {at_risk_percentage:.1f}% of students scoring below 40")
    
    report['recommendations'] = recommendations
    report['key_correlations'] = key_correlations
    
    return report

# Example usage and testing functions
def run_complete_analysis(n_samples=5000):
    """
    Run the complete ML analysis pipeline.
    
    Parameters:
    n_samples (int): Number of samples to generate
    
    Returns:
    tuple: (pipeline, report)
    """
    print("🚀 Starting Comprehensive Student Performance ML Analysis")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = StudentPerformanceMLPipeline()
    
    # Generate dataset
    print("📊 Generating synthetic dataset...")
    df = pipeline.generate_synthetic_dataset(n_samples)
    print(f"   Generated {len(df)} student records")
    
    # Preprocess data
    print("🔧 Preprocessing data...")
    df_processed = pipeline.preprocess_data(df)
    
    # Engineer features
    print("⚙️ Engineering features...")
    df_engineered = pipeline.engineer_features(df_processed)
    
    # Feature selection
    print("🎯 Selecting optimal features...")
    feature_cols = [col for col in df_engineered.columns if col not in ['student_id', 'assessment_score']]
    X = df_engineered[feature_cols]
    y = df_engineered['assessment_score']
    selected_features = pipeline.select_features(X, y, k=15)
    print(f"   Selected {len(selected_features)} features")
    
    # Split data
    X_selected = df_engineered[selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    
    # Train models
    print("🤖 Training multiple ML models...")
    results, best_model_name = pipeline.train_models(X_train, y_train, X_test, y_test)
    print(f"   Best model: {best_model_name}")
    
    # Optimize hyperparameters
    print("🔍 Optimizing hyperparameters...")
    optimized_model, best_params, best_cv_score = pipeline.optimize_hyperparameters(X_train, y_train, best_model_name)
    
    # Generate visualizations
    print("📈 Generating visualizations...")
    pipeline.generate_visualizations(df, results, best_model_name, X_test, y_test)
    
    # Clustering analysis
    print("👥 Performing clustering analysis...")
    clustering_features = ['comprehension', 'attention_span', 'memory_retention', 'problem_solving', 
                          'critical_thinking', 'study_hours_per_week', 'family_support', 'assessment_score']
    optimal_k, cluster_labels, cluster_characteristics = pipeline.perform_clustering_analysis(df, clustering_features)
    print(f"   Identified {optimal_k} student personas")
    
    # Generate comprehensive report
    print("📋 Generating comprehensive report...")
    report = create_comprehensive_report(pipeline, df, results, best_model_name)
    
    # Save artifacts
    print("💾 Saving model artifacts...")
    pipeline.save_model_artifacts()
    
    print("\n✅ Analysis Complete!")
    print(f"🏆 Best Model: {best_model_name}")
    print(f"📊 Test R²: {results[best_model_name]['test_r2']:.4f}")
    print(f"📉 Test RMSE: {results[best_model_name]['test_rmse']:.2f}")
    
    return pipeline, report

if __name__ == "__main__":
    # Run the complete analysis
    pipeline, report = run_complete_analysis(n_samples=5000)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 ANALYSIS SUMMARY")
    print("="*60)
    print(f"Dataset: {report['dataset_overview']['total_students']} students")
    print(f"Best Model: {report['best_model']['algorithm']}")
    print(f"Performance: R² = {report['best_model']['test_r2']:.4f}")
    print(f"Top Features: {len(report['feature_importance']) if report['feature_importance'] else 0}")
    print(f"Recommendations: {len(report['recommendations'])}")
    print("="*60)
