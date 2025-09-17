import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_and_analyze_data(csv_path='uploaded_data.csv'):
    """
    Load student data and perform comprehensive analysis
    """
    print("🔍 Loading student cognitive skills data...")
    
    # For demo purposes, create sample data if no file exists
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} student records from {csv_path}")
    except FileNotFoundError:
        print("📝 Creating sample dataset for demonstration...")
        df = create_sample_data()
    
    return df

def create_sample_data(n_students=250):
    """
    Create sample student data for demonstration
    """
    np.random.seed(42)
    
    # Generate student IDs and names
    student_ids = [f"STU{str(i).zfill(4)}" for i in range(1, n_students + 1)]
    names = [f"Student_{i}" for i in range(1, n_students + 1)]
    classes = np.random.choice(['Class_A', 'Class_B', 'Class_C', 'Class_D', 'Class_E'], n_students)
    
    # Generate correlated cognitive skills (0-100 scale)
    base_ability = np.random.normal(70, 15, n_students)
    base_ability = np.clip(base_ability, 30, 95)
    
    comprehension = base_ability + np.random.normal(0, 8, n_students)
    attention = base_ability + np.random.normal(-5, 10, n_students)
    focus = attention + np.random.normal(0, 6, n_students)
    retention = comprehension + np.random.normal(-3, 7, n_students)
    
    # Clip values to realistic ranges
    comprehension = np.clip(comprehension, 20, 100)
    attention = np.clip(attention, 15, 100)
    focus = np.clip(focus, 15, 100)
    retention = np.clip(retention, 20, 100)
    
    # Generate assessment scores based on cognitive skills with some noise
    assessment_score = (
        0.3 * comprehension + 
        0.25 * attention + 
        0.2 * focus + 
        0.25 * retention + 
        np.random.normal(0, 8, n_students)
    )
    assessment_score = np.clip(assessment_score, 30, 100)
    
    # Generate engagement time (hours) - correlated with attention and focus
    engagement_time = (
        2 + (attention + focus) / 50 + np.random.normal(0, 1, n_students)
    )
    engagement_time = np.clip(engagement_time, 0.5, 8)
    
    df = pd.DataFrame({
        'student_id': student_ids,
        'name': names,
        'class': classes,
        'comprehension': comprehension.round(1),
        'attention': attention.round(1),
        'focus': focus.round(1),
        'retention': retention.round(1),
        'assessment_score': assessment_score.round(1),
        'engagement_time': engagement_time.round(1)
    })
    
    return df

def perform_correlation_analysis(df):
    """
    Analyze correlations between cognitive skills and assessment scores
    """
    print("\n📊 Performing correlation analysis...")
    
    # Select numeric columns for correlation
    numeric_cols = ['comprehension', 'attention', 'focus', 'retention', 'assessment_score', 'engagement_time']
    correlation_matrix = df[numeric_cols].corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f', cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Correlation Matrix: Cognitive Skills vs Assessment Performance', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find strongest correlations with assessment_score
    correlations_with_score = correlation_matrix['assessment_score'].drop('assessment_score').sort_values(ascending=False)
    
    print("🎯 Strongest correlations with assessment score:")
    for skill, corr in correlations_with_score.items():
        print(f"   {skill.capitalize()}: {corr:.3f}")
    
    return correlation_matrix, correlations_with_score

def train_prediction_model(df):
    """
    Train Random Forest model to predict assessment scores
    """
    print("\n🤖 Training Random Forest prediction model...")
    
    # Prepare features and target
    feature_cols = ['comprehension', 'attention', 'focus', 'retention', 'engagement_time']
    X = df[feature_cols]
    y = df['assessment_score']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"📈 Model Performance:")
    print(f"   R² Score: {r2:.3f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   Mean Absolute Error: {np.mean(np.abs(y_test - y_pred)):.2f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🎯 Feature Importance:")
    for _, row in feature_importance.iterrows():
        print(f"   {row['feature'].capitalize()}: {row['importance']:.3f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
    plt.title('Feature Importance in Predicting Assessment Scores', fontsize=14)
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='steelblue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Assessment Score')
    plt.ylabel('Predicted Assessment Score')
    plt.title(f'Actual vs Predicted Assessment Scores (R² = {r2:.3f})')
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return rf_model, feature_importance, {'r2': r2, 'rmse': rmse}

def perform_clustering_analysis(df):
    """
    Perform KMeans clustering to identify student learning personas
    """
    print("\n🎭 Performing KMeans clustering analysis...")
    
    # Prepare features for clustering
    feature_cols = ['comprehension', 'attention', 'focus', 'retention', 'engagement_time']
    X = df[feature_cols]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform KMeans clustering (4 clusters as specified)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to dataframe
    df['cluster'] = clusters
    
    # Analyze cluster characteristics
    cluster_summary = df.groupby('cluster')[feature_cols + ['assessment_score']].mean().round(2)
    cluster_counts = df['cluster'].value_counts().sort_index()
    
    print("📊 Cluster Summary (Average Values):")
    print(cluster_summary)
    print(f"\n👥 Cluster Sizes:")
    for cluster, count in cluster_counts.items():
        print(f"   Cluster {cluster}: {count} students")
    
    # Create cluster personas
    personas = {}
    for cluster in range(4):
        cluster_data = cluster_summary.loc[cluster]
        
        # Determine persona based on characteristics
        if cluster_data['assessment_score'] >= 80:
            if cluster_data['engagement_time'] >= 4:
                persona = "High Achievers"
                description = "Excellent performance with high engagement"
            else:
                persona = "Efficient Learners"
                description = "High performance with moderate engagement"
        elif cluster_data['assessment_score'] >= 65:
            if cluster_data['attention'] >= 70:
                persona = "Steady Performers"
                description = "Good attention and consistent performance"
            else:
                persona = "Potential Improvers"
                description = "Average performance with room for growth"
        else:
            if cluster_data['engagement_time'] >= 3:
                persona = "Struggling Engagers"
                description = "High effort but need additional support"
            else:
                persona = "At-Risk Students"
                description = "Low performance and engagement - need intervention"
        
        personas[cluster] = {
            'name': persona,
            'description': description,
            'avg_score': cluster_data['assessment_score'],
            'count': cluster_counts[cluster]
        }
    
    print(f"\n🎭 Learning Personas:")
    for cluster, persona in personas.items():
        print(f"   Cluster {cluster} - {persona['name']}: {persona['description']}")
        print(f"      Average Score: {persona['avg_score']:.1f}, Students: {persona['count']}")
    
    # Visualize clusters
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Comprehension vs Assessment Score
    axes[0,0].scatter(df['comprehension'], df['assessment_score'], c=df['cluster'], cmap='viridis', alpha=0.6)
    axes[0,0].set_xlabel('Comprehension')
    axes[0,0].set_ylabel('Assessment Score')
    axes[0,0].set_title('Clusters: Comprehension vs Assessment Score')
    
    # Plot 2: Attention vs Focus
    axes[0,1].scatter(df['attention'], df['focus'], c=df['cluster'], cmap='viridis', alpha=0.6)
    axes[0,1].set_xlabel('Attention')
    axes[0,1].set_ylabel('Focus')
    axes[0,1].set_title('Clusters: Attention vs Focus')
    
    # Plot 3: Engagement Time vs Assessment Score
    axes[1,0].scatter(df['engagement_time'], df['assessment_score'], c=df['cluster'], cmap='viridis', alpha=0.6)
    axes[1,0].set_xlabel('Engagement Time (hours)')
    axes[1,0].set_ylabel('Assessment Score')
    axes[1,0].set_title('Clusters: Engagement vs Assessment Score')
    
    # Plot 4: Cluster distribution
    cluster_counts.plot(kind='bar', ax=axes[1,1], color='skyblue')
    axes[1,1].set_xlabel('Cluster')
    axes[1,1].set_ylabel('Number of Students')
    axes[1,1].set_title('Student Distribution by Cluster')
    axes[1,1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig('cluster_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df, personas, cluster_summary

def generate_insights_report(df, correlations, model_metrics, personas):
    """
    Generate comprehensive insights report
    """
    print("\n📋 Generating Insights Report...")
    
    insights = {
        'dataset_summary': {
            'total_students': len(df),
            'classes': df['class'].nunique(),
            'avg_assessment_score': df['assessment_score'].mean().round(2),
            'score_range': [df['assessment_score'].min(), df['assessment_score'].max()]
        },
        'key_correlations': {
            'strongest_predictor': correlations.index[0],
            'strongest_correlation': correlations.iloc[0].round(3),
            'all_correlations': correlations.round(3).to_dict()
        },
        'model_performance': model_metrics,
        'learning_personas': personas,
        'recommendations': [
            f"Focus on {correlations.index[0]} as it shows the strongest correlation ({correlations.iloc[0]:.3f}) with assessment scores",
            f"The prediction model achieves {model_metrics['r2']:.1%} accuracy in predicting student performance",
            f"Identified {len(personas)} distinct learning personas for targeted interventions",
            "Consider personalized learning paths based on cluster characteristics"
        ]
    }
    
    # Save insights as JSON
    with open('analysis_insights.json', 'w') as f:
        json.dump(insights, f, indent=2)
    
    print("✅ Analysis complete! Key findings:")
    for i, rec in enumerate(insights['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    return insights

def main():
    """
    Main analysis pipeline
    """
    print("🚀 Starting Student Cognitive Skills Analysis")
    print("=" * 50)
    
    # Load data
    df = load_and_analyze_data()
    
    # Perform correlation analysis
    correlation_matrix, correlations_with_score = perform_correlation_analysis(df)
    
    # Train prediction model
    model, feature_importance, model_metrics = train_prediction_model(df)
    
    # Perform clustering
    df_with_clusters, personas, cluster_summary = perform_clustering_analysis(df)
    
    # Generate insights
    insights = generate_insights_report(df_with_clusters, correlations_with_score, model_metrics, personas)
    
    # Save processed dataset
    df_with_clusters.to_csv('processed_students.csv', index=False)
    print(f"\n💾 Saved processed dataset with clusters to 'processed_students.csv'")
    
    print("\n🎉 Analysis pipeline completed successfully!")
    return df_with_clusters, insights

if __name__ == "__main__":
    df_processed, insights = main()
