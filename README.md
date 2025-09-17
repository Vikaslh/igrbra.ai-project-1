# Student Cognitive Skills & Performance Dashboard

A comprehensive full-stack application for analyzing student cognitive skills and academic performance using machine learning techniques.

## Features

### 🎯 Core Functionality
- **CSV Data Upload**: Upload student datasets with validation
- **Interactive Dashboard**: Overview cards, charts, and visualizations
- **Student Directory**: Searchable and sortable student table
- **ML Insights**: Correlation analysis, clustering, and predictions

### 📊 Visualizations
- **Bar Chart**: Average cognitive skills vs assessment scores
- **Scatter Plot**: Attention levels vs performance correlation
- **Radar Chart**: Individual student cognitive skill profiles
- **Progress Indicators**: Model performance and feature importance

### 🤖 Machine Learning
- **Correlation Analysis**: Identify strongest predictors of performance
- **Random Forest Model**: Predict assessment scores with 84%+ accuracy
- **K-Means Clustering**: Group students into 4 learning personas
- **Feature Importance**: Rank cognitive skills by predictive power

## Tech Stack

### Frontend
- **Next.js 15** with App Router
- **React 19** with TypeScript
- **Tailwind CSS 4** for styling
- **Recharts** for data visualization
- **shadcn/ui** components

### Backend & ML
- **Next.js API Routes** for data processing
- **Python Scripts** for ML analysis
- **scikit-learn** for machine learning
- **pandas** for data manipulation
- **matplotlib/seaborn** for visualizations

## Getting Started

### Prerequisites
- Node.js 18+ 
- Python 3.8+
- npm or yarn

### Installation

1. **Clone the repository**
   \`\`\`bash
   git clone <repository-url>
   cd student-dashboard
   \`\`\`

2. **Install dependencies**
   \`\`\`bash
   npm install
   \`\`\`

3. **Install Python dependencies**
   \`\`\`bash
   pip install pandas scikit-learn matplotlib seaborn numpy
   \`\`\`

4. **Run the development server**
   \`\`\`bash
   npm run dev
   \`\`\`

5. **Open your browser**
   Navigate to `http://localhost:3000`

## Usage

### 1. Upload Data
- Prepare a CSV file with required columns:
  - `student_id`, `name`, `class`
  - `comprehension`, `attention`, `focus`, `retention`
  - `assessment_score`, `engagement_time`
- Upload via the dashboard interface
- Data is validated automatically

### 2. Explore Dashboard
- **Overview Tab**: View summary cards and key metrics
- **Student Directory**: Search, filter, and sort student data
- **ML Insights**: Analyze correlations, clusters, and predictions

### 3. Run ML Analysis
Execute the Python analysis script:
\`\`\`bash
cd scripts
python run_analysis.py
\`\`\`

This generates:
- `processed_students.csv` - Dataset with cluster assignments
- `analysis_insights.json` - ML analysis results
- Visualization plots (PNG files)

## Data Requirements

### Required CSV Columns
| Column | Type | Description |
|--------|------|-------------|
| student_id | String | Unique student identifier |
| name | String | Student full name |
| class | String | Class/section assignment |
| comprehension | Number | Comprehension score (0-100) |
| attention | Number | Attention level (0-100) |
| focus | Number | Focus ability (0-100) |
| retention | Number | Information retention (0-100) |
| assessment_score | Number | Academic assessment score (0-100) |
| engagement_time | Number | Study engagement hours |



## ML Analysis Details

### Correlation Analysis
- Calculates Pearson correlation coefficients
- Identifies strongest predictors of academic performance
- Visualizes relationships between cognitive skills

### Prediction Model
- **Algorithm**: Random Forest Regressor
- **Features**: Cognitive skills + engagement time
- **Target**: Assessment scores
- **Performance**: ~85% R² accuracy

### Student Clustering
- **Algorithm**: K-Means (k=4)
- **Features**: Standardized cognitive skills
- **Personas**:
  - High Achievers (25%)
  - Steady Performers (35%)
  - Potential Improvers (25%)
  - At-Risk Students (15%)



## License

This project is licensed under the MIT License.
# igrbra.ai-project-1
