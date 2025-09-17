#!/usr/bin/env python3
"""
Quick script to run the student cognitive skills analysis
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analysis_student_cognitive_skills import main

if __name__ == "__main__":
    print("🎓 Student Cognitive Skills Analysis")
    print("Running comprehensive ML analysis...")
    
    try:
        df_processed, insights = main()
        print("\n✅ Analysis completed successfully!")
        print(f"📊 Processed {len(df_processed)} student records")
        print("📁 Check the following output files:")
        print("   - processed_students.csv")
        print("   - analysis_insights.json")
        print("   - correlation_heatmap.png")
        print("   - feature_importance.png")
        print("   - actual_vs_predicted.png")
        print("   - cluster_analysis.png")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        sys.exit(1)
