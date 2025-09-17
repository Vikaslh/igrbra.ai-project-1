import type { StudentData } from "@/components/student-dashboard"

export interface ClassComparison {
  className: string
  avgScore: number
  studentCount: number
  topSkill: string
  weakestSkill: string
  performanceRating: "excellent" | "good" | "average" | "needs_improvement"
}

export class AdvancedAnalytics {
  static generateClassComparisons(data: StudentData[]): ClassComparison[] {
    const classes = Array.from(new Set(data.map((s) => s.class)))

    return classes.map((className) => {
      const classStudents = data.filter((s) => s.class === className)
      const avgScore = classStudents.reduce((sum, s) => sum + s.assessment_score, 0) / classStudents.length

      // Find strongest and weakest skills
      const skills = ["comprehension", "attention", "focus", "retention"] as const
      const skillAverages = skills.map((skill) => ({
        skill,
        avg: classStudents.reduce((sum, s) => sum + s[skill], 0) / classStudents.length,
      }))

      skillAverages.sort((a, b) => b.avg - a.avg)

      let performanceRating: "excellent" | "good" | "average" | "needs_improvement"
      if (avgScore >= 85) performanceRating = "excellent"
      else if (avgScore >= 75) performanceRating = "good"
      else if (avgScore >= 65) performanceRating = "average"
      else performanceRating = "needs_improvement"

      return {
        className,
        avgScore: Math.round(avgScore * 10) / 10,
        studentCount: classStudents.length,
        topSkill: skillAverages[0].skill,
        weakestSkill: skillAverages[skillAverages.length - 1].skill,
        performanceRating,
      }
    })
  }

  static calculateDetailedCorrelations(data: StudentData[]) {
    const skills = ["comprehension", "attention", "focus", "retention", "engagement_time"] as const
    const correlations: { [key: string]: number } = {}

    skills.forEach((skill) => {
      const x = data.map((s) => s[skill])
      const y = data.map((s) => s.assessment_score)
      correlations[skill] = this.pearsonCorrelation(x, y)
    })

    return correlations
  }

  private static pearsonCorrelation(x: number[], y: number[]): number {
    const n = x.length
    const sumX = x.reduce((a, b) => a + b, 0)
    const sumY = y.reduce((a, b) => a + b, 0)
    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0)
    const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0)
    const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0)

    const numerator = n * sumXY - sumX * sumY
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY))

    return denominator === 0 ? 0 : numerator / denominator
  }
}
