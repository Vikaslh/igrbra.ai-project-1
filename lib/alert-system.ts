import type { StudentData } from "@/components/student-dashboard"

export interface Alert {
  id: string
  studentId: string
  studentName: string
  type: "critical" | "warning" | "info" | "success"
  category: "performance" | "engagement" | "skill" | "improvement"
  title: string
  message: string
  recommendations: string[]
  priority: number
  timestamp: string
  acknowledged: boolean
}

export interface PerformanceMetric {
  studentId: string
  date: string
  assessmentScore: number
  comprehension: number
  attention: number
  focus: number
  retention: number
  engagementTime: number
  trend: "improving" | "declining" | "stable"
  changePercent: number
}

export class AlertSystem {
  static generateAlerts(data: StudentData[]): Alert[] {
    const alerts: Alert[] = []
    const avgScore = data.reduce((sum, s) => sum + s.assessment_score, 0) / data.length
    const avgEngagement = data.reduce((sum, s) => sum + s.engagement_time, 0) / data.length

    data.forEach((student) => {
      const alertId = `${student.student_id}_${Date.now()}`

      // Critical Performance Alert
      if (student.assessment_score < 50) {
        alerts.push({
          id: `${alertId}_critical`,
          studentId: student.student_id,
          studentName: student.name,
          type: "critical",
          category: "performance",
          title: "Critical Performance Alert",
          message: `${student.name} has a very low assessment score of ${student.assessment_score}`,
          recommendations: [
            "Schedule immediate one-on-one meeting",
            "Implement personalized learning plan",
            "Consider additional tutoring support",
            "Review learning materials and methods",
          ],
          priority: 1,
          timestamp: new Date().toISOString(),
          acknowledged: false,
        })
      }

      // Low Engagement Warning
      if (student.engagement_time < 3) {
        alerts.push({
          id: `${alertId}_engagement`,
          studentId: student.student_id,
          studentName: student.name,
          type: "warning",
          category: "engagement",
          title: "Low Engagement Warning",
          message: `${student.name} shows low engagement with only ${student.engagement_time} hours per week`,
          recommendations: [
            "Explore motivational strategies",
            "Introduce gamification elements",
            "Check for external factors affecting engagement",
            "Consider alternative learning formats",
          ],
          priority: 2,
          timestamp: new Date().toISOString(),
          acknowledged: false,
        })
      }

      // Skill Imbalance Alert
      const skills = [student.comprehension, student.attention, student.focus, student.retention]
      const skillRange = Math.max(...skills) - Math.min(...skills)
      if (skillRange > 30) {
        const weakestSkill = ["comprehension", "attention", "focus", "retention"][skills.indexOf(Math.min(...skills))]
        alerts.push({
          id: `${alertId}_skill`,
          studentId: student.student_id,
          studentName: student.name,
          type: "warning",
          category: "skill",
          title: "Skill Imbalance Detected",
          message: `${student.name} shows significant variation in cognitive skills (${skillRange} point range)`,
          recommendations: [
            `Focus on improving ${weakestSkill} through targeted exercises`,
            "Implement balanced skill development program",
            "Monitor progress in weaker areas more closely",
            "Consider specialized training for weak skills",
          ],
          priority: 3,
          timestamp: new Date().toISOString(),
          acknowledged: false,
        })
      }

      // High Performance Recognition
      if (student.assessment_score > avgScore + 15) {
        alerts.push({
          id: `${alertId}_success`,
          studentId: student.student_id,
          studentName: student.name,
          type: "success",
          category: "improvement",
          title: "Outstanding Performance",
          message: `${student.name} is performing exceptionally well with a score of ${student.assessment_score}`,
          recommendations: [
            "Provide advanced challenges to maintain engagement",
            "Consider peer mentoring opportunities",
            "Explore leadership roles in group activities",
            "Maintain current learning strategies",
          ],
          priority: 4,
          timestamp: new Date().toISOString(),
          acknowledged: false,
        })
      }

      // Attention Deficit Alert
      if (student.attention < 40) {
        alerts.push({
          id: `${alertId}_attention`,
          studentId: student.student_id,
          studentName: student.name,
          type: "warning",
          category: "skill",
          title: "Attention Concerns",
          message: `${student.name} shows low attention levels (${student.attention}/100)`,
          recommendations: [
            "Implement attention-building exercises",
            "Reduce distractions in learning environment",
            "Use shorter, focused learning sessions",
            "Consider attention training programs",
          ],
          priority: 2,
          timestamp: new Date().toISOString(),
          acknowledged: false,
        })
      }
    })

    return alerts.sort((a, b) => a.priority - b.priority)
  }

  static generatePerformanceTracking(data: StudentData[]): PerformanceMetric[] {
    // Simulate historical performance data
    return data.map((student) => {
      const baseScore = student.assessment_score
      const previousScore = Math.max(0, Math.min(100, baseScore + (Math.random() - 0.5) * 20))
      const changePercent = ((baseScore - previousScore) / previousScore) * 100

      let trend: "improving" | "declining" | "stable" = "stable"
      if (changePercent > 5) trend = "improving"
      else if (changePercent < -5) trend = "declining"

      return {
        studentId: student.student_id,
        date: new Date().toISOString(),
        assessmentScore: baseScore,
        comprehension: student.comprehension,
        attention: student.attention,
        focus: student.focus,
        retention: student.retention,
        engagementTime: student.engagement_time,
        trend,
        changePercent: Math.round(changePercent * 10) / 10,
      }
    })
  }

  static getAlertStats(alerts: Alert[]) {
    return {
      total: alerts.length,
      critical: alerts.filter((a) => a.type === "critical").length,
      warning: alerts.filter((a) => a.type === "warning").length,
      info: alerts.filter((a) => a.type === "info").length,
      success: alerts.filter((a) => a.type === "success").length,
      unacknowledged: alerts.filter((a) => !a.acknowledged).length,
      byCategory: {
        performance: alerts.filter((a) => a.category === "performance").length,
        engagement: alerts.filter((a) => a.category === "engagement").length,
        skill: alerts.filter((a) => a.category === "skill").length,
        improvement: alerts.filter((a) => a.category === "improvement").length,
      },
    }
  }
}
