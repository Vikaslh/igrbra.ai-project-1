"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import type { StudentData } from "./student-dashboard"
import { Brain, Eye, Target, BookOpen, Clock, Trophy } from "lucide-react"
import { SkillsPerformanceChart } from "./charts/skills-performance-chart"
import { AttentionPerformanceScatter } from "./charts/attention-performance-scatter"
import { StudentRadarChart } from "./charts/student-radar-chart"

interface DashboardOverviewProps {
  data: StudentData[]
}

export function DashboardOverview({ data }: DashboardOverviewProps) {
  const calculateAverage = (field: keyof StudentData) => {
    const sum = data.reduce((acc, student) => acc + Number(student[field]), 0)
    return (sum / data.length).toFixed(1)
  }

  const overviewCards = [
    {
      title: "Comprehension",
      value: calculateAverage("comprehension"),
      icon: Brain,
      description: "Average comprehension score",
      color: "text-blue-600",
    },
    {
      title: "Attention",
      value: calculateAverage("attention"),
      icon: Eye,
      description: "Average attention level",
      color: "text-green-600",
    },
    {
      title: "Focus",
      value: calculateAverage("focus"),
      icon: Target,
      description: "Average focus score",
      color: "text-purple-600",
    },
    {
      title: "Retention",
      value: calculateAverage("retention"),
      icon: BookOpen,
      description: "Average retention rate",
      color: "text-orange-600",
    },
    {
      title: "Engagement Time",
      value: `${calculateAverage("engagement_time")}h`,
      icon: Clock,
      description: "Average engagement hours",
      color: "text-teal-600",
    },
    {
      title: "Assessment Score",
      value: calculateAverage("assessment_score"),
      icon: Trophy,
      description: "Average assessment score",
      color: "text-red-600",
    },
  ]

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {overviewCards.map((card) => {
          const Icon = card.icon
          return (
            <Card key={card.title}>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">{card.title}</CardTitle>
                <Icon className={`h-4 w-4 ${card.color}`} />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{card.value}</div>
                <p className="text-xs text-muted-foreground">{card.description}</p>
              </CardContent>
            </Card>
          )
        })}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Dataset Summary</CardTitle>
          <CardDescription>Overview of your uploaded student data</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <p className="text-muted-foreground">Total Students</p>
              <p className="text-2xl font-bold">{data.length}</p>
            </div>
            <div>
              <p className="text-muted-foreground">Classes</p>
              <p className="text-2xl font-bold">{new Set(data.map((s) => s.class)).size}</p>
            </div>
            <div>
              <p className="text-muted-foreground">Avg Score Range</p>
              <p className="text-2xl font-bold">
                {Math.min(...data.map((s) => s.assessment_score)).toFixed(0)}-
                {Math.max(...data.map((s) => s.assessment_score)).toFixed(0)}
              </p>
            </div>
            <div>
              <p className="text-muted-foreground">Data Quality</p>
              <p className="text-2xl font-bold text-green-600">✓ Valid</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <SkillsPerformanceChart data={data} />
        <AttentionPerformanceScatter data={data} />
      </div>

      <StudentRadarChart data={data} />
    </div>
  )
}
