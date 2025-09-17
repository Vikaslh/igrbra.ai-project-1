"use client"

import { Bar, BarChart, CartesianGrid, XAxis, YAxis, ResponsiveContainer } from "recharts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import type { StudentData } from "../student-dashboard"

interface SkillsPerformanceChartProps {
  data: StudentData[]
}

export function SkillsPerformanceChart({ data }: SkillsPerformanceChartProps) {
  // Calculate average skills and their correlation with assessment scores
  const skillsData = [
    {
      skill: "Comprehension",
      avgSkill: Number((data.reduce((sum, s) => sum + s.comprehension, 0) / data.length).toFixed(1)),
      avgScore: Number((data.reduce((sum, s) => sum + s.assessment_score, 0) / data.length).toFixed(1)),
    },
    {
      skill: "Attention",
      avgSkill: Number((data.reduce((sum, s) => sum + s.attention, 0) / data.length).toFixed(1)),
      avgScore: Number((data.reduce((sum, s) => sum + s.assessment_score, 0) / data.length).toFixed(1)),
    },
    {
      skill: "Focus",
      avgSkill: Number((data.reduce((sum, s) => sum + s.focus, 0) / data.length).toFixed(1)),
      avgScore: Number((data.reduce((sum, s) => sum + s.assessment_score, 0) / data.length).toFixed(1)),
    },
    {
      skill: "Retention",
      avgSkill: Number((data.reduce((sum, s) => sum + s.retention, 0) / data.length).toFixed(1)),
      avgScore: Number((data.reduce((sum, s) => sum + s.assessment_score, 0) / data.length).toFixed(1)),
    },
  ]

  const chartConfig = {
    avgSkill: {
      label: "Average Skill Level",
      color: "hsl(var(--chart-1))",
    },
    avgScore: {
      label: "Assessment Score",
      color: "hsl(var(--chart-2))",
    },
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Skills vs Performance Analysis</CardTitle>
        <CardDescription>Average cognitive skill levels compared to assessment performance</CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig}>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={skillsData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="skill" tick={{ fontSize: 12 }} angle={-45} textAnchor="end" height={80} />
              <YAxis tick={{ fontSize: 12 }} />
              <ChartTooltip content={<ChartTooltipContent />} />
              <Bar dataKey="avgSkill" fill="var(--color-avgSkill)" radius={4} />
            </BarChart>
          </ResponsiveContainer>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
