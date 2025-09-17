"use client"

import { useState } from "react"
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from "recharts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { ChartContainer, ChartTooltip } from "@/components/ui/chart"
import type { StudentData } from "../student-dashboard"

interface StudentRadarChartProps {
  data: StudentData[]
}

export function StudentRadarChart({ data }: StudentRadarChartProps) {
  const [selectedStudent, setSelectedStudent] = useState<string>(data[0]?.student_id || "")

  const student = data.find((s) => s.student_id === selectedStudent) || data[0]

  const radarData = [
    {
      skill: "Comprehension",
      value: student?.comprehension || 0,
      fullMark: 100,
    },
    {
      skill: "Attention",
      value: student?.attention || 0,
      fullMark: 100,
    },
    {
      skill: "Focus",
      value: student?.focus || 0,
      fullMark: 100,
    },
    {
      skill: "Retention",
      value: student?.retention || 0,
      fullMark: 100,
    },
    {
      skill: "Engagement",
      value: (student?.engagement_time || 0) * 12.5, // Scale to 0-100
      fullMark: 100,
    },
  ]

  const chartConfig = {
    value: {
      label: "Skill Level",
      color: "hsl(var(--chart-4))",
    },
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Individual Student Profile</CardTitle>
        <CardDescription>Radar chart showing cognitive skills profile for selected student</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="mb-4">
          <Select value={selectedStudent} onValueChange={setSelectedStudent}>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select a student" />
            </SelectTrigger>
            <SelectContent>
              {data.map((student) => (
                <SelectItem key={student.student_id} value={student.student_id}>
                  {student.name} ({student.class})
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {student && (
          <div className="mb-4 p-3 bg-muted/50 rounded-lg">
            <h4 className="font-medium">{student.name}</h4>
            <p className="text-sm text-muted-foreground">
              Class: {student.class} | Assessment Score: {student.assessment_score}
            </p>
          </div>
        )}

        <ChartContainer config={chartConfig}>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData} margin={{ top: 20, right: 30, bottom: 20, left: 30 }}>
              <PolarGrid />
              <PolarAngleAxis dataKey="skill" tick={{ fontSize: 12 }} />
              <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 10 }} />
              <ChartTooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload
                    return (
                      <div className="bg-background border rounded-lg p-2 shadow-lg">
                        <p className="font-medium">{data.skill}</p>
                        <p className="text-sm">
                          Value:{" "}
                          {data.skill === "Engagement" ? `${(data.value / 12.5).toFixed(1)}h` : data.value.toFixed(1)}
                        </p>
                      </div>
                    )
                  }
                  return null
                }}
              />
              <Radar
                name="Skills"
                dataKey="value"
                stroke="var(--color-value)"
                fill="var(--color-value)"
                fillOpacity={0.3}
                strokeWidth={2}
              />
            </RadarChart>
          </ResponsiveContainer>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
