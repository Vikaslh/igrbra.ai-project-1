"use client"

import { Scatter, ScatterChart, CartesianGrid, XAxis, YAxis, ResponsiveContainer } from "recharts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ChartContainer, ChartTooltip } from "@/components/ui/chart"
import type { StudentData } from "../student-dashboard"

interface AttentionPerformanceScatterProps {
  data: StudentData[]
}

export function AttentionPerformanceScatter({ data }: AttentionPerformanceScatterProps) {
  const scatterData = data.map((student) => ({
    attention: student.attention,
    assessment_score: student.assessment_score,
    name: student.name,
    class: student.class,
  }))

  const chartConfig = {
    attention: {
      label: "Attention Level",
      color: "hsl(var(--chart-3))",
    },
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Attention vs Assessment Performance</CardTitle>
        <CardDescription>Relationship between student attention levels and assessment scores</CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig}>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart data={scatterData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                type="number"
                dataKey="attention"
                name="Attention"
                tick={{ fontSize: 12 }}
                label={{ value: "Attention Level", position: "insideBottom", offset: -10 }}
              />
              <YAxis
                type="number"
                dataKey="assessment_score"
                name="Score"
                tick={{ fontSize: 12 }}
                label={{ value: "Assessment Score", angle: -90, position: "insideLeft" }}
              />
              <ChartTooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload
                    return (
                      <div className="bg-background border rounded-lg p-3 shadow-lg">
                        <p className="font-medium">{data.name}</p>
                        <p className="text-sm text-muted-foreground">Class: {data.class}</p>
                        <p className="text-sm">Attention: {data.attention}</p>
                        <p className="text-sm">Score: {data.assessment_score}</p>
                      </div>
                    )
                  }
                  return null
                }}
              />
              <Scatter dataKey="assessment_score" fill="var(--color-attention)" />
            </ScatterChart>
          </ResponsiveContainer>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
