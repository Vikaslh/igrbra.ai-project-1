"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Users } from "lucide-react"
import { AdvancedAnalytics, type ClassComparison } from "@/lib/advanced-analytics"
import type { StudentData } from "./student-dashboard"

interface AdvancedAnalyticsPanelProps {
  data: StudentData[]
}

export function AdvancedAnalyticsPanel({ data }: AdvancedAnalyticsPanelProps) {
  const [classComparisons, setClassComparisons] = useState<ClassComparison[]>([])

  useEffect(() => {
    if (data.length > 0) {
      setClassComparisons(AdvancedAnalytics.generateClassComparisons(data))
    }
  }, [data])

  const chartConfig = {
    avgScore: {
      label: "Average Score",
      color: "hsl(var(--chart-1))",
    },
    topPerformers: {
      label: "Top Performers",
      color: "hsl(var(--chart-2))",
    },
    atRisk: {
      label: "At Risk",
      color: "hsl(var(--chart-3))",
    },
  }

  const getRiskBadge = (level: string) => {
    switch (level) {
      case "high":
        return <Badge className="bg-red-100 text-red-800">High Risk</Badge>
      case "medium":
        return <Badge className="bg-yellow-100 text-yellow-800">Medium Risk</Badge>
      default:
        return <Badge className="bg-green-100 text-green-800">Low Risk</Badge>
    }
  }

  const getPerformanceBadge = (rating: string) => {
    switch (rating) {
      case "excellent":
        return <Badge className="bg-green-100 text-green-800">Excellent</Badge>
      case "good":
        return <Badge className="bg-blue-100 text-blue-800">Good</Badge>
      case "average":
        return <Badge className="bg-yellow-100 text-yellow-800">Average</Badge>
      default:
        return <Badge className="bg-red-100 text-red-800">Needs Improvement</Badge>
    }
  }

  return (
    <div className="space-y-6">
      <Tabs defaultValue="comparisons" className="space-y-4">
        <TabsList className="grid w-full grid-cols-1">
          <TabsTrigger value="comparisons">Class Comparisons</TabsTrigger>
        </TabsList>

        <TabsContent value="comparisons" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Users className="w-5 h-5" />
                Class Performance Comparison
              </CardTitle>
              <CardDescription>
                Comparative analysis across different classes showing strengths and areas for improvement
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {classComparisons.map((comparison) => (
                  <div key={comparison.className} className="border rounded-lg p-4">
                    <div className="flex items-start justify-between mb-3">
                      <div>
                        <h4 className="font-medium">{comparison.className}</h4>
                        <p className="text-sm text-muted-foreground">{comparison.studentCount} students</p>
                      </div>
                      {getPerformanceBadge(comparison.performanceRating)}
                    </div>

                    <div className="space-y-3">
                      <div>
                        <p className="text-sm text-muted-foreground">Average Score</p>
                        <p className="text-2xl font-bold">{comparison.avgScore}</p>
                      </div>

                      <div className="grid grid-cols-2 gap-3 text-sm">
                        <div>
                          <p className="text-muted-foreground">Strongest Skill</p>
                          <p className="font-medium capitalize text-green-600">{comparison.topSkill}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Needs Focus</p>
                          <p className="font-medium capitalize text-orange-600">{comparison.weakestSkill}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
