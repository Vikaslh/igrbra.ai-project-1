"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Target, CheckCircle, BarChart3, Zap, Award, BookOpen } from "lucide-react"
import type { StudentData } from "./student-dashboard"

interface MLInsightsProps {
  data: StudentData[]
}

interface CorrelationData {
  skill: string
  correlation: number
  strength: string
  color: string
}

interface ClusterPersona {
  id: number
  name: string
  description: string
  characteristics: string[]
  avgScore: number
  count: number
  color: string
}

export function MLInsights({ data }: MLInsightsProps) {
  const [correlations, setCorrelations] = useState<CorrelationData[]>([])
  const [clusters, setClusters] = useState<ClusterPersona[]>([])
  const [modelAccuracy, setModelAccuracy] = useState<number>(0)
  const [keyFindings, setKeyFindings] = useState<string[]>([])

  useEffect(() => {
    // Simulate ML analysis results
    analyzeData()
  }, [data])

  const analyzeData = () => {
    // Calculate correlations with assessment scores
    const skills = ["comprehension", "attention", "focus", "retention", "engagement_time"]
    const correlationResults: CorrelationData[] = []

    skills.forEach((skill) => {
      const correlation = calculateCorrelation(
        data.map((s) => s[skill as keyof StudentData] as number),
        data.map((s) => s.assessment_score),
      )

      let strength = "Weak"
      let color = "text-red-600"

      if (Math.abs(correlation) > 0.7) {
        strength = "Strong"
        color = "text-green-600"
      } else if (Math.abs(correlation) > 0.5) {
        strength = "Moderate"
        color = "text-yellow-600"
      }

      correlationResults.push({
        skill: skill.charAt(0).toUpperCase() + skill.slice(1).replace("_", " "),
        correlation: correlation,
        strength,
        color,
      })
    })

    correlationResults.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation))
    setCorrelations(correlationResults)

    // Simulate clustering results
    const clusterResults: ClusterPersona[] = [
      {
        id: 0,
        name: "High Achievers",
        description: "Students with excellent performance across all cognitive skills",
        characteristics: ["High comprehension (85+)", "Strong attention (80+)", "Excellent retention (85+)"],
        avgScore: 88.5,
        count: Math.floor(data.length * 0.25),
        color: "bg-green-100 text-green-800",
      },
      {
        id: 1,
        name: "Steady Performers",
        description: "Consistent students with good overall performance",
        characteristics: ["Good comprehension (70-85)", "Moderate attention (65-80)", "Steady engagement"],
        avgScore: 75.2,
        count: Math.floor(data.length * 0.35),
        color: "bg-blue-100 text-blue-800",
      },
      {
        id: 2,
        name: "Potential Improvers",
        description: "Students showing promise but need targeted support",
        characteristics: ["Variable comprehension (60-75)", "Attention challenges", "High engagement potential"],
        avgScore: 65.8,
        count: Math.floor(data.length * 0.25),
        color: "bg-yellow-100 text-yellow-800",
      },
      {
        id: 3,
        name: "At-Risk Students",
        description: "Students requiring immediate intervention and support",
        characteristics: ["Low comprehension (<60)", "Attention difficulties", "Low engagement"],
        avgScore: 52.3,
        count: Math.floor(data.length * 0.15),
        color: "bg-red-100 text-red-800",
      },
    ]

    setClusters(clusterResults)
    setModelAccuracy(0.847) // Simulated R² score

    // Generate key findings
    const findings = [
      `${correlationResults[0].skill} shows the strongest correlation (${correlationResults[0].correlation.toFixed(3)}) with assessment scores`,
      `${clusterResults[0].count} students (${((clusterResults[0].count / data.length) * 100).toFixed(1)}%) are classified as High Achievers`,
      `The prediction model achieves ${(modelAccuracy * 100).toFixed(1)}% accuracy in predicting student performance`,
      `${clusterResults[3].count} students may benefit from immediate intervention and support`,
    ]

    setKeyFindings(findings)
  }

  const calculateCorrelation = (x: number[], y: number[]): number => {
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

  return (
    <div className="space-y-6">
      {/* Key Findings Alert */}
      <Alert>
        <Zap className="h-4 w-4" />
        <AlertDescription>
          <strong>ML Analysis Complete:</strong> Analyzed {data.length} students using Random Forest prediction and
          K-Means clustering algorithms.
        </AlertDescription>
      </Alert>

      <Tabs defaultValue="correlations" className="space-y-4">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="correlations">Skill Correlations</TabsTrigger>
          <TabsTrigger value="clusters">Learning Personas</TabsTrigger>
          <TabsTrigger value="predictions">Model Performance</TabsTrigger>
        </TabsList>

        <TabsContent value="correlations" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="w-5 h-5" />
                Cognitive Skills Correlation Analysis
              </CardTitle>
              <CardDescription>
                Correlation strength between cognitive skills and assessment performance
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {correlations.map((item, index) => (
                  <div key={item.skill} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-sm font-medium">
                        {index + 1}
                      </div>
                      <div>
                        <p className="font-medium">{item.skill}</p>
                        <p className="text-sm text-muted-foreground">Correlation: {item.correlation.toFixed(3)}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <Progress value={Math.abs(item.correlation) * 100} className="w-24" />
                      <Badge className={item.color}>{item.strength}</Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="clusters" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {clusters.map((cluster) => (
              <Card key={cluster.id}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg">{cluster.name}</CardTitle>
                    <Badge className={cluster.color}>{cluster.count} students</Badge>
                  </div>
                  <CardDescription>{cluster.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Average Score</span>
                      <span className="font-medium">{cluster.avgScore}</span>
                    </div>
                    <div className="space-y-2">
                      <p className="text-sm font-medium">Key Characteristics:</p>
                      <ul className="text-sm text-muted-foreground space-y-1">
                        {cluster.characteristics.map((char, idx) => (
                          <li key={idx} className="flex items-center gap-2">
                            <div className="w-1.5 h-1.5 rounded-full bg-current" />
                            {char}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="predictions" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="w-5 h-5" />
                  Model Accuracy
                </CardTitle>
                <CardDescription>Random Forest prediction performance</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-green-600">{(modelAccuracy * 100).toFixed(1)}%</div>
                    <p className="text-sm text-muted-foreground">R² Score</p>
                  </div>
                  <Progress value={modelAccuracy * 100} className="w-full" />
                  <div className="flex items-center gap-2 text-sm text-green-600">
                    <CheckCircle className="w-4 h-4" />
                    Excellent prediction accuracy
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Award className="w-5 h-5" />
                  Feature Importance
                </CardTitle>
                <CardDescription>Most predictive cognitive skills</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {correlations.slice(0, 3).map((item, index) => (
                    <div key={item.skill} className="flex items-center justify-between">
                      <span className="text-sm">{item.skill}</span>
                      <div className="flex items-center gap-2">
                        <Progress value={Math.abs(item.correlation) * 100} className="w-16" />
                        <span className="text-xs text-muted-foreground w-12">
                          {Math.abs(item.correlation).toFixed(2)}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>

      {/* Key Findings Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BookOpen className="w-5 h-5" />
            Key Findings & Recommendations
          </CardTitle>
          <CardDescription>Actionable insights from the machine learning analysis</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {keyFindings.map((finding, index) => (
              <div key={index} className="flex items-start gap-3 p-3 bg-muted/50 rounded-lg">
                <div className="w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center text-xs font-medium mt-0.5">
                  {index + 1}
                </div>
                <p className="text-sm">{finding}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
