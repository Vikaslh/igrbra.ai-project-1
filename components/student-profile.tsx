"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Line,
  LineChart,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import {
  User,
  Brain,
  Target,
  TrendingUp,
  Award,
  AlertTriangle,
  BookOpen,
  Clock,
  Eye,
  Users,
  Star,
  Zap,
} from "lucide-react"
import type { StudentData } from "./student-dashboard"

interface StudentProfileProps {
  student: StudentData
  allStudents: StudentData[]
  trigger?: React.ReactNode
}

export function StudentProfile({ student, allStudents, trigger }: StudentProfileProps) {
  const [isOpen, setIsOpen] = useState(false)

  // Calculate percentiles and rankings
  const calculatePercentile = (value: number, field: keyof StudentData) => {
    const values = allStudents.map((s) => s[field] as number).sort((a, b) => a - b)
    const rank = values.findIndex((v) => v >= value) + 1
    return Math.round((rank / values.length) * 100)
  }

  const getClassRank = () => {
    const classmates = allStudents.filter((s) => s.class === student.class)
    const sorted = classmates.sort((a, b) => b.assessment_score - a.assessment_score)
    const rank = sorted.findIndex((s) => s.student_id === student.student_id) + 1
    return { rank, total: classmates.length }
  }

  const getOverallRank = () => {
    const sorted = allStudents.sort((a, b) => b.assessment_score - a.assessment_score)
    const rank = sorted.findIndex((s) => s.student_id === student.student_id) + 1
    return { rank, total: allStudents.length }
  }

  // Generate skill analysis
  const skillAnalysis = [
    {
      skill: "Comprehension",
      value: student.comprehension,
      icon: Brain,
      color: "text-blue-600",
      description: "Understanding and interpreting information",
    },
    {
      skill: "Attention",
      value: student.attention,
      icon: Eye,
      color: "text-green-600",
      description: "Ability to focus and concentrate",
    },
    {
      skill: "Focus",
      value: student.focus,
      icon: Target,
      color: "text-purple-600",
      description: "Sustained concentration on tasks",
    },
    {
      skill: "Retention",
      value: student.retention,
      icon: BookOpen,
      color: "text-orange-600",
      description: "Memory and information retention",
    },
  ]

  // Generate performance insights
  const generateInsights = () => {
    const insights = []
    const avgScore = allStudents.reduce((sum, s) => sum + s.assessment_score, 0) / allStudents.length
    const classAvg =
      allStudents.filter((s) => s.class === student.class).reduce((sum, s) => sum + s.assessment_score, 0) /
      allStudents.filter((s) => s.class === student.class).length

    if (student.assessment_score > avgScore + 10) {
      insights.push({ type: "positive", text: "Performing significantly above average" })
    } else if (student.assessment_score < avgScore - 10) {
      insights.push({ type: "concern", text: "Performing below average - may need additional support" })
    }

    if (student.assessment_score > classAvg + 5) {
      insights.push({ type: "positive", text: "Top performer in class" })
    }

    const strongestSkill = skillAnalysis.reduce((max, skill) => (skill.value > max.value ? skill : max))
    const weakestSkill = skillAnalysis.reduce((min, skill) => (skill.value < min.value ? skill : min))

    insights.push({
      type: "info",
      text: `Strongest skill: ${strongestSkill.skill} (${strongestSkill.value}/100)`,
    })

    if (weakestSkill.value < 60) {
      insights.push({
        type: "concern",
        text: `${weakestSkill.skill} needs improvement (${weakestSkill.value}/100)`,
      })
    }

    if (student.engagement_time < 4) {
      insights.push({ type: "concern", text: "Low engagement time - consider motivation strategies" })
    } else if (student.engagement_time > 7) {
      insights.push({ type: "positive", text: "High engagement - excellent dedication" })
    }

    return insights
  }

  // Generate simulated progress data
  const progressData = Array.from({ length: 6 }, (_, i) => ({
    month: new Date(Date.now() - (5 - i) * 30 * 24 * 60 * 60 * 1000).toLocaleDateString("en-US", { month: "short" }),
    score: Math.max(0, Math.min(100, student.assessment_score + (Math.random() - 0.5) * 20)),
    comprehension: Math.max(0, Math.min(100, student.comprehension + (Math.random() - 0.5) * 15)),
    attention: Math.max(0, Math.min(100, student.attention + (Math.random() - 0.5) * 15)),
  }))

  const radarData = skillAnalysis.map((skill) => ({
    skill: skill.skill,
    value: skill.value,
    fullMark: 100,
  }))

  const classRank = getClassRank()
  const overallRank = getOverallRank()
  const insights = generateInsights()

  const chartConfig = {
    score: { label: "Assessment Score", color: "hsl(var(--chart-1))" },
    comprehension: { label: "Comprehension", color: "hsl(var(--chart-2))" },
    attention: { label: "Attention", color: "hsl(var(--chart-3))" },
    value: { label: "Skill Level", color: "hsl(var(--chart-4))" },
  }

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        {trigger || (
          <Button variant="outline" size="sm" className="hover:bg-primary/5 transition-colors bg-transparent">
            <User className="w-4 h-4 mr-2" />
            View Profile
          </Button>
        )}
      </DialogTrigger>
      <DialogContent className="max-w-[95vw] w-full max-h-[95vh] h-[95vh] overflow-hidden p-0 gap-0 flex flex-col">
        <DialogHeader className="px-8 py-6 bg-gradient-to-r from-blue-50 to-purple-50 border-b flex-shrink-0">
          <DialogTitle className="flex items-center gap-4">
            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg">
              <User className="w-8 h-8 text-white" />
            </div>
            <div className="flex-1">
              <h2 className="text-3xl font-bold text-gray-900 mb-1">{student.name}</h2>
              <div className="flex items-center gap-4 text-sm text-gray-600">
                <span className="flex items-center gap-1">
                  <span className="font-medium">ID:</span> {student.student_id}
                </span>
                <span className="flex items-center gap-1">
                  <span className="font-medium">Class:</span> {student.class}
                </span>
                <Badge variant="secondary" className="bg-blue-100 text-blue-700 border-blue-200">
                  <Star className="w-3 h-3 mr-1" />
                  Rank #{classRank.rank} in class
                </Badge>
              </div>
            </div>
          </DialogTitle>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto px-8 py-6 min-h-0">
          <Tabs defaultValue="overview" className="space-y-8">
            <TabsList className="grid w-full grid-cols-2 md:grid-cols-4 h-12 bg-gray-100 p-1 rounded-xl">
              <TabsTrigger
                value="overview"
                className="text-sm font-medium rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm"
              >
                <Award className="w-4 h-4 mr-2" />
                Overview
              </TabsTrigger>
              <TabsTrigger
                value="skills"
                className="text-sm font-medium rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm"
              >
                <Brain className="w-4 h-4 mr-2" />
                Skills
              </TabsTrigger>
              <TabsTrigger
                value="progress"
                className="text-sm font-medium rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm"
              >
                <TrendingUp className="w-4 h-4 mr-2" />
                Progress
              </TabsTrigger>
              <TabsTrigger
                value="insights"
                className="text-sm font-medium rounded-lg data-[state=active]:bg-white data-[state=active]:shadow-sm"
              >
                <Zap className="w-4 h-4 mr-2" />
                Insights
              </TabsTrigger>
            </TabsList>

            <TabsContent value="overview" className="space-y-8">
              <div className="flex gap-6 overflow-x-auto pb-4 md:grid md:grid-cols-3 lg:grid-cols-4 md:overflow-x-visible md:pb-0">
                <Card className="flex-shrink-0 w-[28rem] md:w-auto border-0 shadow-lg bg-gradient-to-br from-blue-50 to-blue-100 hover:shadow-xl transition-shadow">
                  <CardHeader className="pb-4">
                    <CardTitle className="text-base font-semibold flex items-center gap-3 min-w-0">
                      <div className="w-12 h-12 rounded-full bg-blue-500 flex items-center justify-center flex-shrink-0">
                        <Award className="w-6 h-6 text-white" />
                      </div>
                      <span className="text-gray-800 leading-tight">Assessment Score</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="pt-2">
                    <div className="text-4xl font-bold text-blue-700 mb-2">{student.assessment_score}</div>
                    <p className="text-sm text-blue-600 font-medium">
                      {calculatePercentile(student.assessment_score, "assessment_score")}th percentile
                    </p>
                  </CardContent>
                </Card>

                <Card className="flex-shrink-0 w-[28rem] md:w-auto border-0 shadow-lg bg-gradient-to-br from-green-50 to-green-100 hover:shadow-xl transition-shadow">
                  <CardHeader className="pb-4">
                    <CardTitle className="text-base font-semibold flex items-center gap-3 min-w-0">
                      <div className="w-12 h-12 rounded-full bg-green-500 flex items-center justify-center flex-shrink-0">
                        <Users className="w-6 h-6 text-white" />
                      </div>
                      <span className="text-gray-800 leading-tight">Class Rank</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="pt-2">
                    <div className="text-4xl font-bold text-green-700 mb-2">{classRank.rank}</div>
                    <p className="text-sm text-green-600 font-medium">of {classRank.total} students</p>
                  </CardContent>
                </Card>

                <Card className="flex-shrink-0 w-[28rem] md:w-auto border-0 shadow-lg bg-gradient-to-br from-purple-50 to-purple-100 hover:shadow-xl transition-shadow">
                  <CardHeader className="pb-4">
                    <CardTitle className="text-base font-semibold flex items-center gap-3 min-w-0">
                      <div className="w-12 h-12 rounded-full bg-purple-500 flex items-center justify-center flex-shrink-0">
                        <TrendingUp className="w-6 h-6 text-white" />
                      </div>
                      <span className="text-gray-800 leading-tight">Overall Rank</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="pt-2">
                    <div className="text-4xl font-bold text-purple-700 mb-2">{overallRank.rank}</div>
                    <p className="text-sm text-purple-600 font-medium">of {overallRank.total} students</p>
                  </CardContent>
                </Card>

                <Card className="flex-shrink-0 w-[28rem] md:w-auto border-0 shadow-lg bg-gradient-to-br from-orange-50 to-orange-100 hover:shadow-xl transition-shadow">
                  <CardHeader className="pb-4">
                    <CardTitle className="text-base font-semibold flex items-center gap-3 min-w-0">
                      <div className="w-12 h-12 rounded-full bg-orange-500 flex items-center justify-center flex-shrink-0">
                        <Clock className="w-6 h-6 text-white" />
                      </div>
                      <span className="text-gray-800 leading-tight">Engagement</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="pt-2">
                    <div className="text-4xl font-bold text-orange-700 mb-2">{student.engagement_time}h</div>
                    <p className="text-sm text-orange-600 font-medium">weekly average</p>
                  </CardContent>
                </Card>
              </div>

              <Card className="border-0 shadow-lg">
                <CardHeader className="pb-6">
                  <CardTitle className="text-xl font-bold text-gray-900 flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                      <Brain className="w-4 h-4 text-white" />
                    </div>
                    Cognitive Skills Profile
                  </CardTitle>
                  <CardDescription className="text-gray-600">
                    Comprehensive view of all cognitive abilities
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ChartContainer config={chartConfig}>
                    <ResponsiveContainer width="100%" height={350}>
                      <RadarChart data={radarData}>
                        <PolarGrid stroke="#e5e7eb" />
                        <PolarAngleAxis dataKey="skill" className="text-sm font-medium" />
                        <PolarRadiusAxis angle={90} domain={[0, 100]} className="text-xs" />
                        <ChartTooltip content={<ChartTooltipContent />} />
                        <Radar
                          name="Skills"
                          dataKey="value"
                          stroke="#3b82f6"
                          fill="url(#skillsGradient)"
                          fillOpacity={0.4}
                          strokeWidth={3}
                        />
                        <defs>
                          <linearGradient id="skillsGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stopColor="#3b82f6" />
                            <stop offset="100%" stopColor="#8b5cf6" />
                          </linearGradient>
                        </defs>
                      </RadarChart>
                    </ResponsiveContainer>
                  </ChartContainer>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="skills" className="space-y-8">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {skillAnalysis.map((skill) => {
                  const Icon = skill.icon
                  return (
                    <Card key={skill.skill} className="border-0 shadow-lg hover:shadow-xl transition-shadow">
                      <CardHeader className="pb-4">
                        <CardTitle className="text-lg flex items-center gap-3 min-w-0">
                          <div
                            className={`w-12 h-12 rounded-full ${skill.color.replace("text-", "bg-").replace("-600", "-500")} flex items-center justify-center`}
                          >
                            <Icon className="w-6 h-6 text-white" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <span className="font-bold text-gray-900 block">{skill.skill}</span>
                            <CardDescription className="text-sm mt-1">{skill.description}</CardDescription>
                          </div>
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-6">
                        <div className="flex items-center justify-between">
                          <span className="text-3xl font-bold text-gray-900">{skill.value}/100</span>
                        </div>
                        <Progress value={skill.value} className="w-full h-3" />
                        <div
                          className={`text-sm font-medium ${
                            skill.value >= 80
                              ? "text-green-600"
                              : skill.value >= 60
                                ? "text-blue-600"
                                : skill.value >= 40
                                  ? "text-yellow-600"
                                  : "text-red-600"
                          }`}
                        >
                          {skill.value >= 80
                            ? "🌟 Excellent performance"
                            : skill.value >= 60
                              ? "✅ Good performance"
                              : skill.value >= 40
                                ? "⚠️ Needs improvement"
                                : "🚨 Requires immediate attention"}
                        </div>
                      </CardContent>
                    </Card>
                  )
                })}
              </div>
            </TabsContent>

            <TabsContent value="progress" className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg break-words">Performance Trends</CardTitle>
                  <CardDescription className="break-words">
                    6-month progress tracking across key metrics
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ChartContainer config={chartConfig}>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={progressData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="month" />
                        <YAxis />
                        <ChartTooltip content={<ChartTooltipContent />} />
                        <Line
                          type="monotone"
                          dataKey="score"
                          stroke="#3b82f6"
                          strokeWidth={3}
                          dot={{ fill: "#3b82f6", strokeWidth: 2, r: 4 }}
                        />
                        <Line
                          type="monotone"
                          dataKey="comprehension"
                          stroke="#10b981"
                          strokeWidth={2}
                          strokeDasharray="5 5"
                        />
                        <Line
                          type="monotone"
                          dataKey="attention"
                          stroke="#8b5cf6"
                          strokeWidth={2}
                          strokeDasharray="5 5"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </ChartContainer>
                </CardContent>
              </Card>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm break-words">6-Month Change</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold text-green-600">
                      +{(progressData[progressData.length - 1].score - progressData[0].score).toFixed(1)}
                    </div>
                    <p className="text-sm text-muted-foreground">Assessment score improvement</p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm break-words">Best Month</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">
                      {progressData.reduce((best, current) => (current.score > best.score ? current : best)).month}
                    </div>
                    <p className="text-sm text-muted-foreground">Highest performance period</p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm break-words">Consistency</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">
                      {(
                        100 -
                        (Math.max(...progressData.map((p) => p.score)) - Math.min(...progressData.map((p) => p.score)))
                      ).toFixed(0)}
                      %
                    </div>
                    <p className="text-sm text-muted-foreground">Performance stability</p>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="insights" className="space-y-8">
              <Card className="border-0 shadow-lg">
                <CardHeader className="pb-6">
                  <CardTitle className="text-xl font-bold text-gray-900 flex items-center gap-3">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                      <Brain className="w-4 h-4 text-white" />
                    </div>
                    Performance Insights
                  </CardTitle>
                  <CardDescription className="text-gray-600">AI-generated analysis and recommendations</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {insights.map((insight, index) => (
                      <div
                        key={index}
                        className={`flex items-start gap-4 p-6 rounded-xl border-l-4 ${
                          insight.type === "positive"
                            ? "bg-green-50 border-l-green-500 border border-green-200"
                            : insight.type === "concern"
                              ? "bg-red-50 border-l-red-500 border border-red-200"
                              : "bg-blue-50 border-l-blue-500 border border-blue-200"
                        }`}
                      >
                        <div
                          className={`w-10 h-10 rounded-full flex items-center justify-center ${
                            insight.type === "positive"
                              ? "bg-green-500"
                              : insight.type === "concern"
                                ? "bg-red-500"
                                : "bg-blue-500"
                          }`}
                        >
                          {insight.type === "positive" ? (
                            <Award className="w-5 h-5 text-white" />
                          ) : insight.type === "concern" ? (
                            <AlertTriangle className="w-5 h-5 text-white" />
                          ) : (
                            <Brain className="w-5 h-5 text-white" />
                          )}
                        </div>
                        <p className="text-sm font-medium text-gray-800 leading-relaxed">{insight.text}</p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-lg break-words">Personalized Recommendations</CardTitle>
                  <CardDescription className="break-words">Tailored strategies for improvement</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="p-4 border rounded-lg">
                      <h4 className="font-medium mb-2">Learning Style Optimization</h4>
                      <p className="text-sm text-muted-foreground">
                        Based on cognitive profile, recommend visual learning aids and interactive exercises
                      </p>
                    </div>
                    <div className="p-4 border rounded-lg">
                      <h4 className="font-medium mb-2">Skill Development Focus</h4>
                      <p className="text-sm text-muted-foreground">
                        Prioritize{" "}
                        {skillAnalysis
                          .reduce((min, skill) => (skill.value < min.value ? skill : min))
                          .skill.toLowerCase()}
                        improvement through targeted practice sessions
                      </p>
                    </div>
                    <div className="p-4 border rounded-lg">
                      <h4 className="font-medium mb-2">Engagement Strategy</h4>
                      <p className="text-sm text-muted-foreground">
                        {student.engagement_time < 5
                          ? "Implement gamification and shorter study sessions to increase engagement"
                          : "Maintain current engagement levels with challenging advanced materials"}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </DialogContent>
    </Dialog>
  )
}
