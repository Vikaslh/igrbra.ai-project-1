"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Bar,
  BarChart,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { GitCompare, Download } from "lucide-react"
import type { StudentData } from "./student-dashboard"

interface ComparisonToolsProps {
  data: StudentData[]
}

export function ComparisonTools({ data }: ComparisonToolsProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [selectedStudents, setSelectedStudents] = useState<string[]>([])
  const [selectedClasses, setSelectedClasses] = useState<string[]>([])
  const [comparisonType, setComparisonType] = useState<"students" | "classes">("students")

  const classes = Array.from(new Set(data.map((s) => s.class))).sort()

  const handleStudentToggle = (studentId: string) => {
    setSelectedStudents((prev) =>
      prev.includes(studentId) ? prev.filter((id) => id !== studentId) : prev.length < 5 ? [...prev, studentId] : prev,
    )
  }

  const handleClassToggle = (className: string) => {
    setSelectedClasses((prev) =>
      prev.includes(className)
        ? prev.filter((name) => name !== className)
        : prev.length < 4
          ? [...prev, className]
          : prev,
    )
  }

  const getStudentComparisonData = () => {
    const students = data.filter((s) => selectedStudents.includes(s.student_id))

    const skills = ["comprehension", "attention", "focus", "retention"] as const

    return skills.map((skill) => ({
      skill: skill.charAt(0).toUpperCase() + skill.slice(1),
      ...students.reduce(
        (acc, student) => ({
          ...acc,
          [student.name]: student[skill],
        }),
        {},
      ),
    }))
  }

  const getClassComparisonData = () => {
    const skills = ["comprehension", "attention", "focus", "retention", "assessment_score"] as const

    return skills.map((skill) => {
      const skillData: any = {
        skill: skill === "assessment_score" ? "Assessment" : skill.charAt(0).toUpperCase() + skill.slice(1),
      }

      selectedClasses.forEach((className) => {
        const classStudents = data.filter((s) => s.class === className)
        const average = classStudents.reduce((sum, s) => sum + s[skill], 0) / classStudents.length
        skillData[className] = Math.round(average * 10) / 10
      })

      return skillData
    })
  }

  const getStudentRadarData = (studentId: string) => {
    const student = data.find((s) => s.student_id === studentId)
    if (!student) return []

    return [
      { skill: "Comprehension", value: student.comprehension },
      { skill: "Attention", value: student.attention },
      { skill: "Focus", value: student.focus },
      { skill: "Retention", value: student.retention },
      { skill: "Engagement", value: student.engagement_time * 12.5 }, // Scale to 0-100
    ]
  }

  const generateComparisonReport = () => {
    if (comparisonType === "students") {
      const students = data.filter((s) => selectedStudents.includes(s.student_id))
      const report = {
        type: "Student Comparison",
        date: new Date().toISOString(),
        students: students.map((s) => ({
          name: s.name,
          id: s.student_id,
          class: s.class,
          assessment_score: s.assessment_score,
          comprehension: s.comprehension,
          attention: s.attention,
          focus: s.focus,
          retention: s.retention,
          engagement_time: s.engagement_time,
        })),
        summary: {
          highest_scorer: students.reduce((max, s) => (s.assessment_score > max.assessment_score ? s : max)).name,
          lowest_scorer: students.reduce((min, s) => (s.assessment_score < min.assessment_score ? s : min)).name,
          average_score: students.reduce((sum, s) => sum + s.assessment_score, 0) / students.length,
        },
      }

      const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" })
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `student_comparison_${new Date().toISOString().split("T")[0]}.json`
      a.click()
      URL.revokeObjectURL(url)
    } else {
      const classData = selectedClasses.map((className) => {
        const classStudents = data.filter((s) => s.class === className)
        return {
          class: className,
          student_count: classStudents.length,
          average_score: classStudents.reduce((sum, s) => sum + s.assessment_score, 0) / classStudents.length,
          average_comprehension: classStudents.reduce((sum, s) => sum + s.comprehension, 0) / classStudents.length,
          average_attention: classStudents.reduce((sum, s) => sum + s.attention, 0) / classStudents.length,
          average_focus: classStudents.reduce((sum, s) => sum + s.focus, 0) / classStudents.length,
          average_retention: classStudents.reduce((sum, s) => sum + s.retention, 0) / classStudents.length,
          top_performer: classStudents.reduce((max, s) => (s.assessment_score > max.assessment_score ? s : max)).name,
        }
      })

      const report = {
        type: "Class Comparison",
        date: new Date().toISOString(),
        classes: classData,
        summary: {
          best_performing_class: classData.reduce((max, c) => (c.average_score > max.average_score ? c : max)).class,
          total_students: classData.reduce((sum, c) => sum + c.student_count, 0),
        },
      }

      const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" })
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `class_comparison_${new Date().toISOString().split("T")[0]}.json`
      a.click()
      URL.revokeObjectURL(url)
    }
  }

  const chartConfig = {
    comprehension: { label: "Comprehension", color: "hsl(var(--chart-1))" },
    attention: { label: "Attention", color: "hsl(var(--chart-2))" },
    focus: { label: "Focus", color: "hsl(var(--chart-3))" },
    retention: { label: "Retention", color: "hsl(var(--chart-4))" },
    assessment: { label: "Assessment", color: "hsl(var(--chart-5))" },
    value: { label: "Score", color: "hsl(var(--chart-1))" },
  }

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" className="gap-2 bg-transparent">
          <GitCompare className="w-4 h-4" />
          Compare Students & Classes
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-6xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <GitCompare className="w-5 h-5" />
            Comparison Tools
          </DialogTitle>
          <DialogDescription>
            Compare students or classes side-by-side to identify patterns and performance differences
          </DialogDescription>
        </DialogHeader>

        <Tabs
          value={comparisonType}
          onValueChange={(value) => setComparisonType(value as "students" | "classes")}
          className="space-y-6"
        >
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="students">Student Comparison</TabsTrigger>
            <TabsTrigger value="classes">Class Comparison</TabsTrigger>
          </TabsList>

          <TabsContent value="students" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span>Select Students to Compare</span>
                  <Badge variant="secondary">{selectedStudents.length}/5 selected</Badge>
                </CardTitle>
                <CardDescription>Choose up to 5 students for detailed comparison</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 max-h-60 overflow-y-auto">
                  {data.map((student) => (
                    <div
                      key={student.student_id}
                      className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                        selectedStudents.includes(student.student_id)
                          ? "bg-primary/10 border-primary"
                          : "hover:bg-muted/50"
                      }`}
                      onClick={() => handleStudentToggle(student.student_id)}
                    >
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="font-medium text-sm">{student.name}</p>
                          <p className="text-xs text-muted-foreground">
                            {student.class} • Score: {student.assessment_score}
                          </p>
                        </div>
                        {selectedStudents.includes(student.student_id) && (
                          <Badge variant="default" className="text-xs">
                            Selected
                          </Badge>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {selectedStudents.length >= 2 && (
              <>
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center justify-between">
                      Skills Comparison
                      <Button
                        onClick={generateComparisonReport}
                        size="sm"
                        variant="outline"
                        className="gap-2 bg-transparent"
                      >
                        <Download className="w-4 h-4" />
                        Export Report
                      </Button>
                    </CardTitle>
                    <CardDescription>Side-by-side comparison of cognitive skills</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ChartContainer config={chartConfig}>
                      <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={getStudentComparisonData()}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="skill" />
                          <YAxis />
                          <ChartTooltip content={<ChartTooltipContent />} />
                          {data
                            .filter((s) => selectedStudents.includes(s.student_id))
                            .map((student, index) => (
                              <Bar
                                key={student.student_id}
                                dataKey={student.name}
                                fill={`hsl(var(--chart-${(index % 5) + 1}))`}
                              />
                            ))}
                        </BarChart>
                      </ResponsiveContainer>
                    </ChartContainer>
                  </CardContent>
                </Card>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {data
                    .filter((s) => selectedStudents.includes(s.student_id))
                    .map((student) => (
                      <Card key={student.student_id}>
                        <CardHeader>
                          <CardTitle className="text-lg">{student.name}</CardTitle>
                          <CardDescription>
                            Class: {student.class} • Score: {student.assessment_score}
                          </CardDescription>
                        </CardHeader>
                        <CardContent>
                          <ChartContainer config={chartConfig}>
                            <ResponsiveContainer width="100%" height={200}>
                              <RadarChart data={getStudentRadarData(student.student_id)}>
                                <PolarGrid />
                                <PolarAngleAxis dataKey="skill" />
                                <PolarRadiusAxis angle={90} domain={[0, 100]} />
                                <ChartTooltip content={<ChartTooltipContent />} />
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
                    ))}
                </div>
              </>
            )}
          </TabsContent>

          <TabsContent value="classes" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span>Select Classes to Compare</span>
                  <Badge variant="secondary">{selectedClasses.length}/4 selected</Badge>
                </CardTitle>
                <CardDescription>Choose up to 4 classes for performance comparison</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {classes.map((className) => {
                    const classStudents = data.filter((s) => s.class === className)
                    const avgScore =
                      classStudents.reduce((sum, s) => sum + s.assessment_score, 0) / classStudents.length

                    return (
                      <div
                        key={className}
                        className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                          selectedClasses.includes(className) ? "bg-primary/10 border-primary" : "hover:bg-muted/50"
                        }`}
                        onClick={() => handleClassToggle(className)}
                      >
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="font-medium">{className}</p>
                            <p className="text-sm text-muted-foreground">
                              {classStudents.length} students • Avg: {avgScore.toFixed(1)}
                            </p>
                          </div>
                          {selectedClasses.includes(className) && <Badge variant="default">Selected</Badge>}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </CardContent>
            </Card>

            {selectedClasses.length >= 2 && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    Class Performance Comparison
                    <Button
                      onClick={generateComparisonReport}
                      size="sm"
                      variant="outline"
                      className="gap-2 bg-transparent"
                    >
                      <Download className="w-4 h-4" />
                      Export Report
                    </Button>
                  </CardTitle>
                  <CardDescription>Average performance across all cognitive skills and assessments</CardDescription>
                </CardHeader>
                <CardContent>
                  <ChartContainer config={chartConfig}>
                    <ResponsiveContainer width="100%" height={400}>
                      <BarChart data={getClassComparisonData()}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="skill" />
                        <YAxis />
                        <ChartTooltip content={<ChartTooltipContent />} />
                        {selectedClasses.map((className, index) => (
                          <Bar key={className} dataKey={className} fill={`hsl(var(--chart-${(index % 5) + 1}))`} />
                        ))}
                      </BarChart>
                    </ResponsiveContainer>
                  </ChartContainer>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  )
}
