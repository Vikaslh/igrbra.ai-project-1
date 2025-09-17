"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Upload, FileText, AlertCircle, BarChart3, Users, Lightbulb, TrendingUp } from "lucide-react"
import { DataUpload } from "./data-upload"
import { DashboardOverview } from "./dashboard-overview"
import { StudentTable } from "./student-table"
import { MLInsights } from "./ml-insights"
import { DatasetManager } from "./dataset-manager"
import { AdvancedAnalyticsPanel } from "./advanced-analytics-panel"
import { ComparisonTools } from "./comparison-tools"
import { ExportTools } from "./export-tools"
import { DataStorage } from "@/lib/storage"

export interface StudentData {
  student_id: string
  name: string
  class: string
  comprehension: number
  attention: number
  focus: number
  retention: number
  assessment_score: number
  engagement_time: number
  cluster?: number
}

export function StudentDashboard() {
  const [studentData, setStudentData] = useState<StudentData[]>([])
  const [isDataLoaded, setIsDataLoaded] = useState(false)
  const [currentDatasetId, setCurrentDatasetId] = useState<string>()

  useEffect(() => {
    const savedDataset = DataStorage.getCurrentDataset()
    if (savedDataset) {
      setStudentData(savedDataset.data)
      setIsDataLoaded(true)
      setCurrentDatasetId(savedDataset.id)
    }
  }, [])

  const handleDataUpload = (data: StudentData[]) => {
    const datasetId = DataStorage.saveDataset(`Dataset ${new Date().toLocaleDateString()}`, data)
    setStudentData(data)
    setIsDataLoaded(true)
    setCurrentDatasetId(datasetId)
  }

  const handleDatasetSelect = (data: StudentData[]) => {
    setStudentData(data)
    setIsDataLoaded(true)
    const current = DataStorage.getCurrentDataset()
    setCurrentDatasetId(current?.id)
  }

  if (!isDataLoaded) {
    return (
      <div className="container mx-auto p-6">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-foreground mb-4">Student Cognitive Skills & Performance Dashboard</h1>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            Upload your student data to analyze cognitive skills, performance patterns, and generate insights using
            machine learning.
          </p>
        </div>

        <div className="max-w-2xl mx-auto">
          <Card>
            <CardHeader className="text-center">
              <div className="mx-auto w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mb-4">
                <Upload className="w-8 h-8 text-primary" />
              </div>
              <CardTitle>Upload Student Data</CardTitle>
              <CardDescription>Upload a CSV file with student cognitive skills and performance data</CardDescription>
            </CardHeader>
            <CardContent>
              <DataUpload onDataUpload={handleDataUpload} />

              <div className="mt-6 p-4 bg-muted/50 rounded-lg">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-muted-foreground mt-0.5" />
                  <div className="text-sm text-muted-foreground">
                    <p className="font-medium mb-2">Required CSV columns:</p>
                    <ul className="space-y-1">
                      <li>• student_id, name, class</li>
                      <li>• comprehension, attention, focus, retention</li>
                      <li>• assessment_score, engagement_time</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="mt-6 pt-6 border-t">
                <div className="text-center">
                  <p className="text-sm text-muted-foreground mb-4">Or load a previously saved dataset</p>
                  <DatasetManager onDatasetSelect={handleDatasetSelect} />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    )
  }

  return (
    <div className="container mx-auto p-6">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-4xl font-bold text-foreground mb-2">Student Dashboard</h1>
          <p className="text-muted-foreground">
            Analyzing {studentData.length} students across cognitive skills and performance
          </p>
        </div>
        <div className="flex gap-2 flex-wrap">
          <ComparisonTools data={studentData} />
          <ExportTools data={studentData} />
          <DatasetManager onDatasetSelect={handleDatasetSelect} currentDatasetId={currentDatasetId} />
          <Button variant="outline" onClick={() => setIsDataLoaded(false)} className="gap-2">
            <FileText className="w-4 h-4" />
            Upload New Data
          </Button>
        </div>
      </div>

      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview" className="gap-2">
            <BarChart3 className="w-4 h-4" />
            Overview & Charts
          </TabsTrigger>
          <TabsTrigger value="students" className="gap-2">
            <Users className="w-4 h-4" />
            Student Directory
          </TabsTrigger>
          <TabsTrigger value="insights" className="gap-2">
            <Lightbulb className="w-4 h-4" />
            ML Insights
          </TabsTrigger>
          <TabsTrigger value="analytics" className="gap-2">
            <TrendingUp className="w-4 h-4" />
            Advanced Analytics
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          <DashboardOverview data={studentData} />
        </TabsContent>

        <TabsContent value="students" className="space-y-6">
          <StudentTable data={studentData} />
        </TabsContent>

        <TabsContent value="insights" className="space-y-6">
          <MLInsights data={studentData} />
        </TabsContent>

        <TabsContent value="analytics" className="space-y-6">
          <AdvancedAnalyticsPanel data={studentData} />
        </TabsContent>
      </Tabs>
    </div>
  )
}
