"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Download, FileText, FileSpreadsheet, FileImage, Settings } from "lucide-react"
import type { StudentData } from "./student-dashboard"

interface ExportToolsProps {
  data: StudentData[]
}

export function ExportTools({ data }: ExportToolsProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [exportFormat, setExportFormat] = useState<"csv" | "json" | "pdf">("csv")
  const [selectedFields, setSelectedFields] = useState<string[]>(["student_id", "name", "class", "assessment_score"])
  const [filterClass, setFilterClass] = useState<string>("all")
  const [reportTitle, setReportTitle] = useState("Student Performance Report")

  const availableFields = [
    { id: "student_id", label: "Student ID" },
    { id: "name", label: "Name" },
    { id: "class", label: "Class" },
    { id: "comprehension", label: "Comprehension" },
    { id: "attention", label: "Attention" },
    { id: "focus", label: "Focus" },
    { id: "retention", label: "Retention" },
    { id: "assessment_score", label: "Assessment Score" },
    { id: "engagement_time", label: "Engagement Time" },
  ]

  const classes = Array.from(new Set(data.map((s) => s.class))).sort()

  const handleFieldToggle = (fieldId: string) => {
    setSelectedFields((prev) => (prev.includes(fieldId) ? prev.filter((id) => id !== fieldId) : [...prev, fieldId]))
  }

  const getFilteredData = () => {
    return filterClass === "all" ? data : data.filter((student) => student.class === filterClass)
  }

  const exportToCSV = () => {
    const filteredData = getFilteredData()
    const headers = selectedFields.map((field) => availableFields.find((f) => f.id === field)?.label || field)

    const csvContent = [
      headers.join(","),
      ...filteredData.map((student) => selectedFields.map((field) => student[field as keyof StudentData]).join(",")),
    ].join("\n")

    const blob = new Blob([csvContent], { type: "text/csv" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `${reportTitle.replace(/\s+/g, "_").toLowerCase()}_${new Date().toISOString().split("T")[0]}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  const exportToJSON = () => {
    const filteredData = getFilteredData()
    const exportData = {
      title: reportTitle,
      exportDate: new Date().toISOString(),
      totalStudents: filteredData.length,
      classFilter: filterClass,
      fields: selectedFields,
      data: filteredData.map((student) => {
        const filtered: any = {}
        selectedFields.forEach((field) => {
          filtered[field] = student[field as keyof StudentData]
        })
        return filtered
      }),
      summary: {
        averageScore: filteredData.reduce((sum, s) => sum + s.assessment_score, 0) / filteredData.length,
        highestScore: Math.max(...filteredData.map((s) => s.assessment_score)),
        lowestScore: Math.min(...filteredData.map((s) => s.assessment_score)),
        classDistribution: classes.reduce(
          (acc, className) => {
            acc[className] = filteredData.filter((s) => s.class === className).length
            return acc
          },
          {} as Record<string, number>,
        ),
      },
    }

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `${reportTitle.replace(/\s+/g, "_").toLowerCase()}_${new Date().toISOString().split("T")[0]}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const exportToPDF = () => {
    const filteredData = getFilteredData()

    // Create HTML content for PDF
    const htmlContent = `
      <!DOCTYPE html>
      <html>
      <head>
        <title>${reportTitle}</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 20px; }
          h1 { color: #333; border-bottom: 2px solid #333; padding-bottom: 10px; }
          .summary { background: #f5f5f5; padding: 15px; margin: 20px 0; border-radius: 5px; }
          table { width: 100%; border-collapse: collapse; margin: 20px 0; }
          th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
          th { background-color: #f2f2f2; font-weight: bold; }
          .footer { margin-top: 30px; font-size: 12px; color: #666; }
        </style>
      </head>
      <body>
        <h1>${reportTitle}</h1>
        <div class="summary">
          <h3>Report Summary</h3>
          <p><strong>Export Date:</strong> ${new Date().toLocaleDateString()}</p>
          <p><strong>Total Students:</strong> ${filteredData.length}</p>
          <p><strong>Class Filter:</strong> ${filterClass === "all" ? "All Classes" : filterClass}</p>
          <p><strong>Average Score:</strong> ${(filteredData.reduce((sum, s) => sum + s.assessment_score, 0) / filteredData.length).toFixed(1)}</p>
        </div>
        <table>
          <thead>
            <tr>
              ${selectedFields
                .map((field) => `<th>${availableFields.find((f) => f.id === field)?.label || field}</th>`)
                .join("")}
            </tr>
          </thead>
          <tbody>
            ${filteredData
              .map(
                (student) =>
                  `<tr>
                ${selectedFields.map((field) => `<td>${student[field as keyof StudentData]}</td>`).join("")}
              </tr>`,
              )
              .join("")}
          </tbody>
        </table>
        <div class="footer">
          <p>Generated by Student Performance Dashboard</p>
        </div>
      </body>
      </html>
    `

    const blob = new Blob([htmlContent], { type: "text/html" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `${reportTitle.replace(/\s+/g, "_").toLowerCase()}_${new Date().toISOString().split("T")[0]}.html`
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleExport = () => {
    switch (exportFormat) {
      case "csv":
        exportToCSV()
        break
      case "json":
        exportToJSON()
        break
      case "pdf":
        exportToPDF()
        break
    }
    setIsOpen(false)
  }

  const getPreviewData = () => {
    const filteredData = getFilteredData().slice(0, 3)
    return filteredData.map((student) => {
      const filtered: any = {}
      selectedFields.forEach((field) => {
        filtered[field] = student[field as keyof StudentData]
      })
      return filtered
    })
  }

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" className="gap-2 bg-transparent">
          <Download className="w-4 h-4" />
          Export Data
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Download className="w-5 h-5" />
            Export Tools
          </DialogTitle>
          <DialogDescription>Export student data and generate reports in various formats</DialogDescription>
        </DialogHeader>

        <div className="space-y-6">
          {/* Export Configuration */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="w-4 h-4" />
                Export Configuration
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="report-title">Report Title</Label>
                  <Input
                    id="report-title"
                    value={reportTitle}
                    onChange={(e) => setReportTitle(e.target.value)}
                    placeholder="Enter report title..."
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="export-format">Export Format</Label>
                  <Select value={exportFormat} onValueChange={(value: any) => setExportFormat(value)}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="csv">
                        <div className="flex items-center gap-2">
                          <FileSpreadsheet className="w-4 h-4" />
                          CSV (Excel Compatible)
                        </div>
                      </SelectItem>
                      <SelectItem value="json">
                        <div className="flex items-center gap-2">
                          <FileText className="w-4 h-4" />
                          JSON (Data Analysis)
                        </div>
                      </SelectItem>
                      <SelectItem value="pdf">
                        <div className="flex items-center gap-2">
                          <FileImage className="w-4 h-4" />
                          HTML Report (Printable)
                        </div>
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="space-y-2">
                <Label>Class Filter</Label>
                <Select value={filterClass} onValueChange={setFilterClass}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Classes</SelectItem>
                    {classes.map((className) => (
                      <SelectItem key={className} value={className}>
                        {className}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          {/* Field Selection */}
          <Card>
            <CardHeader>
              <CardTitle>Select Fields to Export</CardTitle>
              <CardDescription>Choose which data fields to include in your export</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {availableFields.map((field) => (
                  <div key={field.id} className="flex items-center space-x-2">
                    <Checkbox
                      id={field.id}
                      checked={selectedFields.includes(field.id)}
                      onCheckedChange={() => handleFieldToggle(field.id)}
                    />
                    <Label htmlFor={field.id} className="text-sm">
                      {field.label}
                    </Label>
                  </div>
                ))}
              </div>
              <div className="mt-4 flex items-center gap-2">
                <Badge variant="secondary">{selectedFields.length} fields selected</Badge>
                <Button variant="outline" size="sm" onClick={() => setSelectedFields(availableFields.map((f) => f.id))}>
                  Select All
                </Button>
                <Button variant="outline" size="sm" onClick={() => setSelectedFields([])}>
                  Clear All
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Preview */}
          {selectedFields.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Export Preview</CardTitle>
                <CardDescription>Preview of the first 3 records that will be exported</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse border border-gray-300">
                    <thead>
                      <tr className="bg-muted">
                        {selectedFields.map((field) => (
                          <th key={field} className="border border-gray-300 px-3 py-2 text-left text-sm font-medium">
                            {availableFields.find((f) => f.id === field)?.label || field}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {getPreviewData().map((row, index) => (
                        <tr key={index} className="hover:bg-muted/50">
                          {selectedFields.map((field) => (
                            <td key={field} className="border border-gray-300 px-3 py-2 text-sm">
                              {row[field]}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="mt-4 text-sm text-muted-foreground">
                  Total records to export: {getFilteredData().length}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Export Actions */}
          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={() => setIsOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleExport} disabled={selectedFields.length === 0} className="gap-2">
              <Download className="w-4 h-4" />
              Export {exportFormat.toUpperCase()}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
