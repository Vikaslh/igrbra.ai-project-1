"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Upload, CheckCircle, AlertCircle } from "lucide-react"
import type { StudentData } from "./student-dashboard"

interface DataUploadProps {
  onDataUpload: (data: StudentData[]) => void
}

export function DataUpload({ onDataUpload }: DataUploadProps) {
  const [isUploading, setIsUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const requiredColumns = [
    "student_id",
    "name",
    "class",
    "comprehension",
    "attention",
    "focus",
    "retention",
    "assessment_score",
    "engagement_time",
  ]

  const validateCSV = (data: any[]): { isValid: boolean; error?: string } => {
    if (data.length === 0) {
      return { isValid: false, error: "CSV file is empty" }
    }

    const headers = Object.keys(data[0])
    const missingColumns = requiredColumns.filter((col) => !headers.includes(col))

    if (missingColumns.length > 0) {
      return {
        isValid: false,
        error: `Missing required columns: ${missingColumns.join(", ")}`,
      }
    }

    // Validate data types
    for (let i = 0; i < Math.min(data.length, 5); i++) {
      const row = data[i]
      const numericFields = ["comprehension", "attention", "focus", "retention", "assessment_score", "engagement_time"]

      for (const field of numericFields) {
        if (isNaN(Number(row[field]))) {
          return {
            isValid: false,
            error: `Invalid numeric value in row ${i + 1}, column ${field}`,
          }
        }
      }
    }

    return { isValid: true }
  }

  const parseCSV = (text: string): any[] => {
    const lines = text.trim().split("\n")
    const headers = lines[0].split(",").map((h) => h.trim())

    return lines.slice(1).map((line) => {
      const values = line.split(",").map((v) => v.trim())
      const row: any = {}

      headers.forEach((header, index) => {
        row[header] = values[index]
      })

      return row
    })
  }

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    if (!file.name.endsWith(".csv")) {
      setError("Please upload a CSV file")
      return
    }

    setIsUploading(true)
    setError(null)
    setSuccess(false)

    try {
      const text = await file.text()
      const parsedData = parseCSV(text)

      const validation = validateCSV(parsedData)
      if (!validation.isValid) {
        setError(validation.error!)
        setIsUploading(false)
        return
      }

      // Convert to StudentData format
      const studentData: StudentData[] = parsedData.map((row) => ({
        student_id: row.student_id,
        name: row.name,
        class: row.class,
        comprehension: Number(row.comprehension),
        attention: Number(row.attention),
        focus: Number(row.focus),
        retention: Number(row.retention),
        assessment_score: Number(row.assessment_score),
        engagement_time: Number(row.engagement_time),
      }))

      setSuccess(true)
      setTimeout(() => {
        onDataUpload(studentData)
      }, 1000)
    } catch (err) {
      setError("Failed to parse CSV file. Please check the format.")
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <Input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          onChange={handleFileUpload}
          disabled={isUploading}
          className="flex-1"
        />
        <Button onClick={() => fileInputRef.current?.click()} disabled={isUploading} className="gap-2">
          {isUploading ? (
            <>
              <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
              Processing...
            </>
          ) : (
            <>
              <Upload className="w-4 h-4" />
              Choose File
            </>
          )}
        </Button>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {success && (
        <Alert className="border-green-200 bg-green-50 text-green-800">
          <CheckCircle className="h-4 w-4" />
          <AlertDescription>CSV file uploaded successfully! Loading dashboard...</AlertDescription>
        </Alert>
      )}
    </div>
  )
}
