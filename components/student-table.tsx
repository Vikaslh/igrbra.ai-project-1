"use client"

import { useState, useMemo } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Search, ArrowUpDown, ArrowUp, ArrowDown, Users } from "lucide-react"
import { StudentProfile } from "./student-profile"
import type { StudentData } from "./student-dashboard"

interface StudentTableProps {
  data: StudentData[]
}

type SortField = keyof StudentData
type SortDirection = "asc" | "desc"

export function StudentTable({ data }: StudentTableProps) {
  const [searchTerm, setSearchTerm] = useState("")
  const [classFilter, setClassFilter] = useState<string>("all")
  const [sortField, setSortField] = useState<SortField>("name")
  const [sortDirection, setSortDirection] = useState<SortDirection>("asc")

  // Get unique classes for filter
  const classes = useMemo(() => {
    return Array.from(new Set(data.map((student) => student.class))).sort()
  }, [data])

  // Filter and sort data
  const filteredAndSortedData = useMemo(() => {
    const filtered = data.filter((student) => {
      const matchesSearch =
        student.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        student.student_id.toLowerCase().includes(searchTerm.toLowerCase()) ||
        student.class.toLowerCase().includes(searchTerm.toLowerCase())

      const matchesClass = classFilter === "all" || student.class === classFilter

      return matchesSearch && matchesClass
    })

    // Sort data
    filtered.sort((a, b) => {
      const aValue = a[sortField]
      const bValue = b[sortField]

      // Handle numeric sorting
      if (typeof aValue === "number" && typeof bValue === "number") {
        return sortDirection === "asc" ? aValue - bValue : bValue - aValue
      }

      // Handle string sorting
      const aStr = String(aValue).toLowerCase()
      const bStr = String(bValue).toLowerCase()

      if (sortDirection === "asc") {
        return aStr < bStr ? -1 : aStr > bStr ? 1 : 0
      } else {
        return aStr > bStr ? -1 : aStr < bStr ? 1 : 0
      }
    })

    return filtered
  }, [data, searchTerm, classFilter, sortField, sortDirection])

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc")
    } else {
      setSortField(field)
      setSortDirection("asc")
    }
  }

  const getSortIcon = (field: SortField) => {
    if (sortField !== field) {
      return <ArrowUpDown className="w-4 h-4 text-muted-foreground" />
    }
    return sortDirection === "asc" ? (
      <ArrowUp className="w-4 h-4 text-foreground" />
    ) : (
      <ArrowDown className="w-4 h-4 text-foreground" />
    )
  }

  const getPerformanceBadge = (score: number) => {
    if (score >= 85) return <Badge className="bg-green-100 text-green-800 hover:bg-green-100">Excellent</Badge>
    if (score >= 75) return <Badge className="bg-blue-100 text-blue-800 hover:bg-blue-100">Good</Badge>
    if (score >= 65) return <Badge className="bg-yellow-100 text-yellow-800 hover:bg-yellow-100">Average</Badge>
    return <Badge className="bg-red-100 text-red-800 hover:bg-red-100">Needs Support</Badge>
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Users className="w-5 h-5" />
              Student Directory
            </CardTitle>
            <CardDescription>
              Searchable and sortable table of all students with their cognitive skills and performance data
            </CardDescription>
          </div>
          <Badge variant="secondary" className="text-sm">
            {filteredAndSortedData.length} of {data.length} students
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        {/* Search and Filter Controls */}
        <div className="flex flex-col sm:flex-row gap-4 mb-6">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
            <Input
              placeholder="Search by name, ID, or class..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10"
            />
          </div>
          <Select value={classFilter} onValueChange={setClassFilter}>
            <SelectTrigger className="w-full sm:w-48">
              <SelectValue placeholder="Filter by class" />
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

        {/* Table */}
        <div className="rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>
                  <Button
                    variant="ghost"
                    onClick={() => handleSort("student_id")}
                    className="h-auto p-0 font-medium hover:bg-transparent"
                  >
                    Student ID {getSortIcon("student_id")}
                  </Button>
                </TableHead>
                <TableHead>
                  <Button
                    variant="ghost"
                    onClick={() => handleSort("name")}
                    className="h-auto p-0 font-medium hover:bg-transparent"
                  >
                    Name {getSortIcon("name")}
                  </Button>
                </TableHead>
                <TableHead>
                  <Button
                    variant="ghost"
                    onClick={() => handleSort("class")}
                    className="h-auto p-0 font-medium hover:bg-transparent"
                  >
                    Class {getSortIcon("class")}
                  </Button>
                </TableHead>
                <TableHead className="text-center">
                  <Button
                    variant="ghost"
                    onClick={() => handleSort("comprehension")}
                    className="h-auto p-0 font-medium hover:bg-transparent"
                  >
                    Comprehension {getSortIcon("comprehension")}
                  </Button>
                </TableHead>
                <TableHead className="text-center">
                  <Button
                    variant="ghost"
                    onClick={() => handleSort("attention")}
                    className="h-auto p-0 font-medium hover:bg-transparent"
                  >
                    Attention {getSortIcon("attention")}
                  </Button>
                </TableHead>
                <TableHead className="text-center">
                  <Button
                    variant="ghost"
                    onClick={() => handleSort("focus")}
                    className="h-auto p-0 font-medium hover:bg-transparent"
                  >
                    Focus {getSortIcon("focus")}
                  </Button>
                </TableHead>
                <TableHead className="text-center">
                  <Button
                    variant="ghost"
                    onClick={() => handleSort("retention")}
                    className="h-auto p-0 font-medium hover:bg-transparent"
                  >
                    Retention {getSortIcon("retention")}
                  </Button>
                </TableHead>
                <TableHead className="text-center">
                  <Button
                    variant="ghost"
                    onClick={() => handleSort("engagement_time")}
                    className="h-auto p-0 font-medium hover:bg-transparent"
                  >
                    Engagement {getSortIcon("engagement_time")}
                  </Button>
                </TableHead>
                <TableHead className="text-center">
                  <Button
                    variant="ghost"
                    onClick={() => handleSort("assessment_score")}
                    className="h-auto p-0 font-medium hover:bg-transparent"
                  >
                    Score {getSortIcon("assessment_score")}
                  </Button>
                </TableHead>
                <TableHead>Performance</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredAndSortedData.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={11} className="text-center py-8 text-muted-foreground">
                    No students found matching your search criteria.
                  </TableCell>
                </TableRow>
              ) : (
                filteredAndSortedData.map((student) => (
                  <TableRow key={student.student_id} className="hover:bg-muted/50">
                    <TableCell className="font-mono text-sm">{student.student_id}</TableCell>
                    <TableCell className="font-medium">{student.name}</TableCell>
                    <TableCell>
                      <Badge variant="outline">{student.class}</Badge>
                    </TableCell>
                    <TableCell className="text-center">{student.comprehension}</TableCell>
                    <TableCell className="text-center">{student.attention}</TableCell>
                    <TableCell className="text-center">{student.focus}</TableCell>
                    <TableCell className="text-center">{student.retention}</TableCell>
                    <TableCell className="text-center">{student.engagement_time}h</TableCell>
                    <TableCell className="text-center font-medium">{student.assessment_score}</TableCell>
                    <TableCell>{getPerformanceBadge(student.assessment_score)}</TableCell>
                    <TableCell>
                      <StudentProfile student={student} allStudents={data} />
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>

        {/* Summary Stats */}
        {filteredAndSortedData.length > 0 && (
          <div className="mt-4 p-4 bg-muted/50 rounded-lg">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <p className="text-muted-foreground">Filtered Students</p>
                <p className="text-lg font-semibold">{filteredAndSortedData.length}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Average Score</p>
                <p className="text-lg font-semibold">
                  {(
                    filteredAndSortedData.reduce((sum, s) => sum + s.assessment_score, 0) / filteredAndSortedData.length
                  ).toFixed(1)}
                </p>
              </div>
              <div>
                <p className="text-muted-foreground">Top Performer</p>
                <p className="text-lg font-semibold">
                  {Math.max(...filteredAndSortedData.map((s) => s.assessment_score)).toFixed(1)}
                </p>
              </div>
              <div>
                <p className="text-muted-foreground">Classes Shown</p>
                <p className="text-lg font-semibold">{new Set(filteredAndSortedData.map((s) => s.class)).size}</p>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
