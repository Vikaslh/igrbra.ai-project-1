"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Database, Download, Trash2, Calendar, Users, Plus, FileText } from "lucide-react"
import { DataStorage, type StoredDataset } from "@/lib/storage"
import type { StudentData } from "./student-dashboard"

interface DatasetManagerProps {
  onDatasetSelect: (data: StudentData[]) => void
  currentDatasetId?: string
}

export function DatasetManager({ onDatasetSelect, currentDatasetId }: DatasetManagerProps) {
  const [datasets, setDatasets] = useState<StoredDataset[]>([])
  const [isOpen, setIsOpen] = useState(false)
  const [newDatasetName, setNewDatasetName] = useState("")

  useEffect(() => {
    loadDatasets()
  }, [])

  const loadDatasets = () => {
    setDatasets(DataStorage.getDatasets())
  }

  const handleSelectDataset = (dataset: StoredDataset) => {
    DataStorage.setCurrentDataset(dataset.id)
    onDatasetSelect(dataset.data)
    setIsOpen(false)
  }

  const handleDeleteDataset = (id: string) => {
    DataStorage.deleteDataset(id)
    loadDatasets()
  }

  const handleExportDataset = (id: string) => {
    DataStorage.exportDataset(id)
  }

  const saveCurrentAsNew = () => {
    if (!newDatasetName.trim()) return

    const current = DataStorage.getCurrentDataset()
    if (current) {
      DataStorage.saveDataset(newDatasetName, current.data)
      loadDatasets()
      setNewDatasetName("")
    }
  }

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" className="gap-2 bg-transparent">
          <Database className="w-4 h-4" />
          Manage Datasets
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Database className="w-5 h-5" />
            Dataset Manager
          </DialogTitle>
          <DialogDescription>
            Manage your saved datasets, switch between different data sets, and export data
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6">
          {/* Save Current Dataset */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Save Current Dataset</CardTitle>
              <CardDescription>Save the currently loaded data as a new dataset</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex gap-2">
                <Input
                  placeholder="Enter dataset name..."
                  value={newDatasetName}
                  onChange={(e) => setNewDatasetName(e.target.value)}
                />
                <Button onClick={saveCurrentAsNew} disabled={!newDatasetName.trim()}>
                  <Plus className="w-4 h-4 mr-2" />
                  Save
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Saved Datasets */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Saved Datasets ({datasets.length})</h3>

            {datasets.length === 0 ? (
              <Alert>
                <FileText className="h-4 w-4" />
                <AlertDescription>
                  No saved datasets found. Upload and save your first dataset to get started.
                </AlertDescription>
              </Alert>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {datasets.map((dataset) => (
                  <Card
                    key={dataset.id}
                    className={`cursor-pointer transition-colors ${
                      currentDatasetId === dataset.id ? "ring-2 ring-primary" : "hover:bg-muted/50"
                    }`}
                  >
                    <CardHeader className="pb-3">
                      <div className="flex items-start justify-between">
                        <div>
                          <CardTitle className="text-base">{dataset.name}</CardTitle>
                          <div className="flex items-center gap-4 text-sm text-muted-foreground mt-1">
                            <div className="flex items-center gap-1">
                              <Calendar className="w-3 h-3" />
                              {new Date(dataset.uploadDate).toLocaleDateString()}
                            </div>
                            <div className="flex items-center gap-1">
                              <Users className="w-3 h-3" />
                              {dataset.studentCount} students
                            </div>
                          </div>
                        </div>
                        {currentDatasetId === dataset.id && <Badge variant="secondary">Current</Badge>}
                      </div>
                    </CardHeader>
                    <CardContent className="pt-0">
                      <div className="flex gap-2">
                        <Button
                          size="sm"
                          onClick={() => handleSelectDataset(dataset)}
                          disabled={currentDatasetId === dataset.id}
                        >
                          Load Dataset
                        </Button>
                        <Button size="sm" variant="outline" onClick={() => handleExportDataset(dataset.id)}>
                          <Download className="w-3 h-3" />
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleDeleteDataset(dataset.id)}
                          className="text-destructive hover:text-destructive"
                        >
                          <Trash2 className="w-3 h-3" />
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
