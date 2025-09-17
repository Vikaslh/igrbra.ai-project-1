export interface StoredDataset {
  id: string
  name: string
  uploadDate: string
  studentCount: number
  data: any[]
}

export class DataStorage {
  private static readonly DATASETS_KEY = "student_dashboard_datasets"
  private static readonly CURRENT_DATASET_KEY = "student_dashboard_current"

  static saveDataset(name: string, data: any[]): string {
    const datasets = this.getDatasets()
    const id = Date.now().toString()

    const newDataset: StoredDataset = {
      id,
      name,
      uploadDate: new Date().toISOString(),
      studentCount: data.length,
      data,
    }

    datasets.push(newDataset)
    localStorage.setItem(this.DATASETS_KEY, JSON.stringify(datasets))
    localStorage.setItem(this.CURRENT_DATASET_KEY, id)

    return id
  }

  static getDatasets(): StoredDataset[] {
    try {
      const stored = localStorage.getItem(this.DATASETS_KEY)
      return stored ? JSON.parse(stored) : []
    } catch {
      return []
    }
  }

  static getCurrentDataset(): StoredDataset | null {
    try {
      const currentId = localStorage.getItem(this.CURRENT_DATASET_KEY)
      if (!currentId) return null

      const datasets = this.getDatasets()
      return datasets.find((d) => d.id === currentId) || null
    } catch {
      return null
    }
  }

  static setCurrentDataset(id: string): void {
    localStorage.setItem(this.CURRENT_DATASET_KEY, id)
  }

  static deleteDataset(id: string): void {
    const datasets = this.getDatasets().filter((d) => d.id !== id)
    localStorage.setItem(this.DATASETS_KEY, JSON.stringify(datasets))

    const currentId = localStorage.getItem(this.CURRENT_DATASET_KEY)
    if (currentId === id) {
      localStorage.removeItem(this.CURRENT_DATASET_KEY)
    }
  }

  static exportDataset(id: string): void {
    const dataset = this.getDatasets().find((d) => d.id === id)
    if (!dataset) return

    const blob = new Blob([JSON.stringify(dataset.data, null, 2)], {
      type: "application/json",
    })

    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `${dataset.name}_${dataset.uploadDate.split("T")[0]}.json`
    a.click()
    URL.revokeObjectURL(url)
  }
}
