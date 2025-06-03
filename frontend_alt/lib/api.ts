// API configuration and helper functions
const API_BASE_URL = "https://102e-83-219-194-105.ngrok-free.app"

const defaultHeaders = {
  "ngrok-skip-browser-warning": "true",
}

export const api = {
  async uploadVideo(file: File) {
    const formData = new FormData()
    formData.append("file", file)

    const response = await fetch(`${API_BASE_URL}/process-video`, {
      method: "POST",
      body: formData,
      headers: defaultHeaders,
    })

    if (!response.ok) {
      const errorData = await response.json()
      throw new Error(errorData.detail || "Upload failed")
    }

    return response.json()
  },

  async getJobStatus(jobId: string) {
    const response = await fetch(`${API_BASE_URL}/job-status/${jobId}`, {
      headers: defaultHeaders,
    })

    if (!response.ok) {
      throw new Error("Failed to fetch job status")
    }

    return response.json()
  },

  async getFinalResults(jobId: string) {
    const response = await fetch(`${API_BASE_URL}/video-data/${jobId}/final`, {
      headers: defaultHeaders,
    })

    if (!response.ok) {
      throw new Error("Failed to fetch results")
    }

    return response.json()
  },

  async getSkeletonData(jobId: string, type: "original" | "kinect2d" | "kinect3d" | "trimmed" | "untrimmed") {
    const endpoint = type === "untrimmed" ? "kinect3d" : type
    const response = await fetch(`${API_BASE_URL}/video-data/${jobId}/${endpoint}`, {
      headers: defaultHeaders,
    })

    if (!response.ok) {
      throw new Error(`Failed to fetch ${type} data`)
    }

    return response.text() // Returns CSV data
  },

  async downloadSkeletonData(jobId: string, type: "trimmed" | "untrimmed", filename: string) {
    const csvData = await this.getSkeletonData(jobId, type)
    const blob = new Blob([csvData], { type: "text/csv" })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `${filename}_${type}_skeleton.csv`
    document.body.appendChild(a)
    a.click()
    window.URL.revokeObjectURL(url)
    document.body.removeChild(a)
  },
}
