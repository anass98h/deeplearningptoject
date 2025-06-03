"use client"

import { useState, useCallback } from "react"
import { useDropzone } from "react-dropzone"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Upload, FileVideo, CheckCircle, AlertCircle } from "lucide-react"
import { cn } from "@/lib/utils"

interface VideoUploadProps {
  onJobCreated: (jobId: string) => void
}

export function VideoUpload({ onJobCreated }: VideoUploadProps) {
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [error, setError] = useState<string | null>(null)

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (file) {
      setUploadedFile(file)
      setError(null)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "video/*": [".mp4", ".avi", ".mov", ".mkv", ".webm"],
    },
    maxFiles: 1,
    maxSize: 500 * 1024 * 1024, // 500MB
  })

  const handleUpload = async () => {
    if (!uploadedFile) return

    setUploading(true)
    setUploadProgress(0)
    setError(null)

    try {
      const formData = new FormData()
      formData.append("file", uploadedFile)

      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return prev
          }
          return prev + Math.random() * 10
        })
      }, 200)

      console.log("Uploading to: http://localhost:8000/process-video")
      const response = await fetch("http://localhost:8000/process-video", {
        method: "POST",
        body: formData,
      })

      clearInterval(progressInterval)
      setUploadProgress(100)

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`)
      }

      const result = await response.json()
      console.log("Upload successful, job created:", result)

      // According to your docs, the response should have job_id
      if (result.job_id) {
        onJobCreated(result.job_id)
        // Reset form
        setUploadedFile(null)
        setUploadProgress(0)
      } else {
        throw new Error("No job_id in response")
      }
    } catch (err) {
      console.error("Upload error:", err)
      setError(err instanceof Error ? err.message : "Upload failed")
    } finally {
      setUploading(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Upload className="h-5 w-5" />
          Upload Exercise Video
        </CardTitle>
        <CardDescription>Upload your exercise video for AI-powered form analysis and scoring</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div
          {...getRootProps()}
          className={cn(
            "border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors",
            isDragActive ? "border-primary bg-primary/5" : "border-muted-foreground/25 hover:border-primary/50",
          )}
        >
          <input {...getInputProps()} />
          <div className="space-y-4">
            <div className="mx-auto w-12 h-12 bg-primary/10 rounded-full flex items-center justify-center">
              <FileVideo className="h-6 w-6 text-primary" />
            </div>
            {isDragActive ? (
              <p className="text-lg font-medium">Drop your video here...</p>
            ) : (
              <div>
                <p className="text-lg font-medium">Drag & drop your video here</p>
                <p className="text-sm text-muted-foreground">or click to browse</p>
              </div>
            )}
            <p className="text-xs text-muted-foreground">Supports MP4, AVI, MOV, MKV, WebM (max 500MB)</p>
          </div>
        </div>

        {uploadedFile && (
          <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
            <div className="flex items-center gap-3">
              <FileVideo className="h-5 w-5 text-primary" />
              <div>
                <p className="font-medium">{uploadedFile.name}</p>
                <p className="text-sm text-muted-foreground">{(uploadedFile.size / (1024 * 1024)).toFixed(1)} MB</p>
              </div>
            </div>
            <CheckCircle className="h-5 w-5 text-green-500" />
          </div>
        )}

        {uploading && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Uploading...</span>
              <span>{Math.round(uploadProgress)}%</span>
            </div>
            <Progress value={uploadProgress} />
          </div>
        )}

        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <Button onClick={handleUpload} disabled={!uploadedFile || uploading} className="w-full" size="lg">
          {uploading ? "Processing..." : "Start Analysis"}
        </Button>
      </CardContent>
    </Card>
  )
}
