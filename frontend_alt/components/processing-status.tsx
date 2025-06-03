"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Activity, CheckCircle, XCircle, Clock, AlertTriangle, RefreshCw, TrendingUp } from "lucide-react"

interface ProcessingStatusProps {
  jobId: string | null
  onJobCompleted: () => void
}

interface JobStatus {
  id: string
  filename: string
  status: "pending" | "processing" | "completed" | "failed"
  message: string
  created_at: number
  updated_at: number
  quality_scores?: {
    ugly_2d_goodness?: number
    ugly_2d_confidence?: number
    bad_3d_exercise_score?: number
    final_exercise_score?: number
    score_interpretation?: string
    advanced_analysis?: any
  }
}

export function ProcessingStatus({ jobId, onJobCompleted }: ProcessingStatusProps) {
  const [status, setStatus] = useState<JobStatus | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!jobId) return

    const pollStatus = async () => {
      try {
        console.log(`Polling status for job: ${jobId}`)
        const response = await fetch(`http://localhost:8000/job-status/${jobId}`)

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`)
        }

        const data = await response.json()
        console.log("Status received:", data)
        setStatus(data)
        setError(null)

        // Only call onJobCompleted when status is actually "completed"
        if (data.status === "completed") {
          console.log("Job completed, switching to results view")
          onJobCompleted()
        } else if (data.status === "failed") {
          console.log("Job failed:", data.message)
          // Don't call onJobCompleted for failed jobs
        }
      } catch (error) {
        console.error("Failed to fetch status:", error)
        setError(error instanceof Error ? error.message : "Failed to fetch job status")
      }
    }

    // Poll immediately, then every 2 seconds
    pollStatus()
    const interval = setInterval(pollStatus, 2000)

    return () => clearInterval(interval)
  }, [jobId, onJobCompleted])

  if (!jobId) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-center space-y-2">
            <Clock className="h-12 w-12 text-muted-foreground mx-auto" />
            <p className="text-lg font-medium">No active job</p>
            <p className="text-sm text-muted-foreground">Upload a video to start processing</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "pending":
        return <Clock className="h-5 w-5 text-yellow-500" />
      case "processing":
        return <Activity className="h-5 w-5 text-blue-500 animate-spin" />
      case "completed":
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case "failed":
        return <XCircle className="h-5 w-5 text-red-500" />
      default:
        return <Clock className="h-5 w-5 text-gray-500" />
    }
  }

  const getStatusBadge = (status: string) => {
    const variants = {
      pending: "secondary",
      processing: "default",
      completed: "default",
      failed: "destructive",
    } as const

    return (
      <Badge variant={variants[status as keyof typeof variants] || "secondary"}>
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </Badge>
    )
  }

  const getProgress = () => {
    if (!status) return 0
    switch (status.status) {
      case "pending":
        return 10
      case "processing":
        return 50
      case "completed":
        return 100
      case "failed":
        return 0
      default:
        return 0
    }
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {getStatusIcon(status?.status || "pending")}
              <div>
                <CardTitle>Processing Status</CardTitle>
                <CardDescription>Job ID: {jobId}</CardDescription>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {getStatusBadge(status?.status || "pending")}
              <Button variant="outline" size="sm" onClick={() => window.location.reload()}>
                <RefreshCw className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          {status?.filename && (
            <div>
              <p className="text-sm font-medium">File: {status.filename}</p>
              <p className="text-xs text-muted-foreground">
                Started: {new Date(status.created_at * 1000).toLocaleString()}
              </p>
            </div>
          )}

          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Progress</span>
              <span>{getProgress()}%</span>
            </div>
            <Progress value={getProgress()} />
          </div>

          {status?.message && (
            <Alert variant={status.status === "failed" ? "destructive" : "default"}>
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>{status.message}</AlertDescription>
            </Alert>
          )}

          {error && (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Only show quality scores if they exist and are not null/undefined */}
      {status?.quality_scores && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Quality Metrics
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4">
              {typeof status.quality_scores.ugly_2d_goodness === "number" && (
                <div>
                  <p className="text-sm font-medium">2D Goodness</p>
                  <p className="text-2xl font-bold">{status.quality_scores.ugly_2d_goodness.toFixed(3)}</p>
                </div>
              )}
              {typeof status.quality_scores.ugly_2d_confidence === "number" && (
                <div>
                  <p className="text-sm font-medium">2D Confidence</p>
                  <p className="text-2xl font-bold">{status.quality_scores.ugly_2d_confidence.toFixed(3)}</p>
                </div>
              )}
              {typeof status.quality_scores.final_exercise_score === "number" && (
                <div>
                  <p className="text-sm font-medium">Exercise Score</p>
                  <p className="text-2xl font-bold">{status.quality_scores.final_exercise_score.toFixed(1)}/4.0</p>
                </div>
              )}
              {status.quality_scores.score_interpretation && (
                <div>
                  <p className="text-sm font-medium">Interpretation</p>
                  <p className="text-lg font-semibold text-green-600">{status.quality_scores.score_interpretation}</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Show pipeline stages */}
      <Card>
        <CardHeader>
          <CardTitle>Pipeline Stages</CardTitle>
          <CardDescription>Based on your documentation</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[
              "MoveNet pose extraction",
              "2D quality assessment (Ugly 2D Check)",
              "Kinect format conversion",
              "3D depth prediction",
              "Exercise segmentation (Frame Trimming)",
              "3D form quality check (Bad 3D Check)",
              "Final exercise scoring",
            ].map((stage, index) => {
              const isActive = status?.status === "processing"
              const isCompleted = status?.status === "completed"
              const isFailed = status?.status === "failed"

              return (
                <div key={index} className="flex items-center gap-3 text-sm">
                  <div
                    className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium ${
                      isCompleted
                        ? "bg-green-500 text-white"
                        : isActive
                          ? "bg-blue-500 text-white"
                          : isFailed
                            ? "bg-red-500 text-white"
                            : "bg-muted text-muted-foreground"
                    }`}
                  >
                    {isCompleted ? (
                      <CheckCircle className="h-3 w-3" />
                    ) : isFailed ? (
                      <XCircle className="h-3 w-3" />
                    ) : (
                      index + 1
                    )}
                  </div>
                  <span className={isActive ? "text-blue-600 font-medium" : ""}>{stage}</span>
                  {isActive && <Activity className="h-3 w-3 text-blue-500 animate-spin ml-auto" />}
                </div>
              )
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
