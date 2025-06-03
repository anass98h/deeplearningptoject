"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { BarChart3, TrendingUp, TrendingDown, Activity, Users, Clock, CheckCircle, XCircle } from "lucide-react"
import { useState, useEffect } from "react"
import { Alert, AlertDescription } from "@/components/ui/alert"

export function Analytics() {
  // Add this at the top of the Analytics component
  const [analyticsData, setAnalyticsData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        // If you add an analytics endpoint later, uncomment this:
        // const response = await fetch('https://2164-83-219-194-105.ngrok-free.app/analytics', {
        //   headers: {
        //     'ngrok-skip-browser-warning': 'true'
        //   }
        // })
        // const data = await response.json()
        // setAnalyticsData(data)

        // For now, keep the mock data but add a note
        setLoading(false)
      } catch (error) {
        console.error("Failed to fetch analytics:", error)
        setLoading(false)
      }
    }

    fetchAnalytics()
  }, [])

  // Mock data - in real implementation, this would come from your API
  const mockStats = {
    totalJobs: 1247,
    successRate: 87.3,
    avgProcessingTime: 45.2,
    activeJobs: 12,
    recentJobs: [
      { id: "job_001", filename: "squat_video.mp4", status: "completed", score: 0.8, timestamp: "2 min ago" },
      { id: "job_002", filename: "pushup_form.avi", status: "failed", score: null, timestamp: "5 min ago" },
      { id: "job_003", filename: "deadlift_check.mov", status: "completed", score: 1.2, timestamp: "8 min ago" },
      { id: "job_004", filename: "plank_analysis.mp4", status: "processing", score: null, timestamp: "12 min ago" },
      { id: "job_005", filename: "burpee_form.webm", status: "completed", score: 2.1, timestamp: "15 min ago" },
    ],
    qualityDistribution: [
      { range: "0.0-0.5", label: "Excellent", count: 234, percentage: 18.8 },
      { range: "0.5-1.0", label: "Very Good", count: 312, percentage: 25.0 },
      { range: "1.0-1.5", label: "Good", count: 298, percentage: 23.9 },
      { range: "1.5-2.0", label: "Fair", count: 187, percentage: 15.0 },
      { range: "2.0-2.5", label: "Poor", count: 134, percentage: 10.7 },
      { range: "2.5+", label: "Very Poor", count: 82, percentage: 6.6 },
    ],
    commonFailures: [
      { reason: "Low video quality", count: 89, percentage: 56.3 },
      { reason: "Poor lighting", count: 34, percentage: 21.5 },
      { reason: "Camera instability", count: 23, percentage: 14.6 },
      { reason: "Incomplete exercise", count: 12, percentage: 7.6 },
    ],
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case "failed":
        return <XCircle className="h-4 w-4 text-red-500" />
      case "processing":
        return <Activity className="h-4 w-4 text-blue-500 animate-spin" />
      default:
        return <Clock className="h-4 w-4 text-gray-500" />
    }
  }

  const getScoreColor = (score: number | null) => {
    if (score === null) return "text-muted-foreground"
    if (score <= 1.0) return "text-green-600"
    if (score <= 2.0) return "text-yellow-600"
    return "text-red-600"
  }

  return (
    <div className="space-y-6">
      {/* Add this note at the top of the analytics section: */}
      {!analyticsData && (
        <Alert>
          <BarChart3 className="h-4 w-4" />
          <AlertDescription>
            Analytics data shown below is simulated. Connect to your backend analytics endpoint for real data.
          </AlertDescription>
        </Alert>
      )}
      {/* Overview Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Jobs</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{mockStats.totalJobs.toLocaleString()}</div>
            <p className="text-xs text-muted-foreground">
              <TrendingUp className="inline h-3 w-3 mr-1" />
              +12% from last month
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
            <CheckCircle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{mockStats.successRate}%</div>
            <Progress value={mockStats.successRate} className="mt-2" />
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Processing</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{mockStats.avgProcessingTime}s</div>
            <p className="text-xs text-muted-foreground">
              <TrendingDown className="inline h-3 w-3 mr-1" />
              -8% improvement
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Jobs</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{mockStats.activeJobs}</div>
            <p className="text-xs text-muted-foreground">Currently processing</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Jobs */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Jobs</CardTitle>
            <CardDescription>Latest video processing activities</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {mockStats.recentJobs.map((job) => (
                <div key={job.id} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center gap-3">
                    {getStatusIcon(job.status)}
                    <div>
                      <p className="font-medium text-sm">{job.filename}</p>
                      <p className="text-xs text-muted-foreground">{job.timestamp}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    {job.score !== null ? (
                      <div className={`font-bold ${getScoreColor(job.score)}`}>{job.score.toFixed(1)}/4.0</div>
                    ) : (
                      <Badge variant="secondary">{job.status}</Badge>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Quality Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Score Distribution</CardTitle>
            <CardDescription>Exercise quality score breakdown</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {mockStats.qualityDistribution.map((item) => (
                <div key={item.range} className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="font-medium">
                      {item.range} - {item.label}
                    </span>
                    <span>
                      {item.count} ({item.percentage}%)
                    </span>
                  </div>
                  <Progress value={item.percentage} />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Failure Analysis */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Common Failure Reasons
          </CardTitle>
          <CardDescription>Analysis of why videos fail quality checks</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              {mockStats.commonFailures.map((failure, index) => (
                <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <p className="font-medium">{failure.reason}</p>
                    <p className="text-sm text-muted-foreground">{failure.count} occurrences</p>
                  </div>
                  <div className="text-right">
                    <div className="font-bold">{failure.percentage}%</div>
                    <Progress value={failure.percentage} className="w-20 mt-1" />
                  </div>
                </div>
              ))}
            </div>
            <div className="space-y-4">
              <h4 className="font-medium">Recommendations</h4>
              <div className="space-y-2 text-sm">
                <div className="p-3 bg-blue-50 dark:bg-blue-950 rounded-lg">
                  <p className="font-medium text-blue-900 dark:text-blue-100">Improve Video Quality</p>
                  <p className="text-blue-700 dark:text-blue-300">Use better lighting and stable camera setup</p>
                </div>
                <div className="p-3 bg-green-50 dark:bg-green-950 rounded-lg">
                  <p className="font-medium text-green-900 dark:text-green-100">Complete Exercises</p>
                  <p className="text-green-700 dark:text-green-300">Ensure full exercise movement is captured</p>
                </div>
                <div className="p-3 bg-purple-50 dark:bg-purple-950 rounded-lg">
                  <p className="font-medium text-purple-900 dark:text-purple-100">Camera Positioning</p>
                  <p className="text-purple-700 dark:text-purple-300">Position camera to capture full body movement</p>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Pipeline Performance */}
      <Card>
        <CardHeader>
          <CardTitle>Pipeline Performance</CardTitle>
          <CardDescription>Processing stage success rates and timing</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              { stage: "MoveNet Extraction", success: 98.5, avgTime: 8.2 },
              { stage: "Quality Check", success: 87.3, avgTime: 2.1 },
              { stage: "3D Conversion", success: 95.8, avgTime: 12.4 },
              { stage: "Exercise Scoring", success: 92.1, avgTime: 15.8 },
            ].map((stage, index) => (
              <div key={index} className="p-4 border rounded-lg space-y-2">
                <h4 className="font-medium text-sm">{stage.stage}</h4>
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span>Success Rate</span>
                    <span>{stage.success}%</span>
                  </div>
                  <Progress value={stage.success} />
                </div>
                <p className="text-xs text-muted-foreground">Avg: {stage.avgTime}s</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
