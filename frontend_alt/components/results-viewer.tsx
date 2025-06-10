"use client";

import { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import {
  Trophy,
  AlertTriangle,
  Download,
  BarChart3,
  Activity,
  FileText,
  Zap,
  Clock,
  Target,
  TrendingUp,
  CheckCircle,
  XCircle,
  Info,
  Eye,
  Camera,
} from "lucide-react";

interface ResultsViewerProps {
  jobId: string | null;
}

interface JobResult {
  job_id: string;
  filename: string;
  status: string;
  message: string;
  created_at: number;
  updated_at: number;
  quality_scores: {
    ugly_2d_goodness: number;
    ugly_2d_confidence: number;
    bad_3d_exercise_score?: number;
    final_exercise_score?: number;
    score_interpretation?: string;
    advanced_analysis?: {
      summary: {
        total_frames: number;
        windows_analyzed: number;
        has_confidence_data: boolean;
        confidence_stats: { min: number; max: number; avg: number };
        motion_stats: { min: number; max: number; avg: number };
      };
      worst_confidence_windows?: Array<{
        window: number;
        frames: string;
        time: string;
        worst_block: string;
        confidence: number;
      }>;
      most_jittery_windows?: Array<{
        window: number;
        frames: string;
        time: string;
        worst_block: string;
        displacement: number;
      }>;
    };
  };
  skeleton_data?: {
    trimmed: string;
    untrimmed: string;
  };
  data_formats?: {
    trimmed: string;
    untrimmed: string;
  };
}

export function ResultsViewer({ jobId }: ResultsViewerProps) {
  const [results, setResults] = useState<JobResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Mock data for demonstration - replace with actual API call
  useEffect(() => {
    if (jobId) {
      setLoading(true);
      setTimeout(() => {
        setResults({
          job_id: "f49f51fd-9845-4e82-b759-9e2a203c1f85",
          filename: "A6.avi",
          status: "completed",
          message:
            "Processing completed successfully - Score: 1.6/5.0 (Fair form)",
          created_at: 1748975328.0178251,
          updated_at: 1748975350.044874,
          quality_scores: {
            ugly_2d_goodness: 4.24875545501709,
            ugly_2d_confidence: 0.5165290832519531,
            bad_3d_exercise_score: 0.9992509484291077,
            final_exercise_score: 1.5668268203735352,
            score_interpretation: "Fair form",
            advanced_analysis: {
              summary: {
                total_frames: 225,
                windows_analyzed: 45,
                has_confidence_data: true,
                confidence_stats: {
                  min: 0.343,
                  max: 0.631,
                  avg: 0.473,
                },
                motion_stats: {
                  min: 28.2,
                  max: 485.7,
                  avg: 131.9,
                },
              },
              worst_confidence_windows: [
                {
                  window: 25,
                  frames: "126-130",
                  time: "00:00:08-00:00:09",
                  worst_block: "top_left",
                  confidence: 0.343,
                },
                {
                  window: 24,
                  frames: "121-125",
                  time: "00:00:08-00:00:08",
                  worst_block: "top_left",
                  confidence: 0.376,
                },
              ],
              most_jittery_windows: [
                {
                  window: 41,
                  frames: "206-210",
                  time: "00:00:14-00:00:14",
                  worst_block: "top_left",
                  displacement: 485.7,
                },
                {
                  window: 44,
                  frames: "221-225",
                  time: "00:00:15-00:00:15",
                  worst_block: "top_left",
                  displacement: 345.6,
                },
              ],
            },
          },
          skeleton_data: {
            trimmed: "kinect_3d_data",
            untrimmed: "full_sequence_data",
          },
          data_formats: {
            trimmed: "kinect_3d_trimmed_exercise_segment",
            untrimmed: "kinect_3d_full_sequence",
          },
        });
        setLoading(false);
      }, 1000);
    }
  }, [jobId]);

  if (!jobId) {
    return (
      <Card className="border-dashed">
        <CardContent className="flex items-center justify-center py-16">
          <div className="text-center space-y-4">
            <div className="relative">
              <FileText className="h-16 w-16 text-muted-foreground/50 mx-auto" />
              <div className="absolute inset-0 bg-gradient-to-t from-background to-transparent" />
            </div>
            <div className="space-y-2">
              <h3 className="text-xl font-semibold">No Analysis Results</h3>
              <p className="text-muted-foreground max-w-sm">
                Upload and process a video to view detailed exercise form
                analysis and recommendations
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (loading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-16">
          <div className="text-center space-y-4">
            <div className="relative">
              <Activity className="h-12 w-12 text-primary animate-pulse mx-auto" />
              <div className="absolute -inset-2 bg-primary/20 rounded-full animate-ping" />
            </div>
            <div className="space-y-2">
              <h3 className="text-lg font-semibold">Analyzing Results</h3>
              <p className="text-muted-foreground">
                Processing exercise data...
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive" className="border-red-200 bg-red-50">
        <XCircle className="h-4 w-4" />
        <AlertDescription className="font-medium">{error}</AlertDescription>
      </Alert>
    );
  }

  if (!results) return null;

  const getScoreColor = (score: number) => {
    if (score <= 1.0) return "text-emerald-600";
    if (score <= 1.5) return "text-green-600";
    if (score <= 2.0) return "text-yellow-600";
    if (score <= 2.5) return "text-orange-500";
    return "text-red-500";
  };

  const getScoreBadgeVariant = (
    score: number
  ): "default" | "secondary" | "destructive" => {
    if (score <= 1.5) return "default";
    if (score <= 2.0) return "secondary";
    return "destructive";
  };

  const getScoreGrade = (score: number) => {
    if (score <= 0.5) return "A+";
    if (score <= 1.0) return "A";
    if (score <= 1.5) return "B";
    if (score <= 2.0) return "C";
    if (score <= 2.5) return "D";
    return "F";
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const processingDuration = results.updated_at - results.created_at;

  return (
    <div className="space-y-8">
      {/* Enhanced Header */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/20 dark:to-indigo-950/20" />
        <Card className="relative border-none shadow-lg">
          <CardHeader className="pb-6">
            <div className="flex items-start justify-between">
              <div className="flex items-start gap-4">
                <div className="p-3 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl">
                  <Trophy className="h-8 w-8 text-white" />
                </div>
                <div className="space-y-2">
                  <div className="flex items-center gap-3">
                    <CardTitle className="text-2xl">
                      Exercise Analysis Report
                    </CardTitle>
                    <Badge
                      variant={
                        results.status === "completed"
                          ? "default"
                          : "destructive"
                      }
                      className="px-3 py-1"
                    >
                      <CheckCircle className="h-3 w-3 mr-1" />
                      {results.status.toUpperCase()}
                    </Badge>
                  </div>
                  <CardDescription className="text-base">
                    <span className="font-medium">{results.filename}</span> •
                    Processed {formatDuration(Math.round(processingDuration))}{" "}
                    ago
                  </CardDescription>
                </div>
              </div>

              {/* Quick Score Display */}
              {typeof results.quality_scores.final_exercise_score ===
                "number" && (
                <div className="text-right space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-muted-foreground">
                      Overall Grade
                    </span>
                    <div
                      className={`text-3xl font-bold ${getScoreColor(
                        results.quality_scores.final_exercise_score
                      )}`}
                    >
                      {getScoreGrade(
                        results.quality_scores.final_exercise_score
                      )}
                    </div>
                  </div>
                  <div
                    className={`text-sm ${getScoreColor(
                      results.quality_scores.final_exercise_score
                    )}`}
                  >
                    {results.quality_scores.final_exercise_score.toFixed(1)}/5.0
                    • {results.quality_scores.score_interpretation}
                  </div>
                </div>
              )}
            </div>
          </CardHeader>

          <CardContent>
            <Alert className="border-blue-200 bg-blue-50 dark:bg-blue-950/20">
              <Info className="h-4 w-4 text-blue-600" />
              <AlertDescription className="text-blue-800 dark:text-blue-200">
                {results.message}
              </AlertDescription>
            </Alert>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList className="grid w-full grid-cols-3 h-12">
          <TabsTrigger
            value="overview"
            className="data-[state=active]:bg-blue-100"
          >
            <Target className="h-4 w-4 mr-2" />
            Overview
          </TabsTrigger>
          <TabsTrigger
            value="analysis"
            className="data-[state=active]:bg-purple-100"
          >
            <BarChart3 className="h-4 w-4 mr-2" />
            Analysis
          </TabsTrigger>
          <TabsTrigger
            value="diagnostics"
            className="data-[state=active]:bg-orange-100"
          >
            <Zap className="h-4 w-4 mr-2" />
            Diagnostics
          </TabsTrigger>
          {/* <TabsTrigger
            value="data"
            className="data-[state=active]:bg-green-100"
          >
            <FileText className="h-4 w-4 mr-2" />
            Data Export
          </TabsTrigger> */}
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {/* Key Metrics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Card className="border-l-4 border-l-blue-500">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium text-muted-foreground">
                    Final Score
                  </CardTitle>
                  <Target className="h-4 w-4 text-blue-500" />
                </div>
              </CardHeader>
              <CardContent>
                {typeof results.quality_scores.final_exercise_score ===
                "number" ? (
                  <div className="space-y-2">
                    <div
                      className={`text-3xl font-bold ${getScoreColor(
                        results.quality_scores.final_exercise_score
                      )}`}
                    >
                      {results.quality_scores.final_exercise_score.toFixed(1)}
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-muted-foreground">
                        out of 5.0
                      </span>
                      <Badge
                        variant={getScoreBadgeVariant(
                          results.quality_scores.final_exercise_score
                        )}
                        className="text-xs"
                      >
                        {results.quality_scores.score_interpretation}
                      </Badge>
                    </div>
                  </div>
                ) : (
                  <div className="text-3xl font-bold text-muted-foreground">
                    N/A
                  </div>
                )}
              </CardContent>
            </Card>

            <Card className="border-l-4 border-l-green-500">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium text-muted-foreground">
                    Pose Detection
                  </CardTitle>
                  <Eye className="h-4 w-4 text-green-500" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="text-2xl font-bold">
                    {(results.quality_scores.ugly_2d_goodness * 20).toFixed(0)}%
                  </div>
                  <Progress
                    value={Math.min(
                      results.quality_scores.ugly_2d_goodness * 20,
                      100
                    )}
                    className="h-2"
                  />
                  <span className="text-xs text-muted-foreground">
                    Quality Index
                  </span>
                </div>
              </CardContent>
            </Card>

            <Card className="border-l-4 border-l-purple-500">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium text-muted-foreground">
                    Tracking Confidence
                  </CardTitle>
                  <Camera className="h-4 w-4 text-purple-500" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="text-2xl font-bold">
                    {(results.quality_scores.ugly_2d_confidence * 100).toFixed(
                      0
                    )}
                    %
                  </div>
                  <Progress
                    value={results.quality_scores.ugly_2d_confidence * 100}
                    className="h-2"
                  />
                  <span className="text-xs text-muted-foreground">
                    Pose Stability
                  </span>
                </div>
              </CardContent>
            </Card>

            <Card className="border-l-4 border-l-orange-500">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium text-muted-foreground">
                    Processing Time
                  </CardTitle>
                  <Clock className="h-4 w-4 text-orange-500" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="text-2xl font-bold">
                    {formatDuration(Math.round(processingDuration))}
                  </div>
                  <span className="text-xs text-muted-foreground">
                    Analysis Duration
                  </span>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Enhanced Score Interpretation */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5 text-blue-500" />
                Performance Grading System
              </CardTitle>
              <CardDescription>
                Understanding your exercise form assessment scores
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <div className="space-y-4">
                  <h4 className="font-semibold text-sm text-muted-foreground uppercase tracking-wide">
                    Excellent
                  </h4>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between p-3 bg-emerald-50 rounded-lg border border-emerald-200">
                      <div>
                        <span className="font-bold text-emerald-700">
                          A+ Grade
                        </span>
                        <p className="text-xs text-emerald-600">
                          0.0 - 0.5 points
                        </p>
                      </div>
                      <Badge className="bg-emerald-100 text-emerald-700 border-emerald-300">
                        Perfect
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg border border-green-200">
                      <div>
                        <span className="font-bold text-green-700">
                          A Grade
                        </span>
                        <p className="text-xs text-green-600">
                          0.5 - 1.0 points
                        </p>
                      </div>
                      <Badge className="bg-green-100 text-green-700 border-green-300">
                        Excellent
                      </Badge>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <h4 className="font-semibold text-sm text-muted-foreground uppercase tracking-wide">
                    Good
                  </h4>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg border border-blue-200">
                      <div>
                        <span className="font-bold text-blue-700">B Grade</span>
                        <p className="text-xs text-blue-600">
                          1.0 - 1.5 points
                        </p>
                      </div>
                      <Badge className="bg-blue-100 text-blue-700 border-blue-300">
                        Good
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-yellow-50 rounded-lg border border-yellow-200">
                      <div>
                        <span className="font-bold text-yellow-700">
                          C Grade
                        </span>
                        <p className="text-xs text-yellow-600">
                          1.5 - 2.0 points
                        </p>
                      </div>
                      <Badge className="bg-yellow-100 text-yellow-700 border-yellow-300">
                        Fair
                      </Badge>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <h4 className="font-semibold text-sm text-muted-foreground uppercase tracking-wide">
                    Needs Improvement
                  </h4>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between p-3 bg-orange-50 rounded-lg border border-orange-200">
                      <div>
                        <span className="font-bold text-orange-700">
                          D Grade
                        </span>
                        <p className="text-xs text-orange-600">
                          2.0 - 2.5 points
                        </p>
                      </div>
                      <Badge className="bg-orange-100 text-orange-700 border-orange-300">
                        Poor
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-red-50 rounded-lg border border-red-200">
                      <div>
                        <span className="font-bold text-red-700">F Grade</span>
                        <p className="text-xs text-red-600">2.5+ points</p>
                      </div>
                      <Badge className="bg-red-100 text-red-700 border-red-300">
                        Critical
                      </Badge>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analysis" className="space-y-6">
          {results.quality_scores.advanced_analysis ? (
            <>
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5 text-purple-500" />
                    Video Analysis Summary
                  </CardTitle>
                  <CardDescription>
                    Comprehensive breakdown of your exercise session
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
                    <div className="text-center space-y-2">
                      <div className="text-3xl font-bold text-purple-600">
                        {
                          results.quality_scores.advanced_analysis.summary
                            .total_frames
                        }
                      </div>
                      <p className="text-sm font-medium">Total Frames</p>
                      <p className="text-xs text-muted-foreground">
                        Video Length
                      </p>
                    </div>
                    <div className="text-center space-y-2">
                      <div className="text-3xl font-bold text-blue-600">
                        {
                          results.quality_scores.advanced_analysis.summary
                            .windows_analyzed
                        }
                      </div>
                      <p className="text-sm font-medium">Analysis Windows</p>
                      <p className="text-xs text-muted-foreground">
                        Segments Processed
                      </p>
                    </div>
                    {/* <div className="text-center space-y-2">
                      <div className="text-3xl font-bold text-green-600">
                        {(
                          results.quality_scores.advanced_analysis.summary
                            .confidence_stats.avg * 100
                        ).toFixed(0)}
                        %
                      </div>
                      <p className="text-sm font-medium">Avg Confidence</p>
                      <p className="text-xs text-muted-foreground">
                        Tracking Quality
                      </p>
                    </div>
                    <div className="text-center space-y-2">
                      <div className="text-3xl font-bold text-orange-600">
                        {results.quality_scores.advanced_analysis.summary.motion_stats.avg.toFixed(
                          0
                        )}
                      </div>
                      <p className="text-sm font-medium">Motion Index</p>
                      <p className="text-xs text-muted-foreground">
                        Movement Stability
                      </p>
                    </div> */}
                    <div className="text-center space-y-2">
                      <div className="text-3xl font-bold text-indigo-600">
                        {results.quality_scores.advanced_analysis.summary
                          .has_confidence_data
                          ? "✓"
                          : "✗"}
                      </div>
                      <p className="text-sm font-medium">Data Quality</p>
                      <p className="text-xs text-muted-foreground">
                        Confidence Available
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </>
          ) : (
            <Card>
              <CardContent className="flex items-center justify-center py-16">
                <div className="text-center space-y-4">
                  <BarChart3 className="h-16 w-16 text-muted-foreground/50 mx-auto" />
                  <div className="space-y-2">
                    <h3 className="text-xl font-semibold">
                      Analysis Unavailable
                    </h3>
                    <p className="text-muted-foreground max-w-sm">
                      Advanced motion analysis could not be performed on this
                      video
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="diagnostics" className="space-y-6">
          {results.quality_scores.advanced_analysis
            ?.worst_confidence_windows && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5 text-yellow-500" />
                  Tracking Quality Issues
                </CardTitle>
                <CardDescription>
                  Areas where pose detection confidence was lowest
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {results.quality_scores.advanced_analysis.worst_confidence_windows
                    .slice(0, 5)
                    .map((window, index) => (
                      <div
                        key={index}
                        className="flex items-center justify-between p-4 bg-yellow-50 border border-yellow-200 rounded-lg"
                      >
                        <div className="space-y-1">
                          <div className="flex items-center gap-2">
                            <Badge variant="outline" className="text-xs">
                              Window {window.window}
                            </Badge>
                            <span className="font-medium">
                              Frames {window.frames}
                            </span>
                          </div>
                          <p className="text-sm text-muted-foreground">
                            {window.time} • Region:{" "}
                            {window.worst_block.replace("_", " ")}
                          </p>
                        </div>
                        <div className="text-right space-y-1">
                          <Badge variant="destructive" className="text-sm">
                            {(window.confidence * 100).toFixed(0)}%
                          </Badge>
                          <p className="text-xs text-muted-foreground">
                            Confidence
                          </p>
                        </div>
                      </div>
                    ))}
                </div>
              </CardContent>
            </Card>
          )}

          {results.quality_scores.advanced_analysis?.most_jittery_windows && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="h-5 w-5 text-orange-500" />
                  Motion Stability Analysis
                </CardTitle>
                <CardDescription>
                  Periods with excessive movement or instability
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {results.quality_scores.advanced_analysis.most_jittery_windows
                    .slice(0, 5)
                    .map((window, index) => (
                      <div
                        key={index}
                        className="flex items-center justify-between p-4 bg-orange-50 border border-orange-200 rounded-lg"
                      >
                        <div className="space-y-1">
                          <div className="flex items-center gap-2">
                            <Badge variant="outline" className="text-xs">
                              Window {window.window}
                            </Badge>
                            <span className="font-medium">
                              Frames {window.frames}
                            </span>
                          </div>
                          <p className="text-sm text-muted-foreground">
                            {window.time} • Region:{" "}
                            {window.worst_block.replace("_", " ")}
                          </p>
                        </div>
                        <div className="text-right space-y-1">
                          <Badge variant="destructive" className="text-sm">
                            {window.displacement.toFixed(0)}px
                          </Badge>
                          <p className="text-xs text-muted-foreground">
                            Displacement
                          </p>
                        </div>
                      </div>
                    ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="data" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5 text-green-500" />
                3D Skeleton Data Export
              </CardTitle>
              <CardDescription>
                Download processed motion data for external analysis or
                visualization
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card className="border border-green-200 bg-green-50/50">
                  <CardHeader className="pb-4">
                    <CardTitle className="text-lg">Exercise Segment</CardTitle>
                    <CardDescription>
                      Trimmed data containing only the relevant exercise
                      movement
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <p className="text-sm">
                        <strong>Format:</strong> Kinect 3D Compatible CSV
                      </p>
                      <p className="text-sm">
                        <strong>Content:</strong>{" "}
                        {results.data_formats?.trimmed ||
                          "Trimmed exercise segment"}
                      </p>
                      <p className="text-sm">
                        <strong>Size:</strong> ~45 frames analyzed
                      </p>
                    </div>
                    <Button className="w-full" variant="default">
                      <Download className="h-4 w-4 mr-2" />
                      Download Trimmed Data
                    </Button>
                  </CardContent>
                </Card>

                <Card className="border border-blue-200 bg-blue-50/50">
                  <CardHeader className="pb-4">
                    <CardTitle className="text-lg">Complete Sequence</CardTitle>
                    <CardDescription>
                      Full video sequence with all detected pose data
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <p className="text-sm">
                        <strong>Format:</strong> Kinect 3D Compatible CSV
                      </p>
                      <p className="text-sm">
                        <strong>Content:</strong>{" "}
                        {results.data_formats?.untrimmed ||
                          "Complete sequence data"}
                      </p>
                      <p className="text-sm">
                        <strong>Size:</strong>{" "}
                        {results.quality_scores.advanced_analysis?.summary
                          .total_frames || 225}{" "}
                        total frames
                      </p>
                    </div>
                    <Button className="w-full" variant="outline">
                      <Download className="h-4 w-4 mr-2" />
                      Download Full Sequence
                    </Button>
                  </CardContent>
                </Card>
              </div>

              {/* Data Format Information */}
              <div className="mt-6 p-4 bg-muted/50 rounded-lg">
                <h4 className="font-medium mb-2 flex items-center gap-2">
                  <Info className="h-4 w-4 text-blue-500" />
                  Data Format Information
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="font-medium">Coordinate System:</p>
                    <p className="text-muted-foreground">
                      3D Cartesian (X, Y, Z)
                    </p>
                  </div>
                  <div>
                    <p className="font-medium">Joint Points:</p>
                    <p className="text-muted-foreground">
                      39 body landmarks per frame
                    </p>
                  </div>
                  <div>
                    <p className="font-medium">Compatibility:</p>
                    <p className="text-muted-foreground">
                      Kinect SDK, OpenPose, MediaPipe
                    </p>
                  </div>
                  <div>
                    <p className="font-medium">Usage:</p>
                    <p className="text-muted-foreground">
                      3D visualization, biomechanical analysis
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Technical Details */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5 text-purple-500" />
                Technical Analysis Details
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="space-y-2">
                  <h4 className="font-medium">Processing Pipeline</h4>
                  <ul className="space-y-1 text-sm text-muted-foreground">
                    <li>• Video frame extraction</li>
                    <li>• 2D pose detection</li>
                    <li>• 3D reconstruction</li>
                    <li>• Motion analysis</li>
                    <li>• Quality assessment</li>
                  </ul>
                </div>
                <div className="space-y-2">
                  <h4 className="font-medium">Quality Metrics</h4>
                  <ul className="space-y-1 text-sm text-muted-foreground">
                    <li>• Pose detection confidence</li>
                    <li>• Motion smoothness</li>
                    <li>• Temporal consistency</li>
                    <li>• Joint visibility</li>
                    <li>• Exercise form scoring</li>
                  </ul>
                </div>
                <div className="space-y-2">
                  <h4 className="font-medium">Output Formats</h4>
                  <ul className="space-y-1 text-sm text-muted-foreground">
                    <li>• CSV data files</li>
                    <li>• 3D coordinate arrays</li>
                    <li>• Frame-by-frame analysis</li>
                    <li>• Confidence scores</li>
                    <li>• Motion statistics</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Action Bar */}
      {/* <Card className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/20 dark:to-indigo-950/20 border-blue-200">
        <CardContent className="flex items-center justify-between py-4">
          <div>
            <h4 className="font-semibold">Analysis Complete</h4>
            <p className="text-sm text-muted-foreground">
              Ready for download or further analysis
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm">
              <BarChart3 className="h-4 w-4 mr-2" />
              View Detailed Report
            </Button>
            <Button size="sm">
              <Download className="h-4 w-4 mr-2" />
              Export All Data
            </Button>
          </div>
        </CardContent>
      </Card> */}
    </div>
  );
}
