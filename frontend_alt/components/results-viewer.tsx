"use client";

import { useState, useEffect, useRef } from "react";
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
  CheckCircle,
  XCircle,
  Info,
  Play,
  Pause,
  RotateCcw,
} from "lucide-react";
import * as THREE from "three";

interface SkeletonViewerProps {
  skeletonDataCsv: string; // Changed to accept CSV data directly
  frameCount: number;
}

function SkeletonViewer({ skeletonDataCsv, frameCount }: SkeletonViewerProps) {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const skeletonGroupRef = useRef<THREE.Group | null>(null);
  const animationRef = useRef<number | null>(null);

  const [skeletonData, setSkeletonData] = useState<any[]>([]);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const connections = [
    [0, 1],
    [0, 2],
    [1, 3],
    [3, 5],
    [2, 4],
    [4, 6],
    [1, 7],
    [2, 8],
    [7, 8],
    [7, 9],
    [9, 11],
    [8, 10],
    [10, 12],
  ];

  const parseSkeletonData = () => {
    if (!skeletonDataCsv) {
      setError("No skeleton data provided");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const lines = skeletonDataCsv.trim().split("\n");
      console.log("Parsing skeleton data:", lines.length, "lines");

      const data = lines.slice(1).map((line, index) => {
        const values = line.split(",");
        const frame: any = {
          frameNo: parseInt(values[0]) || index,
          joints: [],
        };

        for (let i = 1; i < values.length; i += 3) {
          if (i + 2 < values.length) {
            const x = parseFloat(values[i]);
            const y = parseFloat(values[i + 1]);
            const z = parseFloat(values[i + 2]);

            // Only add valid joints (not NaN or undefined)
            if (!isNaN(x) && !isNaN(y) && !isNaN(z)) {
              frame.joints.push({ x, y, z });
            }
          }
        }
        return frame;
      });

      console.log("Parsed skeleton data:", data.length, "frames");
      console.log("Sample frame:", data[0]);
      setSkeletonData(data);
    } catch (err) {
      console.error("Error parsing skeleton data:", err);
      setError(err instanceof Error ? err.message : "Failed to parse data");
    } finally {
      setLoading(false);
    }
  };

  const initializeScene = () => {
    if (!mountRef.current) return;

    const width = mountRef.current.clientWidth;
    const height = 600;

    // Clear container
    while (mountRef.current.firstChild) {
      mountRef.current.removeChild(mountRef.current.firstChild);
    }

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a2639);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(25, width / height, 0.1, 1000);
    camera.position.set(0, 1.0, 20);
    camera.lookAt(0, 1, 0);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = true;
    rendererRef.current = renderer;
    mountRef.current.appendChild(renderer.domElement);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
    directionalLight.position.set(5, 10, 5);
    scene.add(directionalLight);

    // Basic mouse controls
    let mouseDown = false;
    let mouseX = 0;
    let mouseY = 0;

    const handleMouseDown = (event: MouseEvent) => {
      mouseDown = true;
      mouseX = event.clientX;
      mouseY = event.clientY;
    };

    const handleMouseUp = () => {
      mouseDown = false;
    };

    const handleMouseMove = (event: MouseEvent) => {
      if (!mouseDown) return;

      const deltaX = event.clientX - mouseX;
      const deltaY = event.clientY - mouseY;

      const spherical = new THREE.Spherical();
      spherical.setFromVector3(camera.position);
      spherical.theta -= deltaX * 0.01;
      spherical.phi += deltaY * 0.01;
      spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi));

      camera.position.setFromSpherical(spherical);
      camera.lookAt(0, 1, 0);

      mouseX = event.clientX;
      mouseY = event.clientY;
    };

    const handleWheel = (event: WheelEvent) => {
      const distance = camera.position.length();
      const newDistance = Math.max(
        3,
        Math.min(30, distance + event.deltaY * 0.01)
      );
      camera.position.normalize().multiplyScalar(newDistance);
    };

    renderer.domElement.addEventListener("mousedown", handleMouseDown);
    renderer.domElement.addEventListener("mouseup", handleMouseUp);
    renderer.domElement.addEventListener("mousemove", handleMouseMove);
    renderer.domElement.addEventListener("wheel", handleWheel);

    // Animation loop
    const animate = () => {
      animationRef.current = requestAnimationFrame(animate);
      renderer.render(scene, camera);
    };
    animate();

    // Handle resize
    const handleResize = () => {
      if (!mountRef.current) return;
      const newWidth = mountRef.current.clientWidth;
      camera.aspect = newWidth / height;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, height);
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      renderer.domElement.removeEventListener("mousedown", handleMouseDown);
      renderer.domElement.removeEventListener("mouseup", handleMouseUp);
      renderer.domElement.removeEventListener("mousemove", handleMouseMove);
      renderer.domElement.removeEventListener("wheel", handleWheel);

      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (rendererRef.current && mountRef.current) {
        mountRef.current.removeChild(rendererRef.current.domElement);
      }
    };
  };

  const updateSkeleton = (frameIndex: number) => {
    if (!skeletonData.length || !sceneRef.current) return;

    const validFrameIndex = Math.min(frameIndex, skeletonData.length - 1);
    const frameData = skeletonData[validFrameIndex];

    // Remove previous skeleton
    if (skeletonGroupRef.current) {
      sceneRef.current.remove(skeletonGroupRef.current);
    }

    const skeletonGroup = new THREE.Group();
    skeletonGroupRef.current = skeletonGroup;

    if (!frameData.joints || frameData.joints.length === 0) {
      console.log("No joints found for frame", validFrameIndex);
      return;
    }

    console.log(
      "Updating skeleton for frame",
      validFrameIndex,
      "with",
      frameData.joints.length,
      "joints"
    );

    // Create joint spheres
    const jointRadius = 0.05;

    frameData.joints.forEach((joint: any, index: number) => {
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(jointRadius, 8, 8),
        new THREE.MeshPhongMaterial({
          color: index === 0 ? 0xff0000 : 0x00ff00,
        })
      );
      sphere.position.set(joint.x, joint.y, joint.z);
      skeletonGroup.add(sphere);
    });

    // Create bone connections
    connections.forEach(([startIdx, endIdx]) => {
      if (
        startIdx < frameData.joints.length &&
        endIdx < frameData.joints.length
      ) {
        const start = frameData.joints[startIdx];
        const end = frameData.joints[endIdx];

        const startVec = new THREE.Vector3(start.x, start.y, start.z);
        const endVec = new THREE.Vector3(end.x, end.y, end.z);
        const distance = startVec.distanceTo(endVec);

        if (distance < 0.01) return;

        const boneMaterial = new THREE.MeshPhongMaterial({ color: 0x3498db });
        const boneGeometry = new THREE.CylinderGeometry(
          0.03,
          0.03,
          distance,
          8
        );

        boneGeometry.translate(0, distance / 2, 0);
        boneGeometry.rotateX(Math.PI / 2);

        const bone = new THREE.Mesh(boneGeometry, boneMaterial);
        bone.position.copy(startVec);
        bone.lookAt(endVec);
        skeletonGroup.add(bone);
      }
    });

    sceneRef.current.add(skeletonGroup);
  };

  const playAnimation = () => {
    if (!isPlaying || currentFrame >= skeletonData.length - 1) return;

    setTimeout(() => {
      setCurrentFrame((prev) => {
        const nextFrame = prev + 1;
        if (nextFrame >= skeletonData.length) {
          setIsPlaying(false);
          return 0;
        }
        return nextFrame;
      });
    }, 100);
  };

  useEffect(() => {
    parseSkeletonData(); // Parse data from props instead of fetching
    const cleanup = initializeScene();
    return cleanup;
  }, [skeletonDataCsv]);

  useEffect(() => {
    if (skeletonData.length > 0) {
      updateSkeleton(currentFrame);
    }
  }, [currentFrame, skeletonData]);

  useEffect(() => {
    if (isPlaying) {
      playAnimation();
    }
  }, [isPlaying, currentFrame, skeletonData]);

  const handlePlay = () => {
    if (currentFrame >= skeletonData.length - 1) {
      setCurrentFrame(0);
    }
    setIsPlaying(!isPlaying);
  };

  const handleReset = () => {
    setIsPlaying(false);
    setCurrentFrame(0);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96 border border-dashed border-gray-300 rounded-lg">
        <div className="text-center space-y-2">
          <Activity className="h-8 w-8 text-primary animate-spin mx-auto" />
          <p>Loading 3D skeleton data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-96 border border-dashed border-red-300 rounded-lg bg-red-50">
        <div className="text-center space-y-2">
          <XCircle className="h-8 w-8 text-red-500 mx-auto" />
          <p className="text-red-700">Failed to load skeleton data: {error}</p>
        </div>
      </div>
    );
  }

  if (!skeletonData.length) {
    return (
      <div className="flex items-center justify-center h-96 border border-dashed border-gray-300 rounded-lg">
        <div className="text-center space-y-2">
          <FileText className="h-8 w-8 text-muted-foreground mx-auto" />
          <p>No skeleton data available</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="border border-gray-300 rounded-lg overflow-hidden">
        <div ref={mountRef} className="w-full" style={{ height: "600px" }} />
      </div>

      <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handlePlay}
            disabled={skeletonData.length === 0}
          >
            {isPlaying ? (
              <Pause className="h-4 w-4" />
            ) : (
              <Play className="h-4 w-4" />
            )}
            {isPlaying ? "Pause" : "Play"}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleReset}
            disabled={skeletonData.length === 0}
          >
            <RotateCcw className="h-4 w-4" />
            Reset
          </Button>
        </div>

        <div className="flex items-center gap-4">
          <span className="text-sm text-muted-foreground">
            Frame: {currentFrame + 1} / {skeletonData.length}
          </span>
          {skeletonData.length > 0 && (
            <input
              type="range"
              min="0"
              max={skeletonData.length - 1}
              value={currentFrame}
              onChange={(e) => setCurrentFrame(parseInt(e.target.value))}
              className="w-32"
            />
          )}
        </div>
      </div>
    </div>
  );
}

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

  const fetchResults = async () => {
    if (!jobId) return;

    setLoading(true);
    setError(null);
    try {
      console.log(`Fetching final results for job: ${jobId}`);
      const response = await fetch(
        `http://localhost:8000/video-data/${jobId}/final`
      );

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      console.log("Final results received:", data);
      setResults(data);
    } catch (err) {
      console.error("Failed to fetch results:", err);
      setError(err instanceof Error ? err.message : "Failed to load results");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchResults();
  }, [jobId]);

  if (!jobId) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-center space-y-2">
            <FileText className="h-12 w-12 text-muted-foreground mx-auto" />
            <p className="text-lg font-medium">No results to display</p>
            <p className="text-sm text-muted-foreground">
              Complete a video analysis to view results
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (loading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-center space-y-4">
            <Activity className="h-8 w-8 text-primary animate-spin mx-auto" />
            <p>Loading results...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <XCircle className="h-4 w-4" />
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  if (!results) return null;

  const getScoreColor = (score: number) => {
    if (score <= 1.0) return "text-green-600";
    if (score <= 2.0) return "text-yellow-600";
    return "text-red-600";
  };

  const getScoreBadgeVariant = (
    interpretation: string
  ): "default" | "secondary" | "destructive" => {
    const lower = interpretation.toLowerCase();
    if (lower.includes("excellent") || lower.includes("very good"))
      return "default";
    if (lower.includes("good") || lower.includes("fair")) return "secondary";
    return "destructive";
  };

  const downloadSkeletonData = async (type: "trimmed" | "untrimmed") => {
    try {
      let csvData: string;

      if (type === "trimmed" && results.skeleton_data?.trimmed) {
        // Use data from JSON response
        csvData = results.skeleton_data.trimmed;
      } else if (type === "untrimmed" && results.skeleton_data?.untrimmed) {
        // Use data from JSON response
        csvData = results.skeleton_data.untrimmed;
      } else {
        // Fallback to API endpoint if data not in response
        const endpoint = type === "trimmed" ? "trimmed" : "kinect3d";
        console.log(
          `Downloading ${type} data from: http://localhost:8000/video-data/${jobId}/${endpoint}`
        );

        const response = await fetch(
          `http://localhost:8000/video-data/${jobId}/${endpoint}`
        );

        if (!response.ok) {
          throw new Error(`Failed to download data: HTTP ${response.status}`);
        }

        csvData = await response.text();
      }

      const blob = new Blob([csvData], { type: "text/csv" });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${results?.filename}_${type}_skeleton.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (error) {
      console.error("Download failed:", error);
      alert(
        `Download failed: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    }
  };

  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleString();
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card className="border-l-4 border-l-blue-500">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Trophy className="h-6 w-6 text-blue-500" />
              <div>
                <CardTitle className="text-xl">Analysis Results</CardTitle>
                <CardDescription className="mt-1">
                  <span className="font-medium">{results.filename}</span>
                  <span className="mx-2">•</span>
                  Job ID: {results.job_id.slice(0, 8)}...
                </CardDescription>
              </div>
            </div>
            <Badge
              variant={
                results.status === "completed" ? "default" : "destructive"
              }
              className="flex items-center gap-1"
            >
              {results.status === "completed" ? (
                <CheckCircle className="h-3 w-3" />
              ) : (
                <XCircle className="h-3 w-3" />
              )}
              {results.status.toUpperCase()}
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <Alert className="mb-4">
            <Info className="h-4 w-4" />
            <AlertDescription>{results.message}</AlertDescription>
          </Alert>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div>
              <span className="font-medium">Created:</span>{" "}
              {formatTimestamp(results.created_at)}
            </div>
            <div>
              <span className="font-medium">Updated:</span>{" "}
              {formatTimestamp(results.updated_at)}
            </div>
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="scores" className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="scores">Quality Scores</TabsTrigger>
          <TabsTrigger value="analysis">Advanced Analysis</TabsTrigger>
          <TabsTrigger value="data">Skeleton Data</TabsTrigger>
        </TabsList>

        <TabsContent value="scores" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {/* Final Exercise Score */}
            {typeof results.quality_scores.final_exercise_score ===
              "number" && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">
                    Final Exercise Score
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div
                    className={`text-3xl font-bold ${getScoreColor(
                      results.quality_scores.final_exercise_score
                    )}`}
                  >
                    {results.quality_scores.final_exercise_score.toFixed(3)}
                  </div>
                  <div className="text-sm text-muted-foreground mt-1">
                    out of 5.0
                  </div>
                  {results.quality_scores.score_interpretation && (
                    <Badge
                      variant={getScoreBadgeVariant(
                        results.quality_scores.score_interpretation
                      )}
                      className="mt-2"
                    >
                      {results.quality_scores.score_interpretation}
                    </Badge>
                  )}
                </CardContent>
              </Card>
            )}

            {/* 2D Goodness */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">
                  2D Goodness
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {results.quality_scores.ugly_2d_goodness.toFixed(3)}
                </div>
                <Progress
                  value={Math.min(
                    results.quality_scores.ugly_2d_goodness * 20,
                    100
                  )}
                  className="mt-2"
                />
              </CardContent>
            </Card>

            {/* 2D Confidence */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">
                  2D Confidence
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {results.quality_scores.ugly_2d_confidence.toFixed(3)}
                </div>
                <Progress
                  value={results.quality_scores.ugly_2d_confidence * 100}
                  className="mt-2"
                />
              </CardContent>
            </Card>

            {/* 3D Exercise Score */}
            {typeof results.quality_scores.bad_3d_exercise_score ===
              "number" && (
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">
                    3D Exercise Score
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {results.quality_scores.bad_3d_exercise_score.toFixed(3)}
                  </div>
                  <Progress
                    value={results.quality_scores.bad_3d_exercise_score * 100}
                    className="mt-2"
                  />
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>

        <TabsContent value="analysis" className="space-y-6">
          {results.quality_scores.advanced_analysis ? (
            <>
              {/* Summary Statistics */}
              <Card>
                <CardHeader>
                  <CardTitle>Analysis Summary</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold">
                        {
                          results.quality_scores.advanced_analysis.summary
                            .total_frames
                        }
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Total Frames
                      </p>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold">
                        {
                          results.quality_scores.advanced_analysis.summary
                            .windows_analyzed
                        }
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Windows Analyzed
                      </p>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold">
                        {results.quality_scores.advanced_analysis.summary
                          .has_confidence_data
                          ? "Yes"
                          : "No"}
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Has Confidence Data
                      </p>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold">
                        {results.quality_scores.advanced_analysis.summary.confidence_stats.avg.toFixed(
                          3
                        )}
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Avg Confidence
                      </p>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold">
                        {results.quality_scores.advanced_analysis.summary.motion_stats.avg.toFixed(
                          1
                        )}
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Avg Motion
                      </p>
                    </div>
                  </div>

                  {/* Confidence and Motion Stats */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
                    <div>
                      <h4 className="font-medium mb-3">
                        Confidence Statistics
                      </h4>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span>Minimum:</span>
                          <span className="font-mono">
                            {results.quality_scores.advanced_analysis.summary.confidence_stats.min.toFixed(
                              3
                            )}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Maximum:</span>
                          <span className="font-mono">
                            {results.quality_scores.advanced_analysis.summary.confidence_stats.max.toFixed(
                              3
                            )}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Average:</span>
                          <span className="font-mono">
                            {results.quality_scores.advanced_analysis.summary.confidence_stats.avg.toFixed(
                              3
                            )}
                          </span>
                        </div>
                      </div>
                    </div>
                    <div>
                      <h4 className="font-medium mb-3">Motion Statistics</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span>Minimum:</span>
                          <span className="font-mono">
                            {results.quality_scores.advanced_analysis.summary.motion_stats.min.toFixed(
                              1
                            )}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Maximum:</span>
                          <span className="font-mono">
                            {results.quality_scores.advanced_analysis.summary.motion_stats.max.toFixed(
                              1
                            )}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Average:</span>
                          <span className="font-mono">
                            {results.quality_scores.advanced_analysis.summary.motion_stats.avg.toFixed(
                              1
                            )}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Worst Confidence Windows */}
              {results.quality_scores.advanced_analysis
                .worst_confidence_windows &&
                results.quality_scores.advanced_analysis
                  .worst_confidence_windows.length > 0 && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <AlertTriangle className="h-5 w-5 text-yellow-500" />
                        Worst Confidence Windows
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        {results.quality_scores.advanced_analysis.worst_confidence_windows.map(
                          (window, index) => (
                            <div
                              key={index}
                              className="flex items-center justify-between p-3 border rounded-lg"
                            >
                              <div className="space-y-1">
                                <div className="flex items-center gap-2">
                                  <Badge variant="outline">
                                    Window {window.window}
                                  </Badge>
                                  <span className="font-medium">
                                    Frames {window.frames}
                                  </span>
                                </div>
                                <p className="text-sm text-muted-foreground">
                                  {window.time} • Block: {window.worst_block}
                                </p>
                              </div>
                              <div className="text-right">
                                <div className="font-mono font-bold">
                                  {window.confidence.toFixed(3)}
                                </div>
                                <div className="text-xs text-muted-foreground">
                                  Confidence
                                </div>
                              </div>
                            </div>
                          )
                        )}
                      </div>
                    </CardContent>
                  </Card>
                )}

              {/* Most Jittery Windows */}
              {results.quality_scores.advanced_analysis.most_jittery_windows &&
                results.quality_scores.advanced_analysis.most_jittery_windows
                  .length > 0 && (
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Zap className="h-5 w-5 text-orange-500" />
                        Most Jittery Windows
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        {results.quality_scores.advanced_analysis.most_jittery_windows.map(
                          (window, index) => (
                            <div
                              key={index}
                              className="flex items-center justify-between p-3 border rounded-lg"
                            >
                              <div className="space-y-1">
                                <div className="flex items-center gap-2">
                                  <Badge variant="outline">
                                    Window {window.window}
                                  </Badge>
                                  <span className="font-medium">
                                    Frames {window.frames}
                                  </span>
                                </div>
                                <p className="text-sm text-muted-foreground">
                                  {window.time} • Block: {window.worst_block}
                                </p>
                              </div>
                              <div className="text-right">
                                <div className="font-mono font-bold">
                                  {window.displacement.toFixed(1)}
                                </div>
                                <div className="text-xs text-muted-foreground">
                                  Displacement
                                </div>
                              </div>
                            </div>
                          )
                        )}
                      </div>
                    </CardContent>
                  </Card>
                )}
            </>
          ) : (
            <Card>
              <CardContent className="flex items-center justify-center py-12">
                <div className="text-center space-y-2">
                  <BarChart3 className="h-12 w-12 text-muted-foreground mx-auto" />
                  <p className="text-lg font-medium">
                    No advanced analysis data available
                  </p>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="data" className="space-y-6">
          {results.skeleton_data?.trimmed ? (
            <>
              {/* 3D Skeleton Viewer */}
              <Card>
                <CardHeader>
                  <CardTitle>3D Skeleton Viewer</CardTitle>
                  <CardDescription>
                    Interactive 3D visualization of the skeleton data
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <SkeletonViewer
                    skeletonDataCsv={results.skeleton_data.trimmed}
                    frameCount={
                      results.quality_scores.advanced_analysis?.summary
                        .total_frames || 225
                    }
                  />
                </CardContent>
              </Card>

              {/* Download Section */}
              <Card>
                <CardHeader>
                  <CardTitle>Data Export</CardTitle>
                  <CardDescription>
                    Download CSV files containing 3D pose estimation data
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <div>
                        <h4 className="font-medium">
                          Trimmed Exercise Segment
                        </h4>
                        <p className="text-sm text-muted-foreground mt-1">
                          Contains only the relevant exercise movement data
                        </p>
                        {results.data_formats?.trimmed && (
                          <p className="text-xs text-muted-foreground mt-1">
                            Format: {results.data_formats.trimmed}
                          </p>
                        )}
                      </div>
                      <Button
                        variant="outline"
                        className="w-full"
                        onClick={() => downloadSkeletonData("trimmed")}
                      >
                        <Download className="h-4 w-4 mr-2" />
                        Download Trimmed CSV
                      </Button>
                    </div>

                    <div className="space-y-4">
                      <div>
                        <h4 className="font-medium">Full Video Sequence</h4>
                        <p className="text-sm text-muted-foreground mt-1">
                          Complete 3D pose data for entire video sequence
                        </p>
                        {results.data_formats?.untrimmed && (
                          <p className="text-xs text-muted-foreground mt-1">
                            Format: {results.data_formats.untrimmed}
                          </p>
                        )}
                      </div>
                      <Button
                        variant="outline"
                        className="w-full"
                        onClick={() => downloadSkeletonData("untrimmed")}
                      >
                        <Download className="h-4 w-4 mr-2" />
                        Download Full CSV
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </>
          ) : (
            <Card>
              <CardContent className="flex items-center justify-center py-12">
                <div className="text-center space-y-2">
                  <FileText className="h-12 w-12 text-muted-foreground mx-auto" />
                  <p className="text-lg font-medium">
                    No skeleton data available
                  </p>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
