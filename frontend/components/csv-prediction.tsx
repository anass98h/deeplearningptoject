/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import type React from "react";

import { useState, useCallback, useRef, useEffect } from "react";
import {
  Upload,
  FileUp,
  AlertCircle,
  CheckCircle2,
  Sparkles,
  Activity,
  BarChart3,
  Scale,
  Layers,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Checkbox } from "@/components/ui/checkbox";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

interface CSVData {
  headers: string[];
  rows: any[][];
  hasZValues: boolean;
}

interface DepthModel {
  name: string;
  version: string;
  path: string;
  model_type: string;
  framework: string;
}

interface KeypointMetric {
  keypoint: string;
  mae: number;
  mse: number;
}

interface OverallMetrics {
  mae: number;
  mse: number;
}

interface DepthPredictionResult {
  model_name: string;
  z_values: Record<string, number[]>;
  processing_time_ms: number;
  framework: string;
  sequence_length: number;
  has_ground_truth: boolean;
  overall_metrics?: OverallMetrics;
  keypoint_metrics?: KeypointMetric[];
  ground_truth?: Record<string, number[]>;
}

export function CSVPrediction() {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [csvData, setCsvData] = useState<CSVData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [availableModels, setAvailableModels] = useState<DepthModel[]>([]);
  const [predictionResult, setPredictionResult] =
    useState<DepthPredictionResult | null>(null);
  const [predictionTime, setPredictionTime] = useState<number | null>(null);
  const [includeGroundTruth, setIncludeGroundTruth] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState("predictions");

  const BACKEND_URL =
    process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

  // Fetch available depth models on component mount
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch(`${BACKEND_URL}/depth-models`);
        if (!response.ok) {
          throw new Error(`Failed to fetch models: ${response.status}`);
        }
        const models = await response.json();
        setAvailableModels(models);

        // Auto-select the first model if available
        if (models.length > 0) {
          setSelectedModel(models[0].name);
        }
      } catch (err) {
        console.error("Error fetching depth models:", err);
        setError(
          "Failed to fetch available models. Please check if the API server is running."
        );
      }
    };

    fetchModels();
  }, [BACKEND_URL]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      if (
        droppedFile.type === "text/csv" ||
        droppedFile.name.endsWith(".csv")
      ) {
        setFile(droppedFile);
        processCSV(droppedFile);
      } else {
        setError("Please upload a CSV file");
      }
    }
  }, []);

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files.length > 0) {
        const selectedFile = e.target.files[0];
        setFile(selectedFile);
        processCSV(selectedFile);
      }
    },
    []
  );

  const handleButtonClick = useCallback(() => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  }, []);

  const processCSV = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    setPredictionResult(null);
    setPredictionTime(null);
    setActiveTab("predictions");

    // Simulate progress
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          return 100;
        }
        return prev + 10;
      });
    }, 100);

    try {
      const text = await file.text();
      const lines = text.split("\n");
      // Try to detect delimiter (tab or comma)
      const firstLine = lines[0];
      const delimiter = firstLine.includes("\t") ? "\t" : ",";
      const headers = lines[0].split(delimiter).map((header) => header.trim());

      // Check if CSV contains z-values
      const hasZValues = headers.some((h) => h.endsWith("_z"));

      // Process rows
      const rows: any[][] = [];
      for (let i = 1; i < lines.length; i++) {
        if (lines[i].trim() === "") continue;

        // Split by the detected delimiter
        const values = lines[i].split(delimiter).map((value) => {
          const trimmed = value.trim();
          return isNaN(Number(trimmed)) ? trimmed : Number(trimmed);
        });

        if (values.length === headers.length) {
          rows.push(values);
        }
      }

      setCsvData({
        headers,
        rows,
        hasZValues,
      });
    } catch (err) {
      console.error(err);
      setError("Failed to process CSV file. Please check the format.");
    } finally {
      clearInterval(interval);
      setProgress(100);
      setLoading(false);
    }
  }, []);

  const runDepthPrediction = useCallback(async () => {
    if (!csvData || !selectedModel || !file) {
      setError("Please select a model and upload a CSV file first");
      return;
    }

    setLoading(true);
    setPredictionResult(null);
    setPredictionTime(null);
    setError(null);
    setActiveTab("predictions");

    // Simulate progress
    let progress = 0;
    const interval = setInterval(() => {
      progress += 2;
      setProgress(Math.min(progress, 100));
      if (progress >= 100) clearInterval(interval);
    }, 50);

    try {
      // Create a form data object to send the file
      const formData = new FormData();
      formData.append("file", file);
      formData.append("include_ground_truth", includeGroundTruth.toString());

      // Send the request to the backend
      const startTime = performance.now();
      const response = await fetch(
        `${BACKEND_URL}/predict-depth/${selectedModel}`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Error: ${response.status}`);
      }

      const result = await response.json();
      setPredictionResult(result);

      // If we have metrics, automatically switch to the metrics tab
      if (result.has_ground_truth && result.overall_metrics) {
        setActiveTab("metrics");
      }

      const endTime = performance.now();
      setPredictionTime(endTime - startTime);
    } catch (err) {
      console.error("Error during depth prediction:", err);
      setError(
        err instanceof Error ? err.message : "Failed to predict depth values"
      );
    } finally {
      setLoading(false);
      clearInterval(interval);
      setProgress(100);
    }
  }, [csvData, selectedModel, file, BACKEND_URL, includeGroundTruth]);

  const renderPredictionResults = () => {
    if (!csvData || !predictionResult) return null;

    // Get all keypoint names (e.g., "head_z", "left_shoulder_z", etc.)
    const keypointNames = Object.keys(predictionResult.z_values);

    // Extract first row of predictions for display
    const firstRowPredictions: Record<string, number> = {};
    keypointNames.forEach((keypoint) => {
      firstRowPredictions[keypoint] = predictionResult.z_values[keypoint][0];
    });

    // Get ground truth for first row if available
    const firstRowGroundTruth: Record<string, number> = {};
    if (predictionResult.ground_truth) {
      Object.keys(predictionResult.ground_truth).forEach((keypoint) => {
        firstRowGroundTruth[keypoint] =
          predictionResult.ground_truth![keypoint][0];
      });
    }

    return (
      <div className="mt-8">
        <div className="flex items-center mb-6">
          <h3 className="text-xl font-semibold bg-gradient-to-r from-cyan-600 to-blue-600 bg-clip-text text-transparent">
            Depth Prediction Results
          </h3>
          <Badge
            variant="outline"
            className="ml-3 bg-gradient-to-r from-cyan-50 to-blue-50 border-cyan-200"
          >
            <Sparkles className="h-3 w-3 mr-1 text-cyan-500" />
            {predictionResult.model_name} ({predictionResult.framework})
          </Badge>
          <Badge
            variant="outline"
            className="ml-3 bg-gradient-to-r from-green-50 to-emerald-50 border-green-200"
          >
            <Activity className="h-3 w-3 mr-1 text-green-500" />
            {(predictionResult.processing_time_ms / 1000).toFixed(3)}s
          </Badge>
          <Badge
            variant="outline"
            className="ml-3 bg-gradient-to-r from-purple-50 to-indigo-50 border-purple-200"
          >
            <Layers className="h-3 w-3 mr-1 text-purple-500" />
            {predictionResult.sequence_length} frames
          </Badge>
        </div>

        <div className="border rounded-lg overflow-auto shadow-md bg-white">
          <h4 className="p-3 font-medium text-gray-700 border-b bg-gray-50">
            Predicted Z-Values (First Row)
            {predictionResult.has_ground_truth &&
              " with Ground Truth Comparison"}
          </h4>
          <Table>
            <TableHeader>
              <TableRow className="bg-gradient-to-r from-blue-50 to-cyan-50">
                <TableHead>Keypoint</TableHead>
                <TableHead>Predicted Z-Value</TableHead>
                {predictionResult.has_ground_truth &&
                  predictionResult.ground_truth && (
                    <>
                      <TableHead>Actual Z-Value</TableHead>
                      <TableHead>Difference</TableHead>
                    </>
                  )}
              </TableRow>
            </TableHeader>
            <TableBody>
              {Object.entries(firstRowPredictions).map(
                ([keypoint, value], i) => {
                  const actualValue = predictionResult.ground_truth
                    ? predictionResult.ground_truth[keypoint]?.[0]
                    : undefined;
                  const difference =
                    actualValue !== undefined ? actualValue - value : null;

                  const differenceClass =
                    difference !== null
                      ? Math.abs(difference) < 0.01
                        ? "text-green-600 font-medium"
                        : Math.abs(difference) < 0.05
                        ? "text-amber-600"
                        : "text-red-600"
                      : "";

                  return (
                    <TableRow key={i} className="hover:bg-blue-50/50">
                      <TableCell className="font-medium">{keypoint}</TableCell>
                      <TableCell>{value.toFixed(6)}</TableCell>
                      {predictionResult.has_ground_truth &&
                        predictionResult.ground_truth && (
                          <>
                            <TableCell>
                              {actualValue !== undefined
                                ? actualValue.toFixed(6)
                                : "N/A"}
                            </TableCell>
                            <TableCell className={differenceClass}>
                              {difference !== null
                                ? difference.toFixed(6)
                                : "N/A"}
                            </TableCell>
                          </>
                        )}
                    </TableRow>
                  );
                }
              )}
            </TableBody>
          </Table>
        </div>
      </div>
    );
  };

  const renderMetricsResults = () => {
    if (
      !predictionResult ||
      !predictionResult.has_ground_truth ||
      !predictionResult.overall_metrics
    ) {
      return (
        <div className="mt-8 p-6 border rounded-lg bg-gray-50 text-center">
          <p className="text-gray-600">
            No comparison metrics available. Upload a CSV with Z-values to
            enable comparison.
          </p>
        </div>
      );
    }

    const { overall_metrics, keypoint_metrics } = predictionResult;

    return (
      <div className="mt-8 space-y-6">
        <div className="flex items-center mb-6">
          <h3 className="text-xl font-semibold bg-gradient-to-r from-purple-600 to-indigo-600 bg-clip-text text-transparent">
            Prediction Accuracy Metrics
          </h3>
        </div>

        {/* Overall metrics cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center">
                <Scale className="h-4 w-4 mr-2 text-red-500" />
                <CardTitle className="text-md">Mean Absolute Error</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {overall_metrics.mae.toFixed(5)}
              </div>
              <CardDescription>
                Average absolute difference between predicted and actual values
              </CardDescription>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center">
                <BarChart3 className="h-4 w-4 mr-2 text-amber-500" />
                <CardTitle className="text-md">Mean Squared Error</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {overall_metrics.mse.toFixed(5)}
              </div>
              <CardDescription>
                Average of the squared differences between predicted and actual
                values
              </CardDescription>
            </CardContent>
          </Card>
        </div>

        {/* Per-keypoint metrics table */}
        {keypoint_metrics && keypoint_metrics.length > 0 && (
          <div className="border rounded-lg overflow-auto shadow-md bg-white">
            <h4 className="p-3 font-medium text-gray-700 border-b bg-gray-50">
              Per-Keypoint Metrics
            </h4>
            <Table>
              <TableHeader>
                <TableRow className="bg-gradient-to-r from-purple-50 to-indigo-50">
                  <TableHead>Keypoint</TableHead>
                  <TableHead>MAE</TableHead>
                  <TableHead>MSE</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {keypoint_metrics.map((metric, i) => (
                  <TableRow key={i} className="hover:bg-purple-50/50">
                    <TableCell className="font-medium">
                      {metric.keypoint}
                    </TableCell>
                    <TableCell>{metric.mae.toFixed(5)}</TableCell>
                    <TableCell>{metric.mse.toFixed(5)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-8">
      {/* File upload area */}
      <div
        className={`border-2 border-dashed rounded-xl p-10 text-center transition-all duration-200 ${
          isDragging
            ? "border-primary bg-primary/5 shadow-lg shadow-primary/10"
            : "border-gray-200 hover:border-gray-300 hover:bg-gray-50/50"
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="flex flex-col items-center justify-center space-y-4">
          <div className="p-4 rounded-full bg-gradient-to-br from-primary/10 to-primary/20 shadow-inner">
            <Upload className="h-10 w-10 text-primary" />
          </div>
          <div>
            <h3 className="text-xl font-medium bg-gradient-to-r from-gray-700 to-gray-900 bg-clip-text text-transparent">
              Drag and drop your CSV file
            </h3>
            <p className="text-sm text-muted-foreground mt-1">
              or click to browse files
            </p>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={handleFileChange}
          />
          <Button
            onClick={handleButtonClick}
            variant="outline"
            className="transition-all duration-200 hover:shadow-md hover:bg-gradient-to-r hover:from-gray-50 hover:to-gray-100"
          >
            <FileUp className="mr-2 h-4 w-4" />
            Select CSV File
          </Button>
        </div>
      </div>

      {/* Error message */}
      {error && (
        <Alert
          variant="destructive"
          className="border-0 shadow-lg bg-gradient-to-r from-rose-50 to-red-50"
        >
          <AlertCircle className="h-4 w-4 text-red-500" />
          <AlertTitle className="text-red-700">Error</AlertTitle>
          <AlertDescription className="text-red-600">{error}</AlertDescription>
        </Alert>
      )}

      {/* Loading progress */}
      {loading && (
        <div className="space-y-2 p-4 rounded-lg bg-gradient-to-r from-gray-50 to-gray-100 shadow-md">
          <div className="flex justify-between text-sm">
            <span className="font-medium text-gray-700">Processing...</span>
            <span className="font-semibold text-primary">{progress}%</span>
          </div>
          <Progress value={progress} className="h-2 bg-gray-200" />
        </div>
      )}

      {csvData && (
        <div className="space-y-6 rounded-xl overflow-hidden bg-white shadow-xl">
          <div className="p-6">
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 p-4 rounded-lg bg-gradient-to-r from-blue-50 to-cyan-50 shadow-sm">
              <h3 className="text-lg font-medium text-blue-900">
                Depth Estimation
                {csvData.hasZValues && (
                  <Badge className="ml-2 bg-blue-100 text-blue-800">
                    Z-Values Detected
                  </Badge>
                )}
              </h3>

              <div className="flex flex-col md:flex-row gap-3 w-full md:w-auto">
                <Select value={selectedModel} onValueChange={setSelectedModel}>
                  <SelectTrigger className="w-full md:w-[220px] border-blue-200 bg-white shadow-sm">
                    <SelectValue placeholder="Select a model" />
                  </SelectTrigger>
                  <SelectContent>
                    {availableModels.map((model) => (
                      <SelectItem key={model.name} value={model.name}>
                        {model.name} (v{model.version})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                {csvData.hasZValues && (
                  <div className="flex items-center space-x-2 bg-white px-3 py-2 rounded-md border border-blue-200 shadow-sm">
                    <Checkbox
                      id="includeGroundTruth"
                      checked={includeGroundTruth}
                      onCheckedChange={(checked) =>
                        setIncludeGroundTruth(checked as boolean)
                      }
                    />
                    <label
                      htmlFor="includeGroundTruth"
                      className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                    >
                      Include Ground Truth
                    </label>
                  </div>
                )}

                <Button
                  onClick={runDepthPrediction}
                  disabled={loading || !selectedModel}
                  className="w-full md:w-auto bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 shadow-md hover:shadow-lg transition-all duration-200"
                >
                  Predict Depth
                </Button>
              </div>
            </div>

            <div className="mt-6 border rounded-lg overflow-auto shadow-md bg-white">
              <h4 className="p-3 font-medium text-gray-700 border-b bg-gray-50">
                Input Data (First 5 Rows)
              </h4>
              <Table>
                <TableHeader>
                  <TableRow className="bg-gradient-to-r from-gray-50 to-gray-100">
                    {csvData.headers.map((header, i) => (
                      <TableHead
                        key={i}
                        className={`font-semibold ${
                          header.endsWith("_z") ? "bg-blue-50" : ""
                        }`}
                      >
                        {header}
                        {header.endsWith("_z") && (
                          <Badge
                            className="ml-2 bg-blue-100 text-blue-800 hover:bg-blue-200 border-0"
                            variant="outline"
                          >
                            Z
                          </Badge>
                        )}
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {csvData.rows.slice(0, 5).map((row, i) => (
                    <TableRow key={i} className="hover:bg-blue-50/50">
                      {row.map((cell, j) => (
                        <TableCell
                          key={j}
                          className={
                            csvData.headers[j].endsWith("_z")
                              ? "bg-blue-50/30"
                              : ""
                          }
                        >
                          {typeof cell === "number" ? cell.toFixed(6) : cell}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>

            {/* Show prediction time if available */}
            {predictionTime && (
              <Alert className="mt-6 border-0 shadow-md bg-gradient-to-r from-green-50 to-emerald-50">
                <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                <AlertTitle className="text-emerald-800">
                  Depth Prediction Complete
                </AlertTitle>
                <AlertDescription className="text-emerald-700">
                  Frontend processing time: {(predictionTime / 1000).toFixed(2)}{" "}
                  seconds
                  <br />
                  Backend processing time:{" "}
                  {predictionResult
                    ? (predictionResult.processing_time_ms / 1000).toFixed(3)
                    : "N/A"}{" "}
                  seconds
                </AlertDescription>
              </Alert>
            )}

            {/* Results tabs */}
            {predictionResult && (
              <div className="mt-6">
                <Tabs value={activeTab} onValueChange={setActiveTab}>
                  <TabsList className="w-full">
                    <TabsTrigger value="predictions" className="flex-1">
                      Predictions
                    </TabsTrigger>
                    <TabsTrigger
                      value="metrics"
                      className="flex-1"
                      disabled={
                        !predictionResult.has_ground_truth ||
                        !predictionResult.overall_metrics
                      }
                    >
                      Accuracy Metrics
                    </TabsTrigger>
                  </TabsList>
                  <TabsContent value="predictions">
                    {renderPredictionResults()}
                  </TabsContent>
                  <TabsContent value="metrics">
                    {renderMetricsResults()}
                  </TabsContent>
                </Tabs>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
