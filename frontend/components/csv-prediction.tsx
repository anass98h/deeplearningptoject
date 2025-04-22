/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import React, { useState, useCallback, useRef, useEffect } from "react";
import {
  Upload,
  FileUp,
  AlertCircle,
  Eye,
  Database,
  FileType,
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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent } from "@/components/ui/card";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

// Import the PoseNet3DVisualization component
import { PoseNet3DVisualization } from "./PoseNet3DVisualization";
import { Switch } from "./ui/switch";

interface CSVData {
  headers: string[];
  rows: any[][];
  hasZValues: boolean;
}

export const dynamic = "force-dynamic";
export const revalidate = 0;

export function CSVPrediction() {
  const [isDragging, setIsDragging] = useState(false);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [file, setFile] = useState<File | null>(null);
  const [csvData, setCsvData] = useState<CSVData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // 3D visualization state
  const [groundTruthData, setGroundTruthData] = useState<any[]>([]);
  const [predictedData, setPredictedData] = useState<any[]>([]);
  const [showSideBySide, setShowSideBySide] = useState(false);
  const [activeTab, setActiveTab] = useState("data");
  const [dataKey, setDataKey] = useState(0); // Used to force re-render of visualization

  const BACKEND_URL =
    process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

  // Fetch kinect data on component mount
  useEffect(() => {
    fetchKinectData();
  }, []);

  // Function to fetch kinect data from our new endpoint
  const fetchKinectData = async () => {
    setLoading(true);
    setError(null);

    try {
      console.log("Fetching kinect data from:", `${BACKEND_URL}/kinect-data`);
      const response = await fetch(`${BACKEND_URL}/kinect-data`);

      if (!response.ok) {
        throw new Error(`Failed to fetch kinect data: ${response.status}`);
      }

      const data = await response.json();

      if (data.content) {
        console.log("Received CSV content from backend");

        // Create a File object from the CSV content
        const csvFile = new File([data.content], "kinect_data.csv", {
          type: "text/csv",
        });

        // Process the file with our existing CSV processor
        setFile(csvFile);
        const processedData = await processCSVData(csvFile);
        if (processedData) {
          // When loading from server, set this as ground truth data
          setGroundTruthData(processedData);
        }

        console.log("Successfully processed CSV data");
      } else {
        throw new Error("Received empty content from backend");
      }
    } catch (err) {
      console.error("Error fetching kinect data:", err);
      setError(
        "Failed to fetch kinect data: " +
          (err instanceof Error ? err.message : String(err))
      );
    } finally {
      setLoading(false);
      setProgress(100);
    }
  };

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

  // Process CSV data without updating state (used for ground truth data)
  const processCSVData = async (file: File): Promise<any[] | null> => {
    try {
      const text = await file.text();
      const lines = text.split("\n");

      if (lines.length === 0) {
        throw new Error("CSV file is empty");
      }

      // Try to detect delimiter (tab or comma)
      const firstLine = lines[0];
      const delimiter = firstLine.includes("\t") ? "\t" : ",";

      // Parse headers, making sure to handle spaces after the delimiter
      const headers = firstLine.split(delimiter).map((header) => header.trim());

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

      // Convert to format for 3D visualization - object with properties for each column
      const objectRows = rows.map((row) => {
        const obj: Record<string, any> = {}; // Explicitly type the object
        headers.forEach((header, index) => {
          obj[header] = row[index];
        });
        return obj;
      });

      return objectRows;
    } catch (err) {
      console.error("Error processing CSV data:", err);
      return null;
    }
  };

  // Process CSV and update UI state
  const processCSV = async (file: File) => {
    setLoading(true);
    setError(null);
    setProgress(0);

    // Simulate progress
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 90) {
          clearInterval(interval);
          return 90;
        }
        return prev + 10;
      });
    }, 100);

    try {
      const text = await file.text();
      const lines = text.split("\n");

      if (lines.length === 0) {
        throw new Error("CSV file is empty");
      }

      // Try to detect delimiter (tab or comma)
      const firstLine = lines[0];
      const delimiter = firstLine.includes("\t") ? "\t" : ",";

      // Parse headers, making sure to handle spaces after the delimiter
      const headers = firstLine.split(delimiter).map((header) => header.trim());

      // Check if CSV contains z-values
      const hasZValues = headers.some((h) => h.endsWith("_z"));
      console.log("CSV Headers:", headers);
      console.log("Has Z Values:", hasZValues);

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

      console.log(`Processed ${rows.length} rows from CSV`);

      setCsvData({
        headers,
        rows,
        hasZValues,
      });

      // Convert to format for 3D visualization - object with properties for each column
      const objectRows = rows.map((row) => {
        const obj: Record<string, any> = {}; // Explicitly type the object
        headers.forEach((header, index) => {
          obj[header] = row[index];
        });
        return obj;
      });

      console.log(
        `Created ${objectRows.length} object rows for 3D visualization`
      );

      // Log a sample object to debug
      if (objectRows.length > 0) {
        console.log("Sample object row:", objectRows[0]);
      }

      // Set as predicted data
      setPredictedData(objectRows);

      // If we have both ground truth and prediction, enable side-by-side by default
      if (groundTruthData.length > 0) {
        setShowSideBySide(true);
      }

      setDataKey((prev) => prev + 1); // Force re-render of the visualization
    } catch (err) {
      console.error("Error processing CSV:", err);
      setError("Failed to process CSV file. Please check the format.");
    } finally {
      clearInterval(interval);
      setProgress(100);
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      <Card className="shadow-lg border-blue-100">
        <CardContent className="pt-6">
          <Tabs
            value={activeTab}
            onValueChange={setActiveTab}
            className="w-full"
          >
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="data" className="flex items-center">
                <FileType className="h-4 w-4 mr-2" />
                Data View
              </TabsTrigger>
              <TabsTrigger
                value="visualization"
                className="flex items-center"
                disabled={!groundTruthData.length && !predictedData.length}
              >
                <Eye className="h-4 w-4 mr-2" />
                3D View
                {(groundTruthData.length > 0 || predictedData.length > 0) && (
                  <Badge
                    variant="outline"
                    className="ml-2 bg-green-50 border-green-200"
                  >
                    {Math.max(groundTruthData.length, predictedData.length)}{" "}
                    Frames
                  </Badge>
                )}
              </TabsTrigger>
            </TabsList>

            <TabsContent value="data" className="pt-4">
              {/* File upload area */}
              <div
                className={`border-2 border-dashed rounded-xl p-6 text-center transition-all duration-200 ${
                  isDragging
                    ? "border-primary bg-primary/5 shadow-lg shadow-primary/10"
                    : "border-gray-200 hover:border-gray-300 hover:bg-gray-50/50"
                }`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <div className="flex flex-col items-center justify-center space-y-4">
                  <div className="p-3 rounded-full bg-gradient-to-br from-primary/10 to-primary/20 shadow-inner">
                    <Upload className="h-8 w-8 text-primary" />
                  </div>
                  <div>
                    <h3 className="text-lg font-medium bg-gradient-to-r from-gray-700 to-gray-900 bg-clip-text text-transparent">
                      Your PoseNet CSV Data
                    </h3>
                    <p className="text-sm text-muted-foreground mt-1">
                      Drag and drop or browse to upload your own file
                    </p>
                  </div>
                  <div className="flex gap-3">
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
                      size="sm"
                      className="transition-all duration-200 hover:shadow-md hover:bg-gradient-to-r hover:from-gray-50 hover:to-gray-100"
                    >
                      <FileUp className="mr-2 h-4 w-4" />
                      Upload CSV
                    </Button>

                    <Button
                      onClick={fetchKinectData}
                      variant="outline"
                      size="sm"
                      className="transition-all duration-200 hover:shadow-md hover:bg-gradient-to-r hover:from-blue-50 hover:to-indigo-50"
                    >
                      <Database className="mr-2 h-4 w-4 text-blue-600" />
                      Load Sample Data
                    </Button>
                  </div>
                </div>
              </div>

              {/* Error message */}
              {error && (
                <Alert
                  variant="destructive"
                  className="mt-4 border-0 shadow-md bg-gradient-to-r from-rose-50 to-red-50"
                >
                  <AlertCircle className="h-4 w-4 text-red-500" />
                  <AlertTitle className="text-red-700">Error</AlertTitle>
                  <AlertDescription className="text-red-600">
                    {error}
                  </AlertDescription>
                </Alert>
              )}

              {/* Loading progress */}
              {loading && (
                <div className="mt-4 space-y-2 p-4 rounded-lg bg-gradient-to-r from-gray-50 to-gray-100 shadow-md">
                  <div className="flex justify-between text-sm">
                    <span className="font-medium text-gray-700">
                      Processing...
                    </span>
                    <span className="font-semibold text-primary">
                      {progress}%
                    </span>
                  </div>
                  <Progress value={progress} className="h-2 bg-gray-200" />
                </div>
              )}

              {/* Data Status */}
              {(groundTruthData.length > 0 || predictedData.length > 0) && (
                <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-100">
                  <h3 className="text-lg font-medium text-blue-800 mb-2">
                    Visualization Data Status
                  </h3>
                  <div className="flex flex-col space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-gray-700">
                        True Data
                      </span>
                      <Badge
                        variant={groundTruthData.length ? "default" : "outline"}
                        className={
                          groundTruthData.length
                            ? "bg-green-500"
                            : "bg-gray-200 text-gray-700"
                        }
                      >
                        {groundTruthData.length
                          ? `${groundTruthData.length} frames`
                          : "No data"}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium text-gray-700">
                        Prediction Data (CSV Upload)
                      </span>
                      <Badge
                        variant={predictedData.length ? "default" : "outline"}
                        className={
                          predictedData.length
                            ? "bg-green-500"
                            : "bg-gray-200 text-gray-700"
                        }
                      >
                        {predictedData.length
                          ? `${predictedData.length} frames`
                          : "No data"}
                      </Badge>
                    </div>
                  </div>

                  <div className="mt-4">
                    <Button
                      onClick={() => setActiveTab("visualization")}
                      className="bg-gradient-to-r from-indigo-500 to-blue-500 hover:from-indigo-600 hover:to-blue-600 text-white shadow-md w-full"
                    >
                      <Eye className="mr-2 h-4 w-4" />
                      View 3D Visualization
                    </Button>
                  </div>
                </div>
              )}

              {/* Data table */}
              {csvData && csvData.rows.length > 0 && (
                <div className="mt-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-medium text-gray-900">
                      CSV Data Preview
                      {csvData.hasZValues && (
                        <Badge className="ml-2 bg-blue-100 text-blue-800 border-blue-200">
                          Z-Values Present
                        </Badge>
                      )}
                    </h3>
                  </div>

                  <div className="border rounded-lg overflow-auto shadow-md bg-white">
                    <h4 className="p-3 font-medium text-gray-700 border-b bg-gray-50">
                      Data (First 10 Rows)
                    </h4>
                    <div className="max-h-[400px] overflow-auto">
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
                          {csvData.rows.slice(0, 10).map((row, i) => (
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
                                  {typeof cell === "number"
                                    ? cell.toFixed(6)
                                    : cell}
                                </TableCell>
                              ))}
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                    <div className="text-sm text-gray-500 mt-2 p-2 border-t">
                      Showing 10 of {csvData.rows.length} rows - Full data will
                      be used in 3D visualization
                    </div>
                  </div>
                </div>
              )}
            </TabsContent>

            <TabsContent value="visualization" className="pt-4">
              {groundTruthData.length > 0 || predictedData.length > 0 ? (
                <div className="space-y-4" key={dataKey}>
                  {/* View mode toggle (only show when both data sets are available) */}
                  {groundTruthData.length > 0 && predictedData.length > 0 && (
                    <div className="flex items-center justify-end gap-2 mb-2">
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-medium text-gray-700">
                                Side-by-Side View
                              </span>
                              <Switch
                                checked={showSideBySide}
                                onCheckedChange={setShowSideBySide}
                              />
                            </div>
                          </TooltipTrigger>
                          <TooltipContent>
                            <p>
                              {showSideBySide
                                ? "Switch to overlapping view"
                                : "Switch to side-by-side view"}
                            </p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </div>
                  )}

                  {/* The visualization component with side-by-side capability */}
                  {groundTruthData.length > 0 && predictedData.length > 0 ? (
                    <PoseNet3DVisualization
                      poseData={groundTruthData}
                      predictedData={predictedData}
                      showSideBySide={showSideBySide}
                      groundTruthLabel="True Sample"
                      predictedLabel="Prediction"
                    />
                  ) : groundTruthData.length > 0 ? (
                    <PoseNet3DVisualization
                      poseData={groundTruthData}
                      predictedData={[]}
                      showSideBySide={false}
                      groundTruthLabel="True Sample"
                      predictedLabel=""
                    />
                  ) : (
                    <PoseNet3DVisualization
                      poseData={predictedData}
                      predictedData={[]}
                      showSideBySide={false}
                      groundTruthLabel="Prediction"
                      predictedLabel=""
                    />
                  )}
                </div>
              ) : (
                <div className="p-12 text-center bg-gray-50 rounded-lg border border-gray-200">
                  <p className="text-gray-500">
                    No data available for visualization.
                  </p>
                  <Button
                    onClick={fetchKinectData}
                    variant="outline"
                    className="mt-4"
                  >
                    <Database className="mr-2 h-4 w-4" />
                    Load Sample Data
                  </Button>
                </div>
              )}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}
