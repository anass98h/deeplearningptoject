/* eslint-disable @typescript-eslint/no-explicit-any */
"use client";

import type React from "react";

import { useState, useCallback, useRef } from "react";
import {
  Upload,
  FileUp,
  AlertCircle,
  CheckCircle2,
  BarChartIcon,
  LineChartIcon,
  Sparkles,
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface CSVData {
  headers: string[];
  rows: any[][];
  targetColumn: string;
  features: string[];
}

// Mock data for demonstration
const mockClassificationPredictions = [
  0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
];
const mockRegressionPredictions = [
  23.5, 25.2, 22.8, 26.1, 24.7, 27.3, 25.9, 28.4, 26.5, 29.1, 27.6, 30.2, 28.8,
  31.5, 29.9, 32.6, 31.1, 33.8, 32.3, 34.9,
];

// Mock models for classification and regression
const classificationModels = [
  { id: "logistic-regression", name: "Logistic Regression" },
  { id: "random-forest", name: "Random Forest" },
  { id: "svm", name: "Support Vector Machine" },
  { id: "neural-network", name: "Neural Network" },
  { id: "gradient-boosting", name: "Gradient Boosting" },
];

const regressionModels = [
  { id: "linear-regression", name: "Linear Regression" },
  { id: "decision-tree", name: "Decision Tree" },
  { id: "random-forest-reg", name: "Random Forest" },
  { id: "svr", name: "Support Vector Regression" },
  { id: "neural-network-reg", name: "Neural Network" },
];

export function CSVPrediction() {
  const [isDragging, setIsDragging] = useState(false);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [file, setFile] = useState<File | null>(null);
  const [csvData, setCsvData] = useState<CSVData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [classificationPredictions, setClassificationPredictions] = useState<
    number[] | null
  >(null);
  const [regressionPredictions, setRegressionPredictions] = useState<
    number[] | null
  >(null);
  const [classificationTime, setClassificationTime] = useState<number | null>(
    null
  );
  const [regressionTime, setRegressionTime] = useState<number | null>(null);
  const [progress, setProgress] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [activeTab, setActiveTab] = useState<string>("classification");
  const [selectedClassificationModel, setSelectedClassificationModel] =
    useState<string>("");
  const [selectedRegressionModel, setSelectedRegressionModel] =
    useState<string>("");
  const [usedClassificationModel, setUsedClassificationModel] = useState<
    string | null
  >(null);
  const [usedRegressionModel, setUsedRegressionModel] = useState<string | null>(
    null
  );

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
    setClassificationPredictions(null);
    setRegressionPredictions(null);
    setClassificationTime(null);
    setRegressionTime(null);
    setUsedClassificationModel(null);
    setUsedRegressionModel(null);

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
      const headers = lines[0].split(",").map((header) => header.trim());

      // Process rows
      const rows: any[][] = [];
      for (let i = 1; i < lines.length; i++) {
        if (lines[i].trim() === "") continue;

        const values = lines[i].split(",").map((value) => {
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
        targetColumn: headers[headers.length - 1],
        features: headers.slice(0, -1),
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

  const runClassification = useCallback(async () => {
    if (!csvData || !selectedClassificationModel) {
      setError("Please select a model before running classification");
      return;
    }

    setLoading(true);
    setClassificationPredictions(null);
    setClassificationTime(null);
    setError(null);

    // Simulate progress
    let progress = 0;
    const interval = setInterval(() => {
      progress += 2;
      setProgress(Math.min(progress, 100));
      if (progress >= 100) clearInterval(interval);
    }, 50);

    // Simulate ML processing time
    const startTime = performance.now();

    setTimeout(() => {
      setClassificationPredictions(mockClassificationPredictions);
      setUsedClassificationModel(selectedClassificationModel);

      const endTime = performance.now();
      setClassificationTime(endTime - startTime);
      setLoading(false);
      clearInterval(interval);
      setProgress(100);
    }, 2000);
  }, [csvData, selectedClassificationModel]);

  const runRegression = useCallback(async () => {
    if (!csvData || !selectedRegressionModel) {
      setError("Please select a model before running regression");
      return;
    }

    setLoading(true);
    setRegressionPredictions(null);
    setRegressionTime(null);
    setError(null);

    // Simulate progress
    let progress = 0;
    const interval = setInterval(() => {
      progress += 2;
      setProgress(Math.min(progress, 100));
      if (progress >= 100) clearInterval(interval);
    }, 50);

    // Simulate ML processing time
    const startTime = performance.now();

    setTimeout(() => {
      setRegressionPredictions(mockRegressionPredictions);
      setUsedRegressionModel(selectedRegressionModel);

      const endTime = performance.now();
      setRegressionTime(endTime - startTime);
      setLoading(false);
      clearInterval(interval);
      setProgress(100);
    }, 2000);
  }, [csvData, selectedRegressionModel]);

  const renderClassificationResults = () => {
    if (!csvData || !classificationPredictions) return null;

    return (
      <div className="mt-8">
        <div className="flex items-center mb-6">
          <h3 className="text-xl font-semibold bg-gradient-to-r from-purple-600 to-indigo-600 bg-clip-text text-transparent">
            Classification Results
          </h3>
          {usedClassificationModel && (
            <Badge
              variant="outline"
              className="ml-3 bg-gradient-to-r from-purple-50 to-indigo-50 border-purple-200"
            >
              <Sparkles className="h-3 w-3 mr-1 text-purple-500" />
              {classificationModels.find(
                (m) => m.id === usedClassificationModel
              )?.name || usedClassificationModel}
            </Badge>
          )}
        </div>
      </div>
    );
  };

  const renderRegressionResults = () => {
    if (!csvData || !regressionPredictions) return null;

    return (
      <div className="mt-8">
        <div className="flex items-center mb-6">
          <h3 className="text-xl font-semibold bg-gradient-to-r from-cyan-600 to-blue-600 bg-clip-text text-transparent">
            Regression Results
          </h3>
          {usedRegressionModel && (
            <Badge
              variant="outline"
              className="ml-3 bg-gradient-to-r from-cyan-50 to-blue-50 border-cyan-200"
            >
              <Sparkles className="h-3 w-3 mr-1 text-cyan-500" />
              {regressionModels.find((m) => m.id === usedRegressionModel)
                ?.name || usedRegressionModel}
            </Badge>
          )}
        </div>
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
          <Tabs
            value={activeTab}
            onValueChange={setActiveTab}
            className="w-full"
          >
            <TabsList className="w-full p-0 h-auto bg-gradient-to-r from-gray-50 to-gray-100 rounded-none border-b">
              <TabsTrigger
                value="classification"
                className="flex-1 py-3 rounded-none data-[state=active]:bg-white data-[state=active]:shadow-none data-[state=active]:border-b-2 data-[state=active]:border-purple-500"
              >
                <div className="flex items-center">
                  <div className="p-1 mr-2 rounded-md bg-gradient-to-br from-purple-100 to-indigo-100">
                    <BarChartIcon className="h-4 w-4 text-purple-600" />
                  </div>
                  <span className="font-medium">Classification</span>
                </div>
              </TabsTrigger>
              <TabsTrigger
                value="regression"
                className="flex-1 py-3 rounded-none data-[state=active]:bg-white data-[state=active]:shadow-none data-[state=active]:border-b-2 data-[state=active]:border-cyan-500"
              >
                <div className="flex items-center">
                  <div className="p-1 mr-2 rounded-md bg-gradient-to-br from-cyan-100 to-blue-100">
                    <LineChartIcon className="h-4 w-4 text-cyan-600" />
                  </div>
                  <span className="font-medium">Regression</span>
                </div>
              </TabsTrigger>
            </TabsList>

            <div className="p-6">
              <TabsContent value="classification" className="m-0 space-y-6">
                <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 p-4 rounded-lg bg-gradient-to-r from-purple-50 to-indigo-50 shadow-sm">
                  <h3 className="text-lg font-medium text-purple-900">
                    Classification Task
                  </h3>

                  <div className="flex flex-col md:flex-row gap-3 w-full md:w-auto">
                    <Select
                      value={selectedClassificationModel}
                      onValueChange={setSelectedClassificationModel}
                    >
                      <SelectTrigger className="w-full md:w-[220px] border-purple-200 bg-white shadow-sm">
                        <SelectValue placeholder="Select a model" />
                      </SelectTrigger>
                      <SelectContent>
                        {classificationModels.map((model) => (
                          <SelectItem key={model.id} value={model.id}>
                            {model.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>

                    <Button
                      onClick={runClassification}
                      disabled={loading || !selectedClassificationModel}
                      className="w-full md:w-auto bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 shadow-md hover:shadow-lg transition-all duration-200"
                    >
                      Run Classification
                    </Button>
                  </div>
                </div>

                <div className="border rounded-lg overflow-auto shadow-md bg-white">
                  <Table>
                    <TableHeader>
                      <TableRow className="bg-gradient-to-r from-purple-50 to-indigo-50">
                        {csvData.headers.map((header, i) => (
                          <TableHead key={i} className="font-semibold">
                            {header}
                            {i === csvData.headers.length - 1 && (
                              <Badge
                                className="ml-2 bg-purple-100 text-purple-800 hover:bg-purple-200 border-0"
                                variant="outline"
                              >
                                Target
                              </Badge>
                            )}
                          </TableHead>
                        ))}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {csvData.rows.slice(0, 5).map((row, i) => (
                        <TableRow key={i} className="hover:bg-purple-50/50">
                          {row.map((cell, j) => (
                            <TableCell key={j}>
                              {typeof cell === "number"
                                ? cell.toString()
                                : cell}
                            </TableCell>
                          ))}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>

                {/* Show prediction time if available */}
                {classificationTime && (
                  <Alert className="border-0 shadow-md bg-gradient-to-r from-green-50 to-emerald-50">
                    <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                    <AlertTitle className="text-emerald-800">
                      Classification Complete
                    </AlertTitle>
                    <AlertDescription className="text-emerald-700">
                      Time taken: {(classificationTime / 1000).toFixed(2)}{" "}
                      seconds
                    </AlertDescription>
                  </Alert>
                )}

                {/* Classification results */}
                {renderClassificationResults()}
              </TabsContent>

              <TabsContent value="regression" className="m-0 space-y-6">
                <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 p-4 rounded-lg bg-gradient-to-r from-cyan-50 to-blue-50 shadow-sm">
                  <h3 className="text-lg font-medium text-cyan-900">
                    Regression Task
                  </h3>

                  <div className="flex flex-col md:flex-row gap-3 w-full md:w-auto">
                    <Select
                      value={selectedRegressionModel}
                      onValueChange={setSelectedRegressionModel}
                    >
                      <SelectTrigger className="w-full md:w-[220px] border-cyan-200 bg-white shadow-sm">
                        <SelectValue placeholder="Select a model" />
                      </SelectTrigger>
                      <SelectContent>
                        {regressionModels.map((model) => (
                          <SelectItem key={model.id} value={model.id}>
                            {model.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>

                    <Button
                      onClick={runRegression}
                      disabled={loading || !selectedRegressionModel}
                      className="w-full md:w-auto bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-700 hover:to-blue-700 shadow-md hover:shadow-lg transition-all duration-200"
                    >
                      Run Regression
                    </Button>
                  </div>
                </div>

                <div className="border rounded-lg overflow-auto shadow-md bg-white">
                  <Table>
                    <TableHeader>
                      <TableRow className="bg-gradient-to-r from-cyan-50 to-blue-50">
                        {csvData.headers.map((header, i) => (
                          <TableHead key={i} className="font-semibold">
                            {header}
                            {i === csvData.headers.length - 1 && (
                              <Badge
                                className="ml-2 bg-cyan-100 text-cyan-800 hover:bg-cyan-200 border-0"
                                variant="outline"
                              >
                                Target
                              </Badge>
                            )}
                          </TableHead>
                        ))}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {csvData.rows.slice(0, 5).map((row, i) => (
                        <TableRow key={i} className="hover:bg-cyan-50/50">
                          {row.map((cell, j) => (
                            <TableCell key={j}>
                              {typeof cell === "number"
                                ? cell.toString()
                                : cell}
                            </TableCell>
                          ))}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>

                {/* Show prediction time if available */}
                {regressionTime && (
                  <Alert className="border-0 shadow-md bg-gradient-to-r from-green-50 to-emerald-50">
                    <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                    <AlertTitle className="text-emerald-800">
                      Regression Complete
                    </AlertTitle>
                    <AlertDescription className="text-emerald-700">
                      Time taken: {(regressionTime / 1000).toFixed(2)} seconds
                    </AlertDescription>
                  </Alert>
                )}

                {/* Regression results */}
                {renderRegressionResults()}
              </TabsContent>
            </div>
          </Tabs>
        </div>
      )}
    </div>
  );
}
