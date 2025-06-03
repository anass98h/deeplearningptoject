"use client";

import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { VideoUpload } from "@/components/video-upload";
import { ProcessingStatus } from "@/components/processing-status";
import { ResultsViewer } from "@/components/results-viewer";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Activity, Upload, FileVideo } from "lucide-react";

export default function Dashboard() {
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("upload");

  const handleJobCreated = (jobId: string) => {
    setActiveJobId(jobId);
    setActiveTab("status");
  };

  const handleJobCompleted = () => {
    setActiveTab("results");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto p-6">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-slate-900 dark:text-slate-100 mb-2">
            Exercise Video Analysis
          </h1>
          <p className="text-slate-600 dark:text-slate-400 text-lg">
            AI-powered video processing pipeline for exercise form analysis and
            scoring
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                Pipeline Stages
              </CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">7</div>
              <p className="text-xs text-muted-foreground">
                MoveNet → Quality Check → 3D Conversion → Scoring
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                Quality Gates
              </CardTitle>
              <FileVideo className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">2</div>
              <p className="text-xs text-muted-foreground">
                2D Quality Check + 3D Form Assessment
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Score Range</CardTitle>
              <Upload className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">0.0 - 5.0</div>
              <p className="text-xs text-muted-foreground">
                0.0 = Perfect, 5.0 = Poor form
              </p>
            </CardContent>
          </Card>
        </div>

        <Tabs
          value={activeTab}
          onValueChange={setActiveTab}
          className="space-y-6"
        >
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="upload" className="flex items-center gap-2">
              <Upload className="h-4 w-4" />
              Upload
            </TabsTrigger>
            <TabsTrigger value="status" className="flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Status
            </TabsTrigger>
            <TabsTrigger value="results" className="flex items-center gap-2">
              <FileVideo className="h-4 w-4" />
              Results
            </TabsTrigger>
          </TabsList>

          <TabsContent value="upload" className="space-y-6">
            <VideoUpload onJobCreated={handleJobCreated} />
          </TabsContent>

          <TabsContent value="status" className="space-y-6">
            <ProcessingStatus
              jobId={activeJobId}
              onJobCompleted={handleJobCompleted}
            />
          </TabsContent>

          <TabsContent value="results" className="space-y-6">
            <ResultsViewer jobId={activeJobId} />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
