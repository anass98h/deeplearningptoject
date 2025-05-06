"use client";

import React, { useState, useCallback } from "react";
import { Scissors } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export function VideoFrameTrimmer() {
    const [videoFile, setVideoFile] = useState<File | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [downloadLink, setDownloadLink] = useState<string | null>(null);

    const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

    const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const f = e.target.files?.[0] || null;
        setVideoFile(f);
        setDownloadLink(null);
        setError(null);
    }, []);

    const runPipeline = async () => {
        if (!videoFile) return setError("Please select a video first.");

        setLoading(true);
        setError(null);

        try {
            const formData = new FormData();
            formData.append("video", videoFile);

            const res = await fetch(`${BACKEND_URL}/run-pipeline`, {
                method: "POST",
                body: formData,
            });

            const responseText = await res.text();

            if (!res.ok) {
                let errorData;
                try {
                    errorData = JSON.parse(responseText);
                } catch {
                    errorData = { detail: responseText };
                }
                throw new Error(errorData.detail || `Server error: ${res.status}`);
            }

            const blob = new Blob([responseText], { type: "text/csv" });
            const url = URL.createObjectURL(blob);
            setDownloadLink(url);

        } catch (err: any) {
            console.error("Pipeline error:", err);

            let message = err.message;
            if (err.message.includes("401")) message = "Session expired, please login again";
            if (err.message.includes("500")) message = "Server processing error";

            setError(message || "Failed to process video");
        } finally {
            setLoading(false);
        }
    };

    return (
        <Card className="shadow-lg max-w-md mx-auto p-4">
            <CardHeader>
                <CardTitle>Video Analysis Pipeline</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
                <input
                    type="file"
                    accept="video/*"
                    id="video-upload"
                    className="hidden"
                    onChange={handleFileChange}
                />
                <label
                    htmlFor="video-upload"
                    className="block w-full border-2 border-dashed p-8 text-center cursor-pointer hover:bg-gray-50 transition-colors"
                >
                    {videoFile ? `Selected: ${videoFile.name}` : "Click to upload video"}
                </label>

                <div className="flex justify-end gap-2">
                    {videoFile && (
                        <Button
                            onClick={runPipeline}
                            disabled={loading}
                            className="bg-blue-600 hover:bg-blue-700 text-white"
                        >
                            {loading ? (
                                <span className="flex items-center">
                                    <Scissors className="mr-2 h-4 w-4 animate-pulse" />
                                    Processing...
                                </span>
                            ) : (
                                <span className="flex items-center">
                                    <Scissors className="mr-2 h-4 w-4" />
                                    Start Analysis
                                </span>
                            )}
                        </Button>
                    )}
                </div>

                {error && (
                    <Alert variant="destructive" className="mt-4">
                        <AlertTitle>Error</AlertTitle>
                        <AlertDescription>{error}</AlertDescription>
                    </Alert>
                )}

                {downloadLink && (
                    <div className="mt-4 text-center">
                        <a
                            href={downloadLink}
                            download="analysis_results.csv"
                            className="inline-block bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 transition-colors"
                        >
                            Download Analysis Results
                        </a>
                    </div>
                )}
            </CardContent>
        </Card>
    );
}
