"use client";

import React, { useState, useRef, useMemo, useCallback } from "react";
import { FileVideo, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { SkeletonRenderer } from "./SkeletonRenderer";
import { SkeletonProvider } from "./SkeletonContext";
import { useSkeletonContext } from "./SkeletonContext";

export function VideoTrimmer() {
  const [originalData, setOriginalData] = useState([]);
  const [trimmedData, setTrimmedData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [frameMappings, setFrameMappings] = useState([]);
  const [processingStatus, setProcessingStatus] = useState(null);
  const [jobId, setJobId] = useState(null);
  const fileInputRef = useRef(null);
  const pollIntervalRef = useRef(null);

  const BACKEND_URL =
    process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

  // Clear the polling interval when component unmounts
  React.useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  // Handle file selection
  const handleFileChange = async (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      await uploadVideoAndStartProcessing(file);
    }
  };

  // Handle button click to open file picker
  const handleButtonClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  // Upload video to backend and start the processing pipeline
  const uploadVideoAndStartProcessing = async (file) => {
    setLoading(true);
    setError(null);
    setProcessingStatus("Uploading video...");
    setJobId(null);

    // Clear existing data
    setOriginalData([]);
    setTrimmedData([]);
    setFrameMappings([]);

    // Clear any existing polling
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }

    try {
      // Create a form with the video file to send to backend
      const formData = new FormData();
      formData.append("file", file);

      // Send to the process-video endpoint
      const videoEndpoint = `${BACKEND_URL}/process-video`;
      console.log(`Sending video to ${videoEndpoint}`);

      // Upload the video file
      const response = await fetch(videoEndpoint, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Failed to upload video: ${response.status}`);
      }

      const data = await response.json();
      const id = data.job_id;
      setJobId(id);

      console.log(`Video uploaded successfully. Job ID: ${id}`);
      setProcessingStatus("Processing video - this may take a few minutes...");

      // Start polling for job status
      pollIntervalRef.current = setInterval(() => {
        checkJobStatus(id);
      }, 2000);
    } catch (err) {
      console.error("Error uploading video:", err);
      setError(err.message);
      setProcessingStatus(null);
      setLoading(false);
    }
  };

  // Check job status
  const checkJobStatus = async (id) => {
    try {
      const statusResponse = await fetch(`${BACKEND_URL}/job-status/${id}`);
      if (!statusResponse.ok) {
        throw new Error(`Failed to check job status: ${statusResponse.status}`);
      }

      const jobData = await statusResponse.json();
      console.log(`Job status: ${jobData.status} - ${jobData.message}`);

      // Update the processing status message
      setProcessingStatus(jobData.message);

      // Check if the job is complete or failed
      if (jobData.status === "completed" || jobData.status === "failed") {
        // Stop polling
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;

        // If completed, fetch the data
        if (jobData.status === "completed") {
          await fetchProcessedData(id);
        } else {
          setError(`Processing failed: ${jobData.message}`);
          setProcessingStatus(null);
          setLoading(false);
        }
      }
    } catch (e) {
      console.error("Error checking job status:", e);
      // Don't stop polling just because one check failed
    }
  };

  // Fetch the processed data once the job is complete
  const fetchProcessedData = async (id) => {
    try {
      setProcessingStatus("Loading data...");

      // Fetch Kinect 3D data as "original"
      const kinect3dResponse = await fetch(
        `${BACKEND_URL}/video-data/${id}/kinect3d`
      );
      if (!kinect3dResponse.ok) {
        throw new Error(
          `Failed to fetch Kinect 3D data: ${kinect3dResponse.status}`
        );
      }

      const kinect3dText = await kinect3dResponse.text();
      const kinect3dFrames = parseCSV(kinect3dText);
      setOriginalData(kinect3dFrames);

      // Fetch trimmed data
      const trimmedResponse = await fetch(
        `${BACKEND_URL}/video-data/${id}/trimmed`
      );
      if (!trimmedResponse.ok) {
        throw new Error(
          `Failed to fetch trimmed data: ${trimmedResponse.status}`
        );
      }

      const trimmedText = await trimmedResponse.text();
      const trimmedFrames = parseCSV(trimmedText);
      setTrimmedData(trimmedFrames);

      // Calculate frame mappings
      calculateFrameMappings(kinect3dFrames, trimmedFrames);

      setProcessingStatus(null);
      setLoading(false);

      console.log(
        `Data loaded successfully: Kinect3D=${kinect3dFrames.length} frames, Trimmed=${trimmedFrames.length} frames`
      );
    } catch (err) {
      console.error("Error loading processed data:", err);
      setError(err.message);
      setProcessingStatus(null);
      setLoading(false);
    }
  };

  // Parse CSV text to array of objects
  const parseCSV = useCallback((text) => {
    const lines = text.split("\n");
    if (lines.length === 0) return [];

    // Detect delimiter (tab or comma)
    const firstLine = lines[0];
    const delimiter = firstLine.includes("\t") ? "\t" : ",";

    // Parse headers
    const headers = firstLine.split(delimiter).map((h) => h.trim());

    // Parse rows to objects
    const objects = [];
    for (let i = 1; i < lines.length; i++) {
      if (!lines[i].trim()) continue;

      const values = lines[i].split(delimiter).map((v) => {
        const trimmed = v.trim();
        return isNaN(Number(trimmed)) ? trimmed : Number(trimmed);
      });

      if (values.length === headers.length) {
        const obj = {};
        headers.forEach((header, index) => {
          obj[header] = values[index];
        });
        objects.push(obj);
      }
    }

    return objects;
  }, []);

  // Calculate which original frames were kept in the trimmed dataset
  const calculateFrameMappings = useCallback(
    (originalFrames, trimmedFrames) => {
      if (!originalFrames.length || !trimmedFrames.length) return;

      // For each trimmed frame, find its corresponding index in the original dataset
      const mappings = new Array(originalFrames.length).fill(-1); // -1 means frame was removed

      // Get frame numbers for both datasets
      const originalFrameNums = originalFrames.map((frame) => frame.FrameNo);
      const trimmedFrameNums = trimmedFrames.map((frame) => frame.FrameNo);

      // Map trimmed frames to original frames
      trimmedFrameNums.forEach((frameNo, trimmedIndex) => {
        const originalIndex = originalFrameNums.indexOf(frameNo);
        if (originalIndex !== -1) {
          mappings[originalIndex] = trimmedIndex;
        }
      });

      setFrameMappings(mappings);
    },
    []
  );

  // Custom animation component that plays through ALL frames
  const CustomAnimationManager = () => {
    const { isPlaying, setCurrentFrame, currentFrame } = useSkeletonContext();

    React.useEffect(() => {
      if (!originalData || originalData.length === 0) return;

      let animationFrameId;
      let lastTime = 0;
      const frameTime = 1000 / 30; // 30 fps

      const updateFrame = (timestamp) => {
        if (!isPlaying) return;
        const elapsed = timestamp - lastTime;

        if (elapsed > frameTime) {
          lastTime = timestamp;
          setCurrentFrame((prev) => {
            // Always use original data length as the max frame count
            const nextFrame = (prev + 1) % originalData.length;
            return nextFrame;
          });
        }

        animationFrameId = requestAnimationFrame(updateFrame);
      };

      if (isPlaying) {
        animationFrameId = requestAnimationFrame(updateFrame);
      }

      return () => {
        if (animationFrameId) {
          cancelAnimationFrame(animationFrameId);
        }
      };
    }, [isPlaying, originalData.length, setCurrentFrame]);

    // No need to render anything
    return null;
  };

  // Optimized timeline renderer that uses fixed-size blocks instead of flex
  const FrameTimeline = () => {
    const { currentFrame, setCurrentFrame } = useSkeletonContext();

    // Optimize rendering of the timeline by limiting the number of blocks
    const MAX_VISIBLE_BLOCKS = 100; // Maximum number of blocks to render for performance

    // Calculate which frames to show in the timeline
    const timelineFrames = useMemo(() => {
      if (!originalData.length || !frameMappings.length) return [];

      if (originalData.length <= MAX_VISIBLE_BLOCKS) {
        // Show all frames if we have fewer than the max
        return Array.from({ length: originalData.length }, (_, i) => ({
          index: i,
          isKept: frameMappings[i] !== -1,
        }));
      } else {
        // For large datasets, sample frames to show
        const step = Math.ceil(originalData.length / MAX_VISIBLE_BLOCKS);
        const frames = [];

        for (let i = 0; i < originalData.length; i += step) {
          // For each block, determine if most frames in this block are kept or removed
          let keptCount = 0;
          for (let j = i; j < Math.min(i + step, originalData.length); j++) {
            if (frameMappings[j] !== -1) keptCount++;
          }

          frames.push({
            index: i,
            isKept: keptCount > step / 2,
            blockSize: step,
          });
        }

        return frames;
      }
    }, [originalData.length, frameMappings]);

    // Don't render if no data
    if (!timelineFrames.length) return null;

    return (
      <div className="mb-2 mt-4">
        <div className="text-sm font-medium text-gray-700 mb-1 flex justify-between">
          <span>Frame Timeline</span>
          <span>
            {currentFrame + 1} / {originalData.length}
          </span>
        </div>

        <div className="relative">
          {/* Base track */}
          <div className="h-10 bg-gray-100 rounded-md overflow-hidden flex">
            {timelineFrames.map(({ index, isKept, blockSize = 1 }) => {
              // Highlight current frame or block
              const isCurrentFrame =
                currentFrame >= index &&
                currentFrame < index + (blockSize || 1);

              // Simplified tooltip for performance
              return (
                <div
                  key={index}
                  className={`h-full border-r border-white ${isKept
                      ? "bg-green-200 hover:bg-green-300"
                      : "bg-red-200 hover:bg-red-300"
                    } ${isCurrentFrame ? "ring-2 ring-blue-500 ring-inset" : ""}`}
                  style={{
                    width: `${((blockSize || 1) / originalData.length) * 100}%`,
                    cursor: "pointer",
                  }}
                  onClick={() => setCurrentFrame(index)}
                  title={`Frame ${index + 1}: ${isKept ? "Kept" : "Removed"}`}
                />
              );
            })}
          </div>

          {/* Current position indicator */}
          <div
            className="absolute top-0 w-1 h-10 bg-blue-600"
            style={{
              left: `${(currentFrame / Math.max(1, originalData.length - 1)) * 100
                }%`,
              transform: "translateX(-50%)",
            }}
          />
        </div>
      </div>
    );
  };

  // Optimized frame status display
  const FrameStatusBadge = () => {
    const { currentFrame } = useSkeletonContext();

    if (!frameMappings.length || currentFrame >= frameMappings.length)
      return null;

    const isKept = frameMappings[currentFrame] !== -1;
    const trimmedIndex = frameMappings[currentFrame];

    return (
      <div className="mt-2 flex justify-center">
        <Badge
          className={`px-3 py-1 ${isKept
              ? "bg-green-100 text-green-800 border-green-200"
              : "bg-red-100 text-red-800 border-red-200"
            }`}
        >
          {isKept
            ? `Frame kept (Kinect 3D #${currentFrame + 1} → Trimmed #${trimmedIndex + 1
            })`
            : `Frame removed (Kinect 3D #${currentFrame + 1})`}
        </Badge>
      </div>
    );
  };

  // Custom controls component to replace the standard SkeletonControls
  const CustomSkeletonControls = () => {
    const {
      isPlaying,
      setIsPlaying,
      currentFrame,
      setCurrentFrame,
      autoRotate,
      toggleAutoRotate,
    } = useSkeletonContext();

    // Only use original data length for max frames
    const maxFrames = originalData.length;

    // Handle frame scrubbing
    const handleFrameChange = (value) => {
      const frameIndex = Math.min(Math.max(0, value[0]), maxFrames - 1);
      setCurrentFrame(frameIndex);
    };

    return (
      <div className="flex flex-col space-y-4">
        {/* Animation controls */}
        <div className="flex items-center justify-between">
          <div className="flex space-x-4">
            <Button
              variant="outline"
              size="icon"
              onClick={() => setIsPlaying(!isPlaying)}
              className="transition-all duration-200 hover:bg-blue-50"
            >
              {isPlaying ? (
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="h-5 w-5 text-blue-600"
                >
                  <rect x="6" y="4" width="4" height="16" />
                  <rect x="14" y="4" width="4" height="16" />
                </svg>
              ) : (
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="h-5 w-5 text-blue-600"
                >
                  <circle cx="12" cy="12" r="10" />
                  <polygon points="10 8 16 12 10 16 10 8" />
                </svg>
              )}
            </Button>

            <Button
              variant="outline"
              onClick={toggleAutoRotate}
              className={`transition-all duration-200 ${autoRotate ? "bg-blue-100 text-blue-700" : "hover:bg-blue-50"
                }`}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="h-4 w-4 mr-2"
              >
                <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                <path d="M3 3v5h5" />
              </svg>
              {autoRotate ? "Rotation On" : "Rotation Off"}
            </Button>
          </div>

          <div className="text-sm text-gray-600">
            Frame: {currentFrame + 1} / {maxFrames}
          </div>
        </div>

        {/* Frame slider - always use original data length */}
        <div className="px-1">
          <Slider
            value={[currentFrame]}
            min={0}
            max={Math.max(0, maxFrames - 1)}
            step={1}
            onValueChange={handleFrameChange}
            className="cursor-pointer"
          />
        </div>
      </div>
    );
  };

  // Optimized visualization component
  const FrameVisualizer = () => {
    const { currentFrame } = useSkeletonContext();

    // Memoized calculation of which trimmed frame to show
    const trimmedFrameToShow = useMemo(() => {
      if (
        !frameMappings.length ||
        currentFrame >= frameMappings.length ||
        !trimmedData.length
      )
        return null;

      const trimmedIndex = frameMappings[currentFrame];
      if (trimmedIndex === -1 || trimmedIndex >= trimmedData.length)
        return null;

      return [trimmedData[trimmedIndex]];
    }, [currentFrame, frameMappings, trimmedData]);

    // Memoized calculation of which original frame to show
    const originalFrameToShow = useMemo(() => {
      if (!originalData.length || currentFrame >= originalData.length)
        return null;

      return [originalData[currentFrame]];
    }, [currentFrame, originalData]);

    return (
      <>
        {/* Visual frame timeline */}
        <Card className="shadow-md mb-6">
          <CardContent className="pt-6">
            <FrameTimeline />

            {/* Status of current frame */}
            <FrameStatusBadge />

            {/* Custom controls that use original data length */}
            <div className="mt-4">
              <CustomSkeletonControls />
            </div>
          </CardContent>
        </Card>

        {/* Skeletons side by side */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Original data visualization */}
          <Card className="shadow-md">
            <CardContent className="pt-6">
              <div className="mb-4 flex justify-between items-center">
                <h3 className="text-lg font-medium text-gray-800">
                  Kinect 3D Data
                </h3>
                <Badge variant="outline" className="bg-blue-50 text-blue-700">
                  Frame {currentFrame + 1} / {originalData.length}
                </Badge>
              </div>
              {originalFrameToShow && (
                <SkeletonRenderer
                  poseData={originalFrameToShow}
                  isGroundTruth={true}
                  label="Kinect 3D"
                />
              )}
            </CardContent>
          </Card>

          {/* Trimmed data visualization */}
          <Card className="shadow-md">
            <CardContent className="pt-6">
              <div className="mb-4 flex justify-between items-center">
                <h3 className="text-lg font-medium text-gray-800">
                  Trimmed Data
                </h3>
                {frameMappings[currentFrame] !== -1 ? (
                  <Badge
                    variant="outline"
                    className="bg-green-50 text-green-700"
                  >
                    Frame {frameMappings[currentFrame] + 1} /{" "}
                    {trimmedData.length}
                  </Badge>
                ) : (
                  <Badge variant="outline" className="bg-red-50 text-red-700">
                    Frame removed
                  </Badge>
                )}
              </div>
              {trimmedFrameToShow ? (
                <SkeletonRenderer
                  poseData={trimmedFrameToShow}
                  isGroundTruth={false}
                  label="Trimmed"
                />
              ) : (
                <div className="h-64 flex items-center justify-center bg-gray-50 border rounded-lg">
                  <p className="text-gray-500">
                    This frame was removed during trimming
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </>
    );
  };

  return (
    <div className="space-y-6">
      {/* File upload card */}
      <Card className="shadow-md">
        <CardContent className="pt-6">
          <div className="flex justify-center gap-4 mb-6">
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              className="hidden"
              onChange={handleFileChange}
            />
            <Button
              onClick={handleButtonClick}
              className="bg-blue-600 hover:bg-blue-700 text-white"
              disabled={loading}
            >
              <FileVideo className="mr-2 h-4 w-4" />
              Upload Video File
            </Button>
          </div>

          {/* Processing status */}
          {loading && processingStatus && (
            <div className="mt-4 flex items-center justify-center">
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600 mr-2"></div>
              <span className="text-sm text-gray-600">{processingStatus}</span>
            </div>
          )}

          {/* Error alert */}
          {error && (
            <Alert variant="destructive" className="mt-4">
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* Status indicators */}
          {originalData.length > 0 && (
            <div className="mt-4 flex gap-4 justify-center">
              <Badge
                variant="outline"
                className="bg-blue-50 text-blue-700 border-blue-200 px-3 py-1"
              >
                Kinect 3D: {originalData.length} frames
              </Badge>

              {trimmedData.length > 0 && (
                <Badge
                  variant="outline"
                  className="bg-green-50 text-green-700 border-green-200 px-3 py-1"
                >
                  Trimmed: {trimmedData.length} frames (
                  {originalData.length > 0
                    ? Math.round(
                      ((originalData.length - trimmedData.length) /
                        originalData.length) *
                      100
                    )
                    : 0}
                  % removed)
                </Badge>
              )}
            </div>
          )}

          {/* Frame reduction information */}
          {originalData.length > 0 && trimmedData.length > 0 && (
            <div className="mt-4 text-sm text-gray-600 flex items-center justify-center">
              <Info className="h-4 w-4 mr-1 text-blue-500" />
              <span>
                Green segments represent frames that were kept, red segments
                were removed
              </span>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Visualization with single SkeletonProvider */}
      {originalData.length > 0 && trimmedData.length > 0 && (
        <SkeletonProvider>
          {/* Custom Animation Manager */}
          <CustomAnimationManager />

          <FrameVisualizer />
        </SkeletonProvider>
      )}
    </div>
  );
}
