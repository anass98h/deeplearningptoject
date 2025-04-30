"use client";

import React, { useState, useRef, useMemo, useCallback } from "react";
import { FileUp, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { SkeletonRenderer } from "./SkeletonRenderer";
import { SkeletonProvider } from "./SkeletonContext";
import { useSkeletonContext } from "./SkeletonContext";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

export function FrameTrimmer() {
  const [originalData, setOriginalData] = useState([]);
  const [trimmedData, setTrimmedData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [frameMappings, setFrameMappings] = useState([]);
  const fileInputRef = useRef(null);

  const BACKEND_URL =
    process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

  // Handle file selection
  const handleFileChange = async (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      await uploadAndGetTrimmed(file);
    }
  };

  // Handle button click to open file picker
  const handleButtonClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  // Upload file to backend and get trimmed version
  const uploadAndGetTrimmed = async (file) => {
    setLoading(true);
    setError(null);

    try {
      // First, read the file locally to get original data
      const originalText = await file.text();
      const originalFrames = parseCSV(originalText);
      setOriginalData(originalFrames);

      // Create a form with the CSV file to send to backend
      const formData = new FormData();
      formData.append("file", file);

      // Send to the trim-frames endpoint
      const trimEndpoint = `${BACKEND_URL}/trim-frames`;
      console.log(`Sending file to ${trimEndpoint}`);

      const response = await fetch(trimEndpoint, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Failed to trim frames: ${response.status}`);
      }

      // Parse the trimmed CSV text directly from the response
      const trimmedText = await response.text();
      const trimmedFrames = parseCSV(trimmedText);
      setTrimmedData(trimmedFrames);

      // Calculate frame mappings to show which frames were kept/removed
      calculateFrameMappings(originalFrames, trimmedFrames);

      console.log(
        `Successfully processed: Original=${originalFrames.length} frames, Trimmed=${trimmedFrames.length} frames`
      );
    } catch (err) {
      console.error("Error processing file:", err);
      setError(err.message);

      // Fallback to kinect-data endpoint if trim-frames fails
      try {
        console.log("Falling back to kinect-data endpoint");
        const fallbackResponse = await fetch(`${BACKEND_URL}/kinect-data`);

        if (!fallbackResponse.ok) {
          throw new Error(`Fallback also failed: ${fallbackResponse.status}`);
        }

        const data = await fallbackResponse.json();

        if (data.content) {
          const trimmedFrames = parseCSV(data.content);
          setTrimmedData(trimmedFrames);

          // Calculate frame mappings if we have both datasets
          if (originalData.length > 0) {
            calculateFrameMappings(originalData, trimmedFrames);
          }

          console.log(
            `Fallback successful: Trimmed=${trimmedFrames.length} frames`
          );
        }
      } catch (fallbackErr) {
        console.error("Fallback also failed:", fallbackErr);
      }
    } finally {
      setLoading(false);
    }
  };

  // Parse CSV text to array of objects - optimized version
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
                  className={`h-full border-r border-white ${
                    isKept
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
              left: `${
                (currentFrame / Math.max(1, originalData.length - 1)) * 100
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
          className={`px-3 py-1 ${
            isKept
              ? "bg-green-100 text-green-800 border-green-200"
              : "bg-red-100 text-red-800 border-red-200"
          }`}
        >
          {isKept
            ? `Frame kept (Original #${currentFrame + 1} → Trimmed #${
                trimmedIndex + 1
              })`
            : `Frame removed (Original #${currentFrame + 1})`}
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
              className={`transition-all duration-200 ${
                autoRotate ? "bg-blue-100 text-blue-700" : "hover:bg-blue-50"
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
                  Original Data
                </h3>
                <Badge variant="outline" className="bg-blue-50 text-blue-700">
                  Frame {currentFrame + 1} / {originalData.length}
                </Badge>
              </div>
              {originalFrameToShow && (
                <SkeletonRenderer
                  poseData={originalFrameToShow}
                  isGroundTruth={true}
                  label="Original"
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
                <div className="h-[600px] flex items-center justify-center bg-gray-50 border rounded-lg">
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
              accept=".csv"
              className="hidden"
              onChange={handleFileChange}
            />
            <Button
              onClick={handleButtonClick}
              className="bg-blue-600 hover:bg-blue-700 text-white"
              disabled={loading}
            >
              <FileUp className="mr-2 h-4 w-4" />
              Upload CSV File
            </Button>
          </div>

          {/* Error alert */}
          {error && (
            <Alert variant="destructive" className="mt-4">
              <AlertTitle>Error</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* Status indicators */}
          {(originalData.length > 0 || trimmedData.length > 0) && (
            <div className="mt-4 flex gap-4 justify-center">
              {originalData.length > 0 && (
                <Badge
                  variant="outline"
                  className="bg-blue-50 text-blue-700 border-blue-200 px-3 py-1"
                >
                  Original: {originalData.length} frames
                </Badge>
              )}
              {trimmedData.length > 0 && (
                <Badge
                  variant="outline"
                  className="bg-green-50 text-green-700 border-green-200 px-3 py-1"
                >
                  Trimmed: {trimmedData.length} frames (
                  {Math.round(
                    ((originalData.length - trimmedData.length) /
                      originalData.length) *
                      100
                  )}
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
