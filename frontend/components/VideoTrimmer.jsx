"use client";

import React, { useState, useRef, useMemo, useCallback, useEffect } from "react";
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
  const [statusLog, setStatusLog] = useState([]);
  const [showModal, setShowModal] = useState(false);
  const [showTipsBox, setShowTipsBox] = useState(false);

  const suggestions = [
    "Position the camera at hip level",
    "Make sure the environment is well lit",
    "Keep your back straight during the squat",
    "Make sure your entire body is in the frame",
  ];

  const tips = [
    "Light it right: Film in a bright, evenly lit room. Face the main light source and avoid strong back-lighting or harsh shadows",
    "You, and only you: Make sure no other people wander into the frame",
    "Full-body frame: Position the camera far enough back so your entire pose—from head to toes—stays visible at all times",
    'Stand out from the backdrop: Wear clothing that contrasts with your background; skip patterns or colors that "camouflage"',
    "Keep it steady: Prop the phone on a tripod or stable surface—no shaky handheld shots",
    "Landscape preferred: Hold the phone horizontally unless we specifically ask for vertical video",
    "Do a 3-second test clip: Check that lighting, framing, and focus look good before your full take",
  ];

  const [finalScore, setFinalScore] = useState(null);
  const [scoreInterpretation, setScoreInterpretation] = useState("");
  const [advancedAnalysis, setAdvancedAnalysis] = useState(null);

  const [worstConfidenceWindow, setWorstConfidenceWindow] = useState(null);
  const [mostJitteryWindow, setMostJitteryWindow] = useState(null);
  const [worstPart, setWorstPart] = useState(null);

  const BACKEND_URL =
    process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  const handleFileChange = async (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      await uploadVideoAndStartProcessing(file);
    }
  };

  const handleButtonClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const uploadVideoAndStartProcessing = async (file) => {
    setLoading(true);
    setError(null);
    setProcessingStatus("Uploading video...");
    setStatusLog(["Uploading video..."]);

    setJobId(null);
    setOriginalData([]);
    setTrimmedData([]);
    setFrameMappings([]);

    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }

    try {
      const formData = new FormData();
      formData.append("file", file);

      const videoEndpoint = `${BACKEND_URL}/process-video`;
      console.log(`Sending video to ${videoEndpoint}`);

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

  const checkJobStatus = async (id) => {
    try {
      const statusResponse = await fetch(`${BACKEND_URL}/job-status/${id}`);
      if (!statusResponse.ok) {
        throw new Error(`Failed to check job status: ${statusResponse.status}`);
      }
      const jobData = await statusResponse.json();
      setProcessingStatus(jobData.message);
      setStatusLog((prev) => [...prev, jobData.message]);


      if (jobData.status === "completed") {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
        await fetchProcessedData(id);
        return;
      }


      if (jobData.status === "failed") {
        const adv = jobData.quality_scores?.advanced_analysis;
        if (adv?.worst_confidence_windows?.length > 0) {
          setWorstConfidenceWindow(adv.worst_confidence_windows[0]);
        } else {
          setWorstConfidenceWindow({ worst_block: "–" });
        }
        if (adv?.most_jittery_windows?.length > 0) {
          setMostJitteryWindow(adv.most_jittery_windows[0]);
        } else {
          setMostJitteryWindow({ worst_block: "–" });
        }

        setError(`Processing failed: ${jobData.message}`);
        setProcessingStatus(null);
        setLoading(false);
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
        return;
      }
    } catch (e) {
      console.error("Error checking job status:", e);
    }
  };

  const fetchProcessedData = async (id) => {
    console.log("▶️ Entrato in fetchProcessedData con id:", id);
    try {
      setProcessingStatus("Loading data...");

      console.log("   ⬇️ Chiamata a /video-data/{id}/kinect3d …");
      const kinect3dResponse = await fetch(`${BACKEND_URL}/video-data/${id}/kinect3d`);
      console.log("   📥 /kinect3d status:", kinect3dResponse.status);
      if (!kinect3dResponse.ok) {
        throw new Error(`Failed to fetch Kinect 3D data: ${kinect3dResponse.status}`);
      }

      const kinect3dText = await kinect3dResponse.text();
      const kinect3dFrames = parseCSV(kinect3dText);
      setOriginalData(kinect3dFrames);

      console.log("   ⬇️ Chiamata a /video-data/{id}/trimmed …");
      const trimmedResponse = await fetch(`${BACKEND_URL}/video-data/${id}/trimmed`);
      console.log("   📥 /trimmed status:", trimmedResponse.status);
      if (!trimmedResponse.ok) {
        throw new Error(`Failed to fetch trimmed data: ${trimmedResponse.status}`);
      }

      const trimmedText = await trimmedResponse.text();
      const trimmedFrames = parseCSV(trimmedText);
      setTrimmedData(trimmedFrames);

      calculateFrameMappings(kinect3dFrames, trimmedFrames);

      console.log("   ⬇️ Chiamata a /video-data/{id}/final …");
      const finalRes = await fetch(`${BACKEND_URL}/video-data/${id}/final`);
      console.log("   📥 /final status:", finalRes.status);
      if (!finalRes.ok) {
        throw new Error(`Failed to fetch final results: ${finalRes.status}`);
      }

      const finalJson = await finalRes.json();

      console.log("   📦 finalJson keys:", Object.keys(finalJson));

      console.log("   📦 finalJson.data_formats:", finalJson.data_formats);
      console.log("   📦 finalJson.skeleton_data:", finalJson.skeleton_data);
      console.log("   📦 finalJson.quality_scores:", finalJson.quality_scores);
      console.log("   📦 finalJson.quality_scores.advanced_analysis:", finalJson.quality_scores.advanced_analysis);


      const wcwArray =
        finalJson.quality_scores?.advanced_analysis?.worst_confidence_windows;
      const mjwArray =
        finalJson.quality_scores?.advanced_analysis?.most_jittery_windows;

      console.log("   ▶️ Estraggo wcwArray da advanced_analysis:", wcwArray);
      console.log("   ▶️ Estraggo mjwArray da advanced_analysis:", mjwArray);

      const wcwCondition = Array.isArray(wcwArray) && wcwArray.length > 0;
      const mjwCondition = Array.isArray(mjwArray) && mjwArray.length > 0;

      console.log("   📦 wcwCondition result:", wcwCondition);
      console.log("   📦 mjwCondition result:", mjwCondition);

      if (wcwCondition) {
        console.log("   ✅ Setting worstConfidenceWindow a:", wcwArray[0]);
        setWorstConfidenceWindow(wcwArray[0]);
      } else {
        console.log("   ❌ Setting worstConfidenceWindow a null");
        setWorstConfidenceWindow(null);
      }

      if (mjwCondition) {
        console.log("   ✅ Setting mostJitteryWindow a:", mjwArray[0]);
        setMostJitteryWindow(mjwArray[0]);
      } else {
        console.log("   ❌ Setting mostJitteryWindow a null");
        setMostJitteryWindow(null);
      }


      setFinalScore(finalJson.quality_scores.final_exercise_score);
      setScoreInterpretation(finalJson.quality_scores.score_interpretation);
      setWorstPart(finalJson.worst_part);
      setAdvancedAnalysis(finalJson.quality_scores.advanced_analysis);

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

  const parseCSV = useCallback((text) => {
    const lines = text.split("\n");
    if (lines.length === 0) return [];

    const firstLine = lines[0];
    const delimiter = firstLine.includes("\t") ? "\t" : ",";
    const headers = firstLine.split(delimiter).map((h) => h.trim());

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

  const calculateFrameMappings = useCallback((originalFrames, trimmedFrames) => {
    if (!originalFrames.length || !trimmedFrames.length) return;

    const mappings = new Array(originalFrames.length).fill(-1);
    const originalFrameNums = originalFrames.map((frame) => frame.FrameNo);
    const trimmedFrameNums = trimmedFrames.map((frame) => frame.FrameNo);

    trimmedFrameNums.forEach((frameNo, trimmedIndex) => {
      const originalIndex = originalFrameNums.indexOf(frameNo);
      if (originalIndex !== -1) {
        mappings[originalIndex] = trimmedIndex;
      }
    });

    setFrameMappings(mappings);
  }, []);

  const CustomAnimationManager = () => {
    const { isPlaying, setCurrentFrame, currentFrame } = useSkeletonContext();

    useEffect(() => {
      if (!originalData || originalData.length === 0) return;

      let animationFrameId;
      let lastTime = 0;
      const frameTime = 1000 / 30;

      const updateFrame = (timestamp) => {
        if (!isPlaying) return;
        const elapsed = timestamp - lastTime;
        if (elapsed > frameTime) {
          lastTime = timestamp;
          setCurrentFrame((prev) => {
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

    return null;
  };

  const FrameTimeline = () => {
    const { currentFrame, setCurrentFrame } = useSkeletonContext();

    const MAX_VISIBLE_BLOCKS = 100;
    const timelineFrames = useMemo(() => {
      if (!originalData.length || !frameMappings.length) return [];

      if (originalData.length <= MAX_VISIBLE_BLOCKS) {
        return Array.from({ length: originalData.length }, (_, i) => ({
          index: i,
          isKept: frameMappings[i] !== -1,
        }));
      } else {
        const step = Math.ceil(originalData.length / MAX_VISIBLE_BLOCKS);
        const frames = [];
        for (let i = 0; i < originalData.length; i += step) {
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

    if (!timelineFrames.length) return null;

    return (
      <div className="mb-2 mt-4">
        <div className="text-sm font-medium text-white-700 mb-1 flex justify-between">
          <span>Frame Timeline</span>
          <span>
            {currentFrame + 1} / {originalData.length}
          </span>
        </div>
        <div className="relative">
          <div className="h-10 bg-white-100 rounded-md overflow-hidden flex">
            {timelineFrames.map(({ index, isKept, blockSize = 1 }) => {
              const isCurrentFrame =
                currentFrame >= index && currentFrame < index + (blockSize || 1);
              return (
                <div
                  key={index}
                  className={`h-full border-r border-blue-200 ${isKept ? "bg-green-200 hover:bg-green-300" : "bg-red-200 hover:bg-red-300"
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
          <div
            className="absolute top-0 w-1 h-10 bg-blue-600"
            style={{
              left: `${(currentFrame / Math.max(1, originalData.length - 1)) * 100}%`,
              transform: "translateX(-50%)",
            }}
          />
        </div>
      </div>
    );
  };

  const FrameStatusBadge = () => {
    const { currentFrame } = useSkeletonContext();
    if (!frameMappings.length || currentFrame >= frameMappings.length) return null;
    const isKept = frameMappings[currentFrame] !== -1;
    const trimmedIndex = frameMappings[currentFrame];
    return (
      <div className="mt-2 flex justify-center">
        <Badge
          className={`px-3 py-1 ${isKept ? "bg-green-100 text-green-800 border-green-200" : "bg-red-100 text-red-800 border-red-200"
            }`}
        >
          {isKept
            ? `Frame kept (Kinect 3D #${currentFrame + 1} → Trimmed #${trimmedIndex + 1})`
            : `Frame removed (Kinect 3D #${currentFrame + 1})`}
        </Badge>
      </div>
    );
  };

  const CustomSkeletonControls = () => {
    const {
      isPlaying,
      setIsPlaying,
      currentFrame,
      setCurrentFrame,
      autoRotate,
      toggleAutoRotate,
    } = useSkeletonContext();

    const maxFrames = originalData.length;
    const handleFrameChange = (value) => {
      const frameIndex = Math.min(Math.max(0, value[0]), maxFrames - 1);
      setCurrentFrame(frameIndex);
    };

    return (
      <div className="flex flex-col space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex space-x-4">
            <Button
              onClick={() => setIsPlaying(!isPlaying)}
              className="bg-blue-600 text-white hover:bg-blue-700 transition-all duration-200 flex items-center space-x-2 px-4 py-2"
            >
              {isPlaying ? (
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="h-5 w-5"
                >
                  <rect x="6" y="4" width="4" height="16" />
                  <rect x="14" y="4" width="4" height="16" />
                </svg>
              ) : (
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="h-5 w-5"
                >
                  <circle cx="12" cy="12" r="10" />
                  <polygon points="10 8 16 12 10 16" />
                </svg>
              )}
              <span>{isPlaying ? "Pause" : "Play"}</span>
            </Button>

            <Button
              onClick={toggleAutoRotate}
              className={`transition-all duration-200 ${autoRotate ? "bg-blue-700" : "bg-blue-600 hover:bg-blue-700"
                } text-white flex items-center space-x-2 px-4 py-2`}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="h-5 w-5"
              >
                <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                <path d="M3 3v5h5" />
              </svg>
              <span>{autoRotate ? "Rotation On" : "Rotation Off"}</span>
            </Button>
          </div>
          <div className="text-sm text-gray-200">
            Frame: {currentFrame + 1} / {maxFrames}
          </div>
        </div>
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

  const FrameVisualizer = () => {
    const { currentFrame } = useSkeletonContext();

    const trimmedFrameToShow = useMemo(() => {
      if (!frameMappings.length || currentFrame >= frameMappings.length || !trimmedData.length)
        return null;
      const trimmedIndex = frameMappings[currentFrame];
      if (trimmedIndex === -1 || trimmedIndex >= trimmedData.length) return null;
      return [trimmedData[trimmedIndex]];
    }, [currentFrame, frameMappings, trimmedData]);

    const originalFrameToShow = useMemo(() => {
      if (!originalData.length || currentFrame >= originalData.length) return null;
      return [originalData[currentFrame]];
    }, [currentFrame, originalData]);

    return (
      <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
        {/* Main Content Area */}
        <div className="xl:col-span-4 space-y-6">

          {/* Controls Section */}
          <Card className="shadow-md">
            <CardContent className="pt-6">
              <FrameTimeline />
              <FrameStatusBadge />
              <div className="mt-4">
                <CustomSkeletonControls />
              </div>
            </CardContent>
          </Card>

          {/* Video Diagnostics Section */}
          <div className="space-y-6">
            <div className="flex items-center space-x-3">
              <div className="w-1 h-8 bg-gradient-to-b from-red-500 to-purple-500 rounded-full"></div>
              <h3 className="text-2xl font-bold text-white bg-gradient-to-r from-red-400 to-purple-400 bg-clip-text text-transparent">
                Video Diagnostics
              </h3>
            </div>

            {loading ? (
              <Card className="shadow-lg border border-gray-700 bg-gray-800/50 backdrop-blur-sm">
                <CardContent className="pt-6 pb-6">
                  <div className="flex items-center justify-center space-x-3">
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-400"></div>
                    <p className="text-gray-300 font-medium">Analyzing video quality...</p>
                  </div>
                </CardContent>
              </Card>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Worst Confidence Window Card */}
                <Card className="group shadow-xl border-2 border-red-500/30 bg-gradient-to-br from-red-900/20 to-red-800/10 backdrop-blur-sm hover:border-red-400/50 transition-all duration-300 hover:shadow-2xl hover:shadow-red-500/20">
                  <CardContent className="pt-4 pb-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-2">
                        <div className="relative">
                          <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
                          <div className="absolute inset-0 w-3 h-3 bg-red-500 rounded-full animate-ping opacity-30"></div>
                        </div>
                        <div>
                          <h4 className="text-base font-bold text-white group-hover:text-red-300 transition-colors">
                            Worst Confidence
                          </h4>
                          <p className="text-xs text-gray-400">
                            Least reliable detection
                          </p>
                        </div>
                      </div>
                      <div className="bg-red-500/20 p-1.5 rounded-lg">
                        <svg className="w-4 h-4 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
                        </svg>
                      </div>
                    </div>

                    {worstConfidenceWindow ? (
                      <div className="bg-black/30 rounded-lg p-3 border border-red-500/20">
                        <div className="text-center">
                          <div className="text-xl font-bold text-red-300 mb-1">
                            {worstConfidenceWindow.worst_block}
                          </div>
                          <div className="text-xs text-gray-400">
                            Critical detection area
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-600 text-center">
                        <div className="text-gray-400 mb-1">
                          <svg className="w-6 h-6 mx-auto mb-1 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 12h6m-6-4h6m2 5.291A7.962 7.962 0 0112 15c-2.034 0-3.9.785-5.291 2.09M8.051 21.951A9.957 9.957 0 0112 22c1.695 0 3.276-.42 4.673-1.16" />
                          </svg>
                        </div>
                        <p className="text-xs text-gray-400">No confidence data available</p>
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* Most Jittery Window Card */}
                <Card className="group shadow-xl border-2 border-purple-500/30 bg-gradient-to-br from-purple-900/20 to-purple-800/10 backdrop-blur-sm hover:border-purple-400/50 transition-all duration-300 hover:shadow-2xl hover:shadow-purple-500/20">
                  <CardContent className="pt-4 pb-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-2">
                        <div className="relative">
                          <div className="w-3 h-3 bg-purple-500 rounded-full animate-bounce"></div>
                          <div className="absolute inset-0 w-3 h-3 bg-purple-500 rounded-full animate-ping opacity-30"></div>
                        </div>
                        <div>
                          <h4 className="text-base font-bold text-white group-hover:text-purple-300 transition-colors">
                            Most Jittery
                          </h4>
                          <p className="text-xs text-gray-400">
                            Highest movement instability
                          </p>
                        </div>
                      </div>
                      <div className="bg-purple-500/20 p-1.5 rounded-lg">
                        <svg className="w-4 h-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                      </div>
                    </div>

                    {mostJitteryWindow ? (
                      <div className="bg-black/30 rounded-lg p-3 border border-purple-500/20">
                        <div className="text-center">
                          <div className="text-xl font-bold text-purple-300 mb-1">
                            {mostJitteryWindow.worst_block}
                          </div>
                          <div className="text-xs text-gray-400">
                            Peak instability detected
                          </div>
                        </div>

                        {/* Movement indicator */}
                        <div className="flex items-center justify-center space-x-1.5 mt-3">
                          <div className="w-1.5 h-1.5 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                          <div className="w-1.5 h-1.5 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                          <div className="w-1.5 h-1.5 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                        </div>
                      </div>
                    ) : (
                      <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-600 text-center">
                        <div className="text-gray-400 mb-1">
                          <svg className="w-6 h-6 mx-auto mb-1 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 00-2-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                          </svg>
                        </div>
                        <p className="text-xs text-gray-400">No jitter data available</p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            )}

            {/* Suggestions Button */}
            <div className="flex justify-center">
              <Button
                onClick={() => setShowModal(true)}
                className="bg-blue-600 text-white hover:bg-blue-700 px-6 py-2 rounded-lg"
              >
                Show Suggestions
              </Button>
            </div>
          </div>

          {/* 3D Visualization Section */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="shadow-md">
              <CardContent className="pt-6">
                <div className="mb-4 flex justify-between items-center">
                  <h3 className="text-lg font-medium text-white">3D Data</h3>
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

            <Card className="shadow-md">
              <CardContent className="pt-6">
                <div className="mb-4 flex justify-between items-center">
                  <h3 className="text-lg font-medium text-white">Trimmed Data</h3>
                  {frameMappings[currentFrame] !== -1 ? (
                    <Badge variant="outline" className="bg-green-50 text-green-700">
                      Frame {frameMappings[currentFrame] + 1} / {trimmedData.length}
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
                  <div className="flex items-center justify-center bg-gray-50 border rounded-lg min-h-[200px]">
                    <p className="text-gray-500">
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Right Sidebar */}
        <div className="xl:col-span-1 space-y-4">
          {/* Score Interpretations Card */}
          <Card className="shadow-md border border-white rounded-lg sticky top-6">
            <CardContent className="pt-4 pb-4">
              <h3 className="text-lg font-semibold text-white mb-3">
                Score Interpretations
              </h3>
              <ul className="space-y-1.5">
                <li className="flex items-center space-x-2 text-xs text-white">
                  <span className="w-2.5 h-2.5 bg-green-500 rounded-full block flex-shrink-0" />
                  <span>0.0–0.5: Excellent form</span>
                </li>
                <li className="flex items-center space-x-2 text-xs text-white">
                  <span className="w-2.5 h-2.5 bg-lime-400 rounded-full block flex-shrink-0" />
                  <span>0.5–1.0: Very good form</span>
                </li>
                <li className="flex items-center space-x-2 text-xs text-white">
                  <span className="w-2.5 h-2.5 bg-yellow-400 rounded-full block flex-shrink-0" />
                  <span>1.0–1.5: Good form</span>
                </li>
                <li className="flex items-center space-x-2 text-xs text-white">
                  <span className="w-2.5 h-2.5 bg-orange-400 rounded-full block flex-shrink-0" />
                  <span>1.5–2.0: Fair form</span>
                </li>
                <li className="flex items-center space-x-2 text-xs text-white">
                  <span className="w-2.5 h-2.5 bg-rose-400 rounded-full block flex-shrink-0" />
                  <span>2.0–2.5: Poor form</span>
                </li>
                <li className="flex items-center space-x-2 text-xs text-white">
                  <span className="w-2.5 h-2.5 bg-red-500 rounded-full block flex-shrink-0" />
                  <span>2.5–3.0: Very poor form</span>
                </li>
                <li className="flex items-center space-x-2 text-xs text-white">
                  <span className="w-2.5 h-2.5 bg-purple-600 rounded-full block flex-shrink-0" />
                  <span>3.0–4.0: Extremely poor form</span>
                </li>
              </ul>
            </CardContent>
          </Card>

          {/* Your Score Card */}
          <Card className="shadow-md border border-white rounded-lg sticky top-[280px]">
            <CardContent className="pt-4 pb-4">
              <h3 className="text-lg font-semibold text-white mb-3">
                Your Score
              </h3>

              {finalScore != null ? (
                <div className="flex items-center space-x-2">
                  <span
                    className={`
                      w-4 h-4 rounded-full block flex-shrink-0 ${finalScore <= 0.5
                        ? "bg-green-500"
                        : finalScore <= 1.0
                          ? "bg-lime-400"
                          : finalScore <= 1.5
                            ? "bg-yellow-400"
                            : finalScore <= 2.0
                              ? "bg-orange-400"
                              : finalScore <= 2.5
                                ? "bg-rose-400"
                                : finalScore <= 3.0
                                  ? "bg-red-500"
                                  : "bg-purple-600"
                      }
                    `}
                  />
                  <div className="flex flex-col">
                    <span className="text-2xl font-bold text-white">
                      {finalScore.toFixed(1)}
                    </span>
                    <span className="text-sm italic text-gray-300">
                      ({scoreInterpretation})
                    </span>
                  </div>
                </div>
              ) : (
                <span className="text-sm text-gray-400">
                  Calculation in progress…
                </span>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Modal */}
        {showModal && (
          <>
            <div
              className="fixed inset-0 bg-black/20 backdrop-blur-sm z-10"
              onClick={() => setShowModal(false)}
            />
            <div className="fixed inset-0 flex items-center justify-center z-20 p-4">
              <div
                className="bg-gray-900 border border-white rounded-lg p-6 w-full max-w-md relative"
                onClick={(e) => e.stopPropagation()}
              >
                <button
                  onClick={() => setShowModal(false)}
                  className="absolute top-3 right-3 text-white text-xl font-bold hover:text-gray-300"
                >
                  ×
                </button>
                <h3 className="text-lg font-medium text-white mb-4">Suggestions</h3>
                <ul className="list-disc list-inside space-y-2 text-sm text-white">
                  {suggestions.map((tip, idx) => (
                    <li key={idx}>{tip}</li>
                  ))}
                </ul>
              </div>
            </div>
          </>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
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

          {loading && processingStatus && (
            <div className="mt-4 flex items-center justify-center">
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600 mr-2"></div>
              <span className="text-sm text-gray-600">{processingStatus}</span>
            </div>
          )}

          {error && (
            <>
              <Alert variant="destructive" className="mt-4">
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>Bad Video - Please reshoot and keep these quick tips in mind so our system can read your moves accurately.</AlertDescription>
              </Alert>


              <div className="mt-3 flex justify-center">
                <Button
                  onClick={() => setShowTipsBox(true)}
                  className="bg-purple-600 hover:bg-purple-700 text-white"
                >
                  Show Tips
                </Button>
              </div>


              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
                {/* --- WORST CONFIDENCE CARD (video “ugly”) --- */}
                <Card className="
      group 
      shadow-xl 
      border-2 border-red-500/30 
      bg-gradient-to-br from-red-900/20 to-red-800/10 
      backdrop-blur-sm 
      hover:border-red-400/50 
      transition-all duration-300 
      hover:shadow-2xl hover:shadow-red-500/20
    ">
                  <CardContent className="pt-4 pb-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-2">
                        <div className="relative">
                          <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
                          <div className="absolute inset-0 w-3 h-3 bg-red-500 rounded-full animate-ping opacity-30"></div>
                        </div>
                        <div>
                          <h4 className="text-base font-bold text-white group-hover:text-red-300 transition-colors">
                            Worst Confidence
                          </h4>
                          <p className="text-xs text-gray-400">
                            Least reliable detection
                          </p>
                        </div>
                      </div>
                      <div className="bg-red-500/20 p-1.5 rounded-lg">
                        <svg
                          className="w-4 h-4 text-red-400"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 \
                 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0 \
                 L3.732 16.5c-.77.833.192 2.5 1.732 2.5z"
                          />
                        </svg>
                      </div>
                    </div>

                    <div className="bg-black/30 rounded-lg p-3 border border-red-500/20">
                      <div className="text-center">
                        <div className="text-xl font-bold text-red-300 mb-1">
                          Worst Block: {worstConfidenceWindow.worst_block}
                        </div>
                        <div className="text-xl font-bold text-red-300 mb-1">
                          Time: {worstConfidenceWindow.time}
                        </div>
                        <div className="text-xs text-gray-400">
                          Critical detection area
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* --- MOST JITTERY CARD (video “ugly”) --- */}
                <Card className="
      group 
      shadow-xl 
      border-2 border-purple-500/30 
      bg-gradient-to-br from-purple-900/20 to-purple-800/10 
      backdrop-blur-sm 
      hover:border-purple-400/50 
      transition-all duration-300 
      hover:shadow-2xl hover:shadow-purple-500/20
    ">
                  <CardContent className="pt-4 pb-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center space-x-2">
                        <div className="relative">
                          <div className="w-3 h-3 bg-purple-500 rounded-full animate-bounce"></div>
                          <div className="absolute inset-0 w-3 h-3 bg-purple-500 rounded-full animate-ping opacity-30"></div>
                        </div>
                        <div>
                          <h4 className="text-base font-bold text-white group-hover:text-purple-300 transition-colors">
                            Most Jittery
                          </h4>
                          <p className="text-xs text-gray-400">
                            Highest movement instability
                          </p>
                        </div>
                      </div>
                      <div className="bg-purple-500/20 p-1.5 rounded-lg">
                        <svg
                          className="w-4 h-4 text-purple-400"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M13 10V3L4 14h7v7l9-11h-7z"
                          />
                        </svg>
                      </div>
                    </div>

                    <div className="bg-black/30 rounded-lg p-3 border border-purple-500/20">
                      <div className="text-center">
                        <div className="text-xl font-bold text-purple-300 mb-1">
                          Worst Block: {mostJitteryWindow.worst_block}
                        </div>
                        <div className="text-xl font-bold text-purple-300 mb-1">
                          Time: {mostJitteryWindow.time}
                        </div>
                        <div className="text-xs text-gray-400">
                          Peak instability detected
                        </div>
                      </div>
                      <div className="flex items-center justify-center space-x-1.5 mt-3">
                        <div
                          className="w-1.5 h-1.5 bg-purple-400 rounded-full animate-bounce"
                          style={{ animationDelay: "0ms" }}
                        ></div>
                        <div
                          className="w-1.5 h-1.5 bg-purple-400 rounded-full animate-bounce"
                          style={{ animationDelay: "150ms" }}
                        ></div>
                        <div
                          className="w-1.5 h-1.5 bg-purple-400 rounded-full animate-bounce"
                          style={{ animationDelay: "300ms" }}
                        ></div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

            </>
          )}


          {originalData.length > 0 && (
            <div className="mt-4 flex gap-4 justify-center">
              <Badge
                variant="outline"
                className="bg-blue-50 text-black border-blue-200 px-3 py-1"
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
                      ((originalData.length - trimmedData.length) / originalData.length) * 100
                    )
                    : 0}
                  % removed)
                </Badge>
              )}
            </div>
          )}

          {originalData.length > 0 && trimmedData.length > 0 && (
            <div className="mt-4 text-sm text-white-600 flex items-center justify-center">
              <Info className="h-4 w-4 mr-1 text-blue-500" />
              <span>
                Green segments represent frames that were kept, red segments were removed
              </span>
            </div>
          )}
        </CardContent>
      </Card>

      {originalData.length > 0 && trimmedData.length > 0 && (
        <SkeletonProvider>
          <CustomAnimationManager />
          <FrameVisualizer />
        </SkeletonProvider>
      )}


      {
        showTipsBox && (
          <>
            <div
              className="fixed inset-0 bg-black/30 z-10"
              onClick={() => setShowTipsBox(false)}
            />


            <div className="fixed inset-0 flex items-center justify-center z-20 p-4">
              <div
                className="bg-gray-900 border border-white rounded-lg p-6 w-full max-w-lg relative"
                onClick={(e) => e.stopPropagation()}
              >
                <button
                  onClick={() => setShowTipsBox(false)}
                  className="absolute top-3 right-3 text-white text-2xl font-bold hover:text-gray-300"
                >
                  ×
                </button>
                <h3 className="text-xl font-semibold text-white mb-4">Quick Tips</h3>
                <ul className="list-disc list-inside space-y-2 text-sm text-white">
                  {tips.map((tip, idx) => (
                    <li key={idx}>{tip}</li>
                  ))}
                </ul>
              </div>
            </div>
          </>
        )
      }
    </div>
  );


}

