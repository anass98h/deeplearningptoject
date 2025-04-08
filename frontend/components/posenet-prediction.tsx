"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  Camera,
  Play,
  Square,
  AlertCircle,
  Download,
  Save,
  Upload,
  CheckCircle2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import * as tf from "@tensorflow/tfjs";
import * as posenet from "@tensorflow-models/posenet";

// Define the connections between keypoints for drawing skeleton
const skeleton = [
  // Torso
  ["leftShoulder", "rightShoulder"],
  ["leftShoulder", "leftHip"],
  ["rightShoulder", "rightHip"],
  ["leftHip", "rightHip"],

  // Arms
  ["leftShoulder", "leftElbow"],
  ["leftElbow", "leftWrist"],
  ["rightShoulder", "rightElbow"],
  ["rightElbow", "rightWrist"],

  // Legs
  ["leftHip", "leftKnee"],
  ["leftKnee", "leftAnkle"],
  ["rightHip", "rightKnee"],
  ["rightKnee", "rightAnkle"],

  // Face
  ["nose", "leftEye"],
  ["nose", "rightEye"],
  ["leftEye", "leftEar"],
  ["rightEye", "rightEar"],
];

// Color scheme for different body parts
const colorMap = {
  face: "#FF0000", // Red for face
  torso: "#00FF00", // Green for torso
  arms: "#0000FF", // Blue for arms
  legs: "#FF00FF", // Magenta for legs
  keypoints: "#FFFF00", // Yellow for keypoints
};

export function PoseNetCapture() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [net, setNet] = useState<posenet.PoseNet | null>(null);
  const [isModelLoaded, setIsModelLoaded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [currentPose, setCurrentPose] = useState<posenet.Pose | null>(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const requestRef = useRef<number | null>(null);
  const isCapturingRef = useRef(false); // Use a ref to track capture state in the animation loop
  const [isUploading, setIsUploading] = useState(false);
  // Add state for camera devices
  const [cameraDevices, setCameraDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedCamera, setSelectedCamera] = useState<string>("");
  const [loadingDevices, setLoadingDevices] = useState(false);

  // Store captured poses for export
  const [capturedPoses, setCapturedPoses] = useState<
    Array<{
      timestamp: number;
      keypoints: Array<{
        part: string;
        position: { x: number; y: number };
        score: number;
      }>;
    }>
  >([]);

  // Load the PoseNet model on component mount
  useEffect(() => {
    const loadModel = async () => {
      setLoading(true);
      setProgress(0);

      const interval = setInterval(() => {
        setProgress((prev) => Math.min(prev + 5, 90));
      }, 200);

      try {
        await tf.setBackend("webgl");
        await tf.ready();
        const backend = tf.getBackend();
        console.log("TensorFlow backend:", backend);

        const loadedNet = await posenet.load({
          architecture: "MobileNetV1",
          outputStride: 16,
          inputResolution: { width: 640, height: 480 },
          multiplier: 0.75,
        });

        setNet(loadedNet);
        setIsModelLoaded(true);
        setProgress(100);
      } catch (error) {
        console.error("Failed to load PoseNet model:", error);
        setError("Failed to load PoseNet model. Please try again.");
      } finally {
        setLoading(false);
        clearInterval(interval);
        setProgress(100);
      }
    };

    loadModel();

    // Load available camera devices
    loadCameraDevices();

    return () => {
      // Cleanup function to ensure animation frame is canceled when component unmounts
      if (requestRef.current !== null) {
        cancelAnimationFrame(requestRef.current);
        requestRef.current = null;
      }
      isCapturingRef.current = false;
    };
  }, []);
  const uploadToBackend = async () => {
    if (capturedPoses.length === 0) {
      setError("No pose data captured yet.");
      return;
    }

    setIsUploading(true);
    setError(null); // Clear any previous errors

    try {
      const backendUrl =
        process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000/";

      // Convert the captured poses to a JSON string
      const jsonData = JSON.stringify(capturedPoses, null, 2);

      // Create a file from the JSON data
      const file = new File(
        [jsonData],
        `posenet-data-${new Date().toISOString()}.json`,
        {
          type: "application/json",
        }
      );

      // Create a FormData object and append the file
      const formData = new FormData();
      formData.append("file", file);

      console.log("Uploading file to backend...");

      // Send as multipart/form-data
      const response = await fetch(`${backendUrl}upload-posenet-data`, {
        method: "POST",
        body: formData,
      });

      // Get response text for better error handling
      const responseText = await response.text();
      let result;

      try {
        // Try to parse the response as JSON
        result = JSON.parse(responseText);
      } catch {
        // If parsing fails, use the raw text
        result = { detail: responseText };
      }

      if (!response.ok) {
        console.error("Error response:", result);
        // Extract error message from the response
        const errorMessage =
          result.detail || `Upload failed with status: ${response.status}`;
        throw new Error(errorMessage);
      }

      // Handle success message
      console.log("Upload successful:", result);
      // Set success message directly from server
      setError(result.message || "File uploaded successfully");
    } catch (err) {
      console.error("Error uploading data:", err);
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setIsUploading(false);
    }
  };

  // Function to load available camera devices
  const loadCameraDevices = async () => {
    setLoadingDevices(true);
    try {
      // First request permission to access media devices
      await navigator.mediaDevices.getUserMedia({ video: true });

      // Then enumerate all media devices
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(
        (device) => device.kind === "videoinput"
      );

      setCameraDevices(videoDevices);

      // Set default camera
      if (videoDevices.length > 0) {
        setSelectedCamera(videoDevices[0].deviceId);
      }
    } catch (err) {
      console.error("Error accessing camera devices:", err);
      setError("Failed to access camera devices. Please check permissions.");
    } finally {
      setLoadingDevices(false);
    }
  };

  // Handle camera selection change
  const handleCameraChange = (deviceId: string) => {
    setSelectedCamera(deviceId);

    // If webcam is already active, restart with new device
    if (isWebcamActive) {
      // Stop current stream
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        const tracks = stream.getTracks();
        tracks.forEach((track) => track.stop());
        videoRef.current.srcObject = null;
      }

      // Stop capturing if active
      if (isCapturing) {
        stopCapture();
      }

      setIsWebcamActive(false);

      // Start webcam with new device
      startWebcamWithDevice(deviceId);
    }
  };

  // Function to start webcam with specific device
  const startWebcamWithDevice = async (deviceId: string) => {
    try {
      const constraints = {
        video: {
          deviceId: { exact: deviceId },
          width: 640,
          height: 480,
        },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = async () => {
          const video = videoRef.current!;
          await video.play();
          setIsWebcamActive(true);

          // Set canvas size
          canvasRef.current!.width = video.videoWidth;
          canvasRef.current!.height = video.videoHeight;

          // Load the model now with correct dimensions
          try {
            setLoading(true);
            const model = await posenet.load({
              architecture: "MobileNetV1",
              outputStride: 16,
              inputResolution: {
                width: video.videoWidth,
                height: video.videoHeight,
              },
              multiplier: 0.75,
            });
            setNet(model);
            setIsModelLoaded(true);
            console.log(
              "Model loaded with resolution",
              video.videoWidth,
              video.videoHeight
            );
          } catch (err) {
            console.error("Failed to load PoseNet", err);
            setError("Could not load PoseNet.");
          } finally {
            setLoading(false);
          }
        };
      }
    } catch (err) {
      console.error("Error accessing webcam:", err);
      setError("Failed to access webcam. Please check permissions.");
    }
  };

  // Toggle webcam on/off
  const toggleWebcam = useCallback(async () => {
    if (isWebcamActive) {
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        const tracks = stream.getTracks();
        tracks.forEach((track) => track.stop());
        videoRef.current.srcObject = null;
      }
      setIsWebcamActive(false);
      // Make sure to stop capturing if webcam is turned off
      if (isCapturing) {
        stopCapture();
      }
    } else {
      // Start webcam with selected device
      if (selectedCamera) {
        startWebcamWithDevice(selectedCamera);
      } else {
        setError("No camera device selected.");
      }
    }
  }, [isWebcamActive, isCapturing, selectedCamera]);

  // The continuous capture loop function
  const captureFrame = useCallback(() => {
    // Use the ref value for checking if capture is active
    if (
      !isCapturingRef.current ||
      !isWebcamActive ||
      !net ||
      !videoRef.current
    ) {
      return;
    }

    // Estimate pose in current video frame
    net
      .estimateSinglePose(videoRef.current, {
        flipHorizontal: false,
      })
      .then((pose) => {
        // Only proceed if still capturing
        if (!isCapturingRef.current) return;

        // Update state with current pose
        setCurrentPose(pose);

        // Draw the pose on canvas
        drawEnhancedPose(pose);

        // Store the pose data
        const poseData = {
          timestamp: Date.now(),
          score: pose.score,
          keypoints: pose.keypoints.map((kp) => ({
            part: kp.part,
            position: {
              x: kp.position.x,
              y: kp.position.y,
            },
            score: kp.score,
          })),
        };

        // Update capturedPoses array
        setCapturedPoses((prev) => [...prev, poseData]);

        // Continue the loop only if still capturing
        if (isCapturingRef.current) {
          requestRef.current = requestAnimationFrame(captureFrame);
        }
      })
      .catch((err) => {
        console.error("Error estimating pose:", err);
        // Try to continue despite error only if still capturing
        if (isCapturingRef.current) {
          requestRef.current = requestAnimationFrame(captureFrame);
        }
      });
  }, [isWebcamActive, net]);

  // Explicitly define start and stop functions
  const startCapture = useCallback(() => {
    // Reset captured data
    setCapturedPoses([]);

    // Set both the state and the ref
    setIsCapturing(true);
    isCapturingRef.current = true;

    // Start the capture loop
    requestRef.current = requestAnimationFrame(captureFrame);
    console.log("Started capturing");
  }, [captureFrame]);

  const stopCapture = useCallback(() => {
    // Set both the state and the ref
    setIsCapturing(false);
    isCapturingRef.current = false;

    // Cancel the animation frame
    if (requestRef.current !== null) {
      cancelAnimationFrame(requestRef.current);
      requestRef.current = null;
    }
    console.log("Stopped capturing");
  }, []);

  // Toggle capture function that calls either start or stop
  const toggleCapture = useCallback(() => {
    if (isCapturing) {
      stopCapture();
    } else {
      startCapture();
    }
  }, [isCapturing, startCapture, stopCapture]);

  // Effect to ensure requestAnimationFrame is canceled when isCapturing changes
  useEffect(() => {
    if (!isCapturing && requestRef.current !== null) {
      cancelAnimationFrame(requestRef.current);
      requestRef.current = null;
    }
  }, [isCapturing]);

  // Enhanced pose drawing function with skeleton and colors
  const drawEnhancedPose = (pose: posenet.Pose) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;

    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Create keypoint lookup for easy access
    const keypointMap = pose.keypoints.reduce((map, keypoint) => {
      map[keypoint.part] = keypoint;
      return map;
    }, {} as Record<string, posenet.Keypoint>);

    // Draw the skeleton lines
    ctx.lineWidth = 3;

    // Draw skeleton connections with different colors for different body parts
    skeleton.forEach(([startPoint, endPoint]) => {
      const start = keypointMap[startPoint];
      const end = keypointMap[endPoint];

      // Only draw if both points are detected with reasonable confidence
      if (start && end && start.score > 0.5 && end.score > 0.5) {
        // Determine segment color based on body part
        let color = colorMap.keypoints;

        if (
          startPoint.includes("Eye") ||
          startPoint.includes("Ear") ||
          startPoint.includes("Nose") ||
          endPoint.includes("Eye") ||
          endPoint.includes("Ear") ||
          endPoint.includes("Nose")
        ) {
          color = colorMap.face;
        } else if (
          startPoint.includes("Shoulder") ||
          startPoint.includes("Hip") ||
          endPoint.includes("Shoulder") ||
          endPoint.includes("Hip")
        ) {
          color = colorMap.torso;
        } else if (
          startPoint.includes("Elbow") ||
          startPoint.includes("Wrist") ||
          endPoint.includes("Elbow") ||
          endPoint.includes("Wrist")
        ) {
          color = colorMap.arms;
        } else if (
          startPoint.includes("Knee") ||
          startPoint.includes("Ankle") ||
          endPoint.includes("Knee") ||
          endPoint.includes("Ankle")
        ) {
          color = colorMap.legs;
        }

        ctx.strokeStyle = color;
        ctx.beginPath();
        ctx.moveTo(start.position.x, start.position.y);
        ctx.lineTo(end.position.x, end.position.y);
        ctx.stroke();
      }
    });

    // Draw keypoints
    ctx.fillStyle = colorMap.keypoints;
    pose.keypoints.forEach((keypoint) => {
      if (keypoint.score > 0.5) {
        ctx.beginPath();
        ctx.arc(keypoint.position.x, keypoint.position.y, 5, 0, 2 * Math.PI);
        ctx.fill();
      }
    });
  };

  // Export data as JSON
  const exportAsJSON = () => {
    if (capturedPoses.length === 0) {
      setError("No pose data captured yet.");
      return;
    }

    const dataStr = JSON.stringify(capturedPoses, null, 2);
    const dataBlob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(dataBlob);

    const link = document.createElement("a");
    link.href = url;
    link.download = `posenet-data-${new Date().toISOString()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Export data as CSV
  const exportAsCSV = () => {
    if (capturedPoses.length === 0) {
      setError("No pose data captured yet.");
      return;
    }

    // Create CSV header
    const headers = [
      "timestamp",
      ...posenet.partNames.flatMap((part) => [
        `${part}_x`,
        `${part}_y`,
        `${part}_score`,
      ]),
    ];

    // Create CSV rows
    const rows = capturedPoses.map((pose) => {
      const keypointMap = pose.keypoints.reduce((map, kp) => {
        map[kp.part] = kp;
        return map;
      }, {} as Record<string, (typeof pose.keypoints)[0]>);

      return [
        pose.timestamp,
        ...posenet.partNames.flatMap((part) => {
          const kp = keypointMap[part];
          return kp ? [kp.position.x, kp.position.y, kp.score] : [0, 0, 0]; // Default values if keypoint not found
        }),
      ];
    });

    // Combine headers and rows
    const csvContent = [
      headers.join(","),
      ...rows.map((row) => row.join(",")),
    ].join("\n");

    const dataBlob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(dataBlob);

    const link = document.createElement("a");
    link.href = url;
    link.download = `posenet-data-${new Date().toISOString()}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-4 p-2 sm:p-6">
      {loading && (
        <div className="space-y-2 p-3 rounded-lg bg-gradient-to-r from-gray-50 to-gray-100 shadow-md">
          <div className="flex justify-between text-sm">
            <span className="font-medium text-gray-700">
              Loading PoseNet model...
            </span>
            <span className="font-semibold text-primary">{progress}%</span>
          </div>
          <Progress value={progress} className="h-2 bg-gray-200" />
        </div>
      )}

      {error && (
        <Alert
          variant={
            error.includes("uploaded successfully") ? "default" : "destructive"
          }
          className={`border-0 shadow-lg ${
            error.includes("uploaded successfully")
              ? "bg-gradient-to-r from-green-50 to-emerald-50"
              : "bg-gradient-to-r from-rose-50 to-red-50"
          }`}
        >
          {error.includes("uploaded successfully") ? (
            <CheckCircle2 className="h-4 w-4 text-emerald-500" />
          ) : (
            <AlertCircle className="h-4 w-4 text-red-500" />
          )}
          <AlertTitle
            className={
              error.includes("uploaded successfully")
                ? "text-emerald-800"
                : "text-red-700"
            }
          >
            {error.includes("uploaded successfully") ? "Success" : "Error"}
          </AlertTitle>
          <AlertDescription
            className={
              error.includes("uploaded successfully")
                ? "text-emerald-700"
                : "text-red-600"
            }
          >
            {error}
          </AlertDescription>
        </Alert>
      )}

      {isModelLoaded && (
        <div className="space-y-4">
          <Card className="bg-white rounded-xl shadow-lg overflow-hidden">
            <CardHeader className="p-3 sm:p-4">
              <CardTitle className="text-lg">PoseNet Data Capture</CardTitle>
            </CardHeader>
            <CardContent className="p-2 sm:p-4">
              {/* Camera selection dropdown */}
              <div className="mb-3">
                <label
                  htmlFor="camera-select"
                  className="block text-sm font-medium text-gray-700 mb-1"
                >
                  Select Camera
                </label>
                <Select
                  disabled={isWebcamActive || loadingDevices}
                  value={selectedCamera}
                  onValueChange={handleCameraChange}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue
                      placeholder={
                        loadingDevices
                          ? "Loading cameras..."
                          : "Select a camera"
                      }
                    />
                  </SelectTrigger>
                  <SelectContent>
                    {cameraDevices.length === 0 && (
                      <SelectItem value="no-cameras" disabled>
                        No cameras found
                      </SelectItem>
                    )}
                    {cameraDevices.map((device) => (
                      <SelectItem key={device.deviceId} value={device.deviceId}>
                        {device.label ||
                          `Camera ${cameraDevices.indexOf(device) + 1}`}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {cameraDevices.length === 0 && !loadingDevices && (
                  <p className="text-xs text-amber-600 mt-1">
                    No cameras detected. Please ensure your camera is connected
                    and you&apos;ve granted permissions.
                  </p>
                )}
                <div className="flex justify-end mt-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={loadCameraDevices}
                    disabled={loadingDevices}
                    className="text-xs"
                  >
                    {loadingDevices ? "Refreshing..." : "Refresh Cameras"}
                  </Button>
                </div>
              </div>

              <div
                className="relative bg-gradient-to-br from-gray-900 to-gray-800 rounded-lg overflow-hidden shadow-inner mx-auto"
                style={{
                  aspectRatio: "4/3",
                  minHeight: "280px",
                  maxHeight: "calc(min(80vh, 540px))",
                  maxWidth: "800px", // Add max width constraint
                  width: "100%",
                }}
              >
                <video
                  ref={videoRef}
                  className="absolute inset-0 w-full h-full object-cover"
                  width={640}
                  height={480}
                  playsInline
                />
                <canvas
                  ref={canvasRef}
                  className="absolute inset-0 w-full h-full"
                  width={640}
                  height={480}
                />

                {!isWebcamActive && (
                  <div className="absolute inset-0 flex items-center justify-center">
                    <Button
                      onClick={toggleWebcam}
                      size="sm"
                      disabled={!selectedCamera || cameraDevices.length === 0}
                      className="bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-700 hover:to-blue-700 shadow-lg"
                    >
                      <Camera className="mr-2 h-4 w-4" />
                      Start Webcam
                    </Button>
                  </div>
                )}
              </div>

              {isWebcamActive && (
                <div className="flex flex-col sm:flex-row justify-between gap-2 mt-3">
                  <Button
                    onClick={toggleWebcam}
                    variant="outline"
                    size="sm"
                    className="border-red-200 text-red-600 hover:bg-red-50"
                  >
                    Stop Webcam
                  </Button>

                  <Button
                    onClick={toggleCapture}
                    variant={isCapturing ? "destructive" : "default"}
                    size="sm"
                    className={
                      isCapturing
                        ? "bg-gradient-to-r from-rose-600 to-red-600"
                        : "bg-gradient-to-r from-cyan-600 to-blue-600"
                    }
                  >
                    {isCapturing ? (
                      <>
                        <Square className="mr-2 h-4 w-4" />
                        Stop Capturing
                      </>
                    ) : (
                      <>
                        <Play className="mr-2 h-4 w-4" />
                        Start Capturing
                      </>
                    )}
                  </Button>
                </div>
              )}

              {capturedPoses.length > 0 && !isCapturing && (
                <div className="flex flex-col sm:flex-row justify-between gap-2 mt-3">
                  <div className="text-sm text-gray-600">
                    {capturedPoses.length} frames captured
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <Button
                      onClick={uploadToBackend}
                      variant="outline"
                      size="sm"
                      disabled={isUploading}
                      className="bg-gradient-to-r from-blue-50 to-blue-100 border-blue-200 text-blue-700 hover:bg-blue-100"
                    >
                      {isUploading ? (
                        <>Uploading...</>
                      ) : (
                        <>
                          <Upload className="mr-2 h-4 w-4" />
                          Upload to Server
                        </>
                      )}
                    </Button>
                    <Button
                      onClick={exportAsJSON}
                      variant="outline"
                      size="sm"
                      className="bg-gradient-to-r from-amber-50 to-amber-100 border-amber-200 text-amber-700 hover:bg-amber-100"
                    >
                      <Save className="mr-2 h-4 w-4" />
                      Export JSON
                    </Button>
                    <Button
                      onClick={exportAsCSV}
                      variant="outline"
                      size="sm"
                      className="bg-gradient-to-r from-emerald-50 to-emerald-100 border-emerald-200 text-emerald-700 hover:bg-emerald-100"
                    >
                      <Download className="mr-2 h-4 w-4" />
                      Export CSV
                    </Button>
                  </div>
                </div>
              )}

              {/* Add the error Alert here, below the upload buttons */}
              {error && (
                <div className="mt-3">
                  <Alert
                    variant={
                      error.includes("uploaded successfully")
                        ? "default"
                        : "destructive"
                    }
                    className={`border-0 shadow-lg ${
                      error.includes("uploaded successfully")
                        ? "bg-gradient-to-r from-green-50 to-emerald-50"
                        : "bg-gradient-to-r from-rose-50 to-red-50"
                    }`}
                  >
                    {error.includes("uploaded successfully") ? (
                      <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                    ) : (
                      <AlertCircle className="h-4 w-4 text-red-500" />
                    )}
                    <AlertTitle
                      className={
                        error.includes("uploaded successfully")
                          ? "text-emerald-800"
                          : "text-red-700"
                      }
                    >
                      {error.includes("uploaded successfully")
                        ? "Success"
                        : "Error"}
                    </AlertTitle>
                    <AlertDescription
                      className={
                        error.includes("uploaded successfully")
                          ? "text-emerald-700"
                          : "text-red-600"
                      }
                    >
                      {error}
                    </AlertDescription>
                  </Alert>
                </div>
              )}

              {currentPose && (
                <div className="mt-3 p-2 bg-gray-50 rounded-lg">
                  <h3 className="font-medium text-gray-700 mb-1 text-sm">
                    Latest Pose Data:
                  </h3>
                  <div className="max-h-[100px] sm:max-h-[150px] overflow-y-auto bg-gray-100 p-2 rounded text-xs font-mono">
                    {JSON.stringify(
                      currentPose.keypoints
                        .filter((kp) => kp.score > 0.5)
                        .map((kp) => ({
                          part: kp.part,
                          x: Math.round(kp.position.x),
                          y: Math.round(kp.position.y),
                          score: kp.score.toFixed(2),
                        })),
                      null,
                      2
                    )}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}

export default PoseNetCapture;
