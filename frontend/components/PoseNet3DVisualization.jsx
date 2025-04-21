"use client";

import React, { useRef, useEffect, useState, useCallback } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Play, Pause, RotateCcw, MoveHorizontal, Box, Bug } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";

// Define connections between keypoints
const SKELETON_CONNECTIONS = [
  ["head", "left_shoulder"],
  ["head", "right_shoulder"],
  ["left_shoulder", "right_shoulder"],
  ["left_shoulder", "left_elbow"],
  ["left_elbow", "left_hand"],
  ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_hand"],
  ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"],
  ["left_hip", "right_hip"],
  ["left_hip", "left_knee"],
  ["left_knee", "left_foot"],
  ["right_hip", "right_knee"],
  ["right_knee", "right_foot"],
];

// Keypoint colors
const KEYPOINT_COLORS = {
  head: 0xff5500,
  left_shoulder: 0x00aaff,
  right_shoulder: 0x00aaff,
  left_elbow: 0x00ccff,
  right_elbow: 0x00ccff,
  left_hand: 0x00ffff,
  right_hand: 0x00ffff,
  left_hip: 0xffaa00,
  right_hip: 0xffaa00,
  left_knee: 0xffcc00,
  right_knee: 0xffcc00,
  left_foot: 0xffff00,
  right_foot: 0xffff00,
};

export function PoseNet3DVisualization({ poseData, showControls = true }) {
  // Debugging state
  const [debugMessages, setDebugMessages] = useState([]);
  const [debugVisible, setDebugVisible] = useState(false);
  const [webglInfo, setWebglInfo] = useState(null);
  const lastLogTimeRef = useRef(0); // For throttling logs

  // DOM refs
  const containerRef = useRef(null);

  // Three.js objects
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const controlsRef = useRef(null);
  const animationFrameRef = useRef(null);
  const timerRef = useRef(null);
  const frameRateRef = useRef(0);
  const frameCountRef = useRef(0);
  const lastFrameTimeRef = useRef(0);

  // Scene objects
  const keypointMeshesRef = useRef({});
  const bonesRef = useRef([]);

  // UI State
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isRotating, setIsRotating] = useState(true);
  const [playbackSpeed, setPlaybackSpeed] = useState(100);
  const [isInitialized, setIsInitialized] = useState(false);
  const [visibleKeypoints, setVisibleKeypoints] = useState(0);
  const [visibleBones, setVisibleBones] = useState(0);
  const [sceneError, setSceneError] = useState(null);
  const [fps, setFps] = useState(0);

  // Custom debug logger with throttling
  const logDebug = useCallback((message, data = null) => {
    const now = Date.now();
    // Only log if it's been at least 100ms since the last log of this type
    // or if it's a critical message
    const isCritical =
      message.includes("error") ||
      message.includes("warning") ||
      message.includes("initialize") ||
      message.includes("cleanup");

    if (now - lastLogTimeRef.current < 100 && !isCritical) {
      return;
    }

    lastLogTimeRef.current = now;
    const timestamp = new Date().toISOString().split("T")[1].split(".")[0];
    const formattedMessage = `[${timestamp}] ${message}`;
    console.log(formattedMessage, data);

    setDebugMessages((prev) => {
      const newMessages = [
        {
          time: timestamp,
          text: message,
          data: data ? JSON.stringify(data).substring(0, 100) : null,
        },
        ...prev,
      ].slice(0, 20); // Keep only last 20 messages
      return newMessages;
    });
  }, []);

  // Check WebGL capabilities
  const checkWebGLCapabilities = useCallback(() => {
    try {
      const canvas = document.createElement("canvas");
      const gl =
        canvas.getContext("webgl") || canvas.getContext("experimental-webgl");

      if (!gl) {
        const errorMsg = "WebGL not supported by your browser";
        logDebug(errorMsg);
        setSceneError(errorMsg);
        return false;
      }

      const info = {
        version: "WebGL 1.0",
        vendor: gl.getParameter(gl.VENDOR),
        renderer: gl.getParameter(gl.RENDERER),
        maxTexSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
      };

      const debugInfo = gl.getExtension("WEBGL_debug_renderer_info");
      if (debugInfo) {
        info.vendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL);
        info.renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
      }

      logDebug("WebGL Capabilities", info);
      setWebglInfo(info);
      return true;
    } catch (e) {
      const errorMsg = `Error checking WebGL: ${e.message}`;
      logDebug(errorMsg);
      setSceneError(errorMsg);
      return false;
    }
  }, [logDebug]);

  // Cleanup function for Three.js objects
  const cleanup = useCallback(() => {
    logDebug("Cleaning up Three.js scene");

    // Cancel animation frame
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    // Clear animation timer
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    // Dispose of Three.js objects
    if (sceneRef.current) {
      // Dispose of keypoint meshes
      Object.values(keypointMeshesRef.current).forEach((mesh) => {
        if (mesh.geometry) mesh.geometry.dispose();
        if (mesh.material) mesh.material.dispose();
        if (sceneRef.current) sceneRef.current.remove(mesh);
      });
      keypointMeshesRef.current = {};

      // Dispose of bones
      bonesRef.current.forEach(({ line }) => {
        if (line.geometry) line.geometry.dispose();
        if (line.material) line.material.dispose();
        if (sceneRef.current) sceneRef.current.remove(line);
      });
      bonesRef.current = [];

      // Clear scene
      while (sceneRef.current.children.length > 0) {
        const object = sceneRef.current.children[0];
        sceneRef.current.remove(object);
      }
    }

    // Remove renderer from DOM
    if (
      containerRef.current &&
      rendererRef.current &&
      rendererRef.current.domElement
    ) {
      try {
        containerRef.current.removeChild(rendererRef.current.domElement);
      } catch (e) {
        logDebug("Error removing renderer from DOM:", e.message);
      }
    }

    // Dispose of renderer
    if (rendererRef.current) {
      rendererRef.current.dispose();
      rendererRef.current = null;
    }

    // Clear controls
    if (controlsRef.current) {
      controlsRef.current.dispose();
      controlsRef.current = null;
    }

    setIsInitialized(false);
    setVisibleKeypoints(0);
    setVisibleBones(0);
  }, [logDebug]);

  // Initialize Three.js scene
  const initializeScene = useCallback(() => {
    if (!containerRef.current) {
      logDebug("Cannot initialize scene: container ref is null");
      return;
    }

    // Check WebGL before proceeding
    if (!checkWebGLCapabilities()) {
      return;
    }

    // Clean up previous scene if needed
    cleanup();

    try {
      // Calculate dimensions
      const width = containerRef.current.clientWidth;
      const height = 500;

      // Create scene
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xf0f0f0);
      sceneRef.current = scene;

      // Create camera
      const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
      camera.position.set(0, 1, 5);
      cameraRef.current = camera;

      // Create renderer with optimized settings
      const renderer = new THREE.WebGLRenderer({
        antialias: true,
        powerPreference: "default", // Use 'default' instead of 'high-performance'
        precision: "mediump", // Use medium precision to save resources
      });
      renderer.setSize(width, height);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // Limit pixel ratio
      renderer.shadowMap.enabled = false; // Disable shadows for better performance

      // Check if container already has a canvas
      const existingCanvas = containerRef.current.querySelector("canvas");
      if (existingCanvas) {
        containerRef.current.removeChild(existingCanvas);
      }

      containerRef.current.appendChild(renderer.domElement);
      rendererRef.current = renderer;

      // Set up controls with optimized settings
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.1;
      controls.autoRotate = isRotating;
      controls.autoRotateSpeed = 1.0; // Reduce rotation speed
      controls.enableZoom = true;
      controls.enablePan = true;
      controlsRef.current = controls;

      // Add a grid helper
      const gridHelper = new THREE.GridHelper(10, 10, 0x555555, 0xcccccc); // Reduced grid division
      scene.add(gridHelper);

      // Add lights
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
      scene.add(ambientLight);

      // Single directional light is sufficient
      const directionalLight = new THREE.DirectionalLight(0xffffff, 0.7);
      directionalLight.position.set(5, 5, 5);
      directionalLight.castShadow = false; // Disable shadow casting
      scene.add(directionalLight);

      // Create keypoint meshes with simplified geometry
      Object.keys(KEYPOINT_COLORS).forEach((keypoint) => {
        // Use lower poly spheres
        const geometry = new THREE.SphereGeometry(0.05, 8, 8); // Reduced segments
        const material = new THREE.MeshBasicMaterial({
          // Use MeshBasicMaterial instead of MeshStandardMaterial
          color: KEYPOINT_COLORS[keypoint],
        });

        const mesh = new THREE.Mesh(geometry, material);
        mesh.visible = false; // Initially hidden
        scene.add(mesh);
        keypointMeshesRef.current[keypoint] = mesh;
      });

      // Create bone connections
      SKELETON_CONNECTIONS.forEach(([start, end]) => {
        const material = new THREE.LineBasicMaterial({
          color: 0x0088ff,
        });

        const geometry = new THREE.BufferGeometry();
        const line = new THREE.Line(geometry, material);
        line.visible = false; // Initially hidden
        scene.add(line);

        bonesRef.current.push({
          line,
          start,
          end,
        });
      });

      // Setup animation loop with frame rate limiting
      let lastTime = 0;
      const animate = (time) => {
        animationFrameRef.current = requestAnimationFrame(animate);

        // Calculate frame rate
        if (time - lastFrameTimeRef.current >= 1000) {
          // Update FPS every second
          setFps(frameCountRef.current);
          frameCountRef.current = 0;
          lastFrameTimeRef.current = time;
        } else {
          frameCountRef.current++;
        }

        // Frame rate limiting (60 FPS max)
        const elapsed = time - lastTime;
        if (elapsed < 16.7) {
          // ~60 FPS
          return;
        }
        lastTime = time;

        if (controlsRef.current) {
          controlsRef.current.update();
        }

        if (rendererRef.current && cameraRef.current && sceneRef.current) {
          rendererRef.current.render(sceneRef.current, cameraRef.current);
        }
      };

      animate(0);
      logDebug("Scene initialization complete");
      setIsInitialized(true);
    } catch (error) {
      const errorMsg = `Error initializing scene: ${error.message}`;
      logDebug(errorMsg);
      setSceneError(errorMsg);
    }
  }, [checkWebGLCapabilities, cleanup, isRotating, logDebug]);

  // Handle WebGL context loss
  useEffect(() => {
    if (!rendererRef.current || !rendererRef.current.domElement) return;

    const handleContextLost = (event) => {
      event.preventDefault();
      logDebug("WebGL context lost");

      // Force re-initialization
      setTimeout(() => {
        cleanup();
        setTimeout(() => {
          initializeScene();
          if (poseData && poseData.length > 0) {
            updatePoseFrame(currentFrame);
          }
        }, 100);
      }, 100);
    };

    rendererRef.current.domElement.addEventListener(
      "webglcontextlost",
      handleContextLost
    );

    return () => {
      if (rendererRef.current && rendererRef.current.domElement) {
        rendererRef.current.domElement.removeEventListener(
          "webglcontextlost",
          handleContextLost
        );
      }
    };
  }, [cleanup, currentFrame, initializeScene, logDebug, poseData]);

  // Attach resize handler
  useEffect(() => {
    const handleResize = () => {
      if (!containerRef.current || !rendererRef.current || !cameraRef.current) {
        return;
      }

      const width = containerRef.current.clientWidth;
      const height = 500;

      cameraRef.current.aspect = width / height;
      cameraRef.current.updateProjectionMatrix();

      rendererRef.current.setSize(width, height);
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  // Handle beforeunload event
  useEffect(() => {
    const handleBeforeUnload = () => {
      // Force cleanup before page unload
      cleanup();
    };

    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
    };
  }, [cleanup]);

  // Initialize scene when component mounts
  useEffect(() => {
    initializeScene();

    // Cleanup when component unmounts
    return cleanup;
  }, [cleanup, initializeScene]);

  // Update rotation state when isRotating changes
  useEffect(() => {
    if (controlsRef.current) {
      controlsRef.current.autoRotate = isRotating;
    }
  }, [isRotating]);

  // Effect for initial data display and when poseData changes
  useEffect(() => {
    if (poseData && poseData.length > 0 && isInitialized) {
      logDebug(`Received ${poseData.length} frames of pose data`);
      setCurrentFrame(0);
      updatePoseFrame(0);
    }
  }, [poseData, isInitialized, logDebug]);

  // Animation playback control
  useEffect(() => {
    // Clear any existing timer
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    // Start new timer if playing
    if (isPlaying && poseData && poseData.length > 0) {
      logDebug(`Starting animation playback at ${playbackSpeed}ms interval`);

      timerRef.current = setInterval(() => {
        setCurrentFrame((prev) => {
          const next = (prev + 1) % poseData.length;
          updatePoseFrame(next);
          return next;
        });
      }, playbackSpeed);
    }

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, [isPlaying, logDebug, playbackSpeed, poseData]);

  // Extract keypoints from frame data - memoized
  const extractKeypoints = useCallback(
    (frameData) => {
      if (!frameData) return {};

      const keypoints = {};

      try {
        Object.keys(frameData).forEach((key) => {
          // Skip non-coordinate keys
          if (key === "FrameNo" || !key.includes("_")) return;

          // Extract keypoint and coordinate
          const parts = key.split("_");
          let keypoint, coord;

          if (parts.length === 2) {
            [keypoint, coord] = parts;
          } else if (parts.length > 2) {
            coord = parts.pop();
            keypoint = parts.join("_");
          } else {
            return;
          }

          // Create keypoint object if needed
          if (!keypoints[keypoint]) {
            keypoints[keypoint] = {};
          }

          // Store coordinate value
          const value = frameData[key];
          keypoints[keypoint][coord] =
            typeof value === "string" ? parseFloat(value) : value;
        });
      } catch (error) {
        logDebug("Error extracting keypoints:", error.message);
      }

      return keypoints;
    },
    [logDebug]
  );

  // Update pose visualization for a specific frame
  const updatePoseFrame = useCallback(
    (frameIndex) => {
      if (
        !poseData ||
        poseData.length === 0 ||
        !sceneRef.current ||
        !isInitialized
      ) {
        return;
      }

      try {
        const frameData = poseData[frameIndex];
        const keypoints = extractKeypoints(frameData);

        // Scale factor for better visualization
        const scaleFactor = 1.0;

        let visiblePoints = 0;
        let visibleLines = 0;

        // Update keypoint positions
        Object.entries(keypointMeshesRef.current).forEach(([name, mesh]) => {
          const position = keypoints[name];

          if (!mesh || !position) {
            if (mesh) mesh.visible = false;
            return;
          }

          if (
            position.x !== undefined &&
            position.y !== undefined &&
            position.z !== undefined
          ) {
            mesh.position.x = position.x * scaleFactor;
            mesh.position.y = position.y * scaleFactor;
            mesh.position.z = position.z * scaleFactor;
            mesh.visible = true;
            visiblePoints++;
          } else {
            mesh.visible = false;
          }
        });

        // Update bones
        bonesRef.current.forEach(({ line, start, end }) => {
          const startPoint = keypoints[start];
          const endPoint = keypoints[end];

          if (
            !line ||
            !startPoint ||
            !endPoint ||
            startPoint.x === undefined ||
            startPoint.y === undefined ||
            startPoint.z === undefined ||
            endPoint.x === undefined ||
            endPoint.y === undefined ||
            endPoint.z === undefined
          ) {
            if (line) line.visible = false;
            return;
          }

          const points = [
            new THREE.Vector3(
              startPoint.x * scaleFactor,
              startPoint.y * scaleFactor,
              startPoint.z * scaleFactor
            ),
            new THREE.Vector3(
              endPoint.x * scaleFactor,
              endPoint.y * scaleFactor,
              endPoint.z * scaleFactor
            ),
          ];

          // Update line geometry
          if (line.geometry) line.geometry.dispose();
          line.geometry = new THREE.BufferGeometry().setFromPoints(points);
          line.visible = true;
          visibleLines++;
        });

        // Update state for debug display
        setVisibleKeypoints(visiblePoints);
        setVisibleBones(visibleLines);
      } catch (error) {
        logDebug(`Error updating pose frame ${frameIndex}:`, error.message);
      }
    },
    [extractKeypoints, isInitialized, logDebug, poseData]
  );

  // Event handlers for UI controls
  const handlePlayPause = useCallback(() => {
    setIsPlaying((prev) => !prev);
  }, []);

  const handleReset = useCallback(() => {
    setIsPlaying(false);
    setCurrentFrame(0);
    updatePoseFrame(0);
  }, [updatePoseFrame]);

  const handleRotationToggle = useCallback(() => {
    setIsRotating((prev) => !prev);
  }, []);

  const handleFrameChange = useCallback(
    (value) => {
      const frameIndex = value[0] || 0;
      setCurrentFrame(frameIndex);
      updatePoseFrame(frameIndex);
    },
    [updatePoseFrame]
  );

  // Handle force re-initialize
  const handleForceInit = useCallback(() => {
    logDebug("Force re-initializing scene");
    cleanup();
    setTimeout(() => {
      initializeScene();
      if (poseData && poseData.length > 0) {
        setTimeout(() => {
          updatePoseFrame(0);
        }, 100);
      }
    }, 100);
  }, [cleanup, initializeScene, logDebug, poseData, updatePoseFrame]);

  // Speed control function
  const setSpeed = useCallback(
    (speed) => {
      setPlaybackSpeed(speed);
      // If already playing, restart the animation to apply new speed
      if (isPlaying) {
        if (timerRef.current) {
          clearInterval(timerRef.current);
        }

        timerRef.current = setInterval(() => {
          setCurrentFrame((prev) => {
            const next = (prev + 1) % poseData.length;
            updatePoseFrame(next);
            return next;
          });
        }, speed);
      }
    },
    [isPlaying, poseData, updatePoseFrame]
  );

  return (
    <Card className="w-full shadow-lg">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-xl flex items-center">
              <Box className="h-5 w-5 mr-2 text-blue-500" />
              PoseNet 3D Visualization
            </CardTitle>
            <CardDescription>
              Interactive 3D skeletal model from PoseNet data
            </CardDescription>
          </div>

          <div className="flex items-center gap-2">
            <Badge variant="outline" className="bg-blue-50 text-blue-700">
              Frame {currentFrame + 1} of {poseData?.length || 0}
            </Badge>
            {fps > 0 && (
              <Badge variant="outline" className="bg-green-50 text-green-700">
                {fps} FPS
              </Badge>
            )}
            <Button
              variant="outline"
              size="sm"
              onClick={() => setDebugVisible(!debugVisible)}
              className="h-8 w-8 p-0"
            >
              <Bug className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {sceneError && (
          <Alert variant="destructive">
            <AlertDescription>{sceneError}</AlertDescription>
          </Alert>
        )}

        {/* 3D viewport container */}
        <div
          ref={containerRef}
          className="w-full h-[500px] rounded-md overflow-hidden bg-gradient-to-b from-gray-50 to-gray-100 border"
        />

        {/* Debug display */}
        {debugVisible && (
          <div className="p-3 bg-gray-50 rounded-md text-xs font-mono border overflow-auto max-h-[200px]">
            <div className="flex justify-between mb-2">
              <h4 className="font-bold">Debug Info</h4>
              <div className="space-x-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleForceInit}
                  className="h-7 text-xs"
                >
                  Force Reinitialize
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setDebugMessages([])}
                  className="h-7 text-xs"
                >
                  Clear Log
                </Button>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 mb-4 text-xs">
              <div>
                <p>
                  <strong>Scene:</strong>{" "}
                  {isInitialized ? "Initialized" : "Not Initialized"}
                </p>
                <p>
                  <strong>Visible Keypoints:</strong> {visibleKeypoints}/
                  {Object.keys(KEYPOINT_COLORS).length}
                </p>
                <p>
                  <strong>Visible Bones:</strong> {visibleBones}/
                  {SKELETON_CONNECTIONS.length}
                </p>
                <p>
                  <strong>FPS:</strong> {fps}
                </p>
              </div>

              <div>
                {webglInfo && (
                  <>
                    <p>
                      <strong>WebGL:</strong> {webglInfo.version}
                    </p>
                    <p>
                      <strong>Renderer:</strong> {webglInfo.renderer}
                    </p>
                    <p>
                      <strong>Vendor:</strong> {webglInfo.vendor}
                    </p>
                  </>
                )}
              </div>
            </div>

            <div className="border-t pt-2">
              <h5 className="font-bold mb-1">Log Messages:</h5>
              {debugMessages.map((msg, i) => (
                <div key={i} className="text-xs mb-1">
                  <span className="text-gray-500">[{msg.time}]</span> {msg.text}
                  {msg.data && (
                    <span className="text-blue-500"> {msg.data}</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Playback controls */}
        {showControls && poseData && poseData.length > 0 && (
          <div className="space-y-4">
            {/* Frame slider */}
            <div className="px-2">
              <Slider
                value={[currentFrame]}
                min={0}
                max={poseData.length - 1}
                step={1}
                onValueChange={handleFrameChange}
                disabled={isPlaying}
              />
            </div>

            {/* Control buttons */}
            <div className="flex justify-between">
              <div className="flex items-center space-x-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handlePlayPause}
                  className={
                    isPlaying ? "bg-red-50 text-red-600 border-red-200" : ""
                  }
                >
                  {isPlaying ? (
                    <>
                      <Pause className="h-4 w-4 mr-1" /> Pause
                    </>
                  ) : (
                    <>
                      <Play className="h-4 w-4 mr-1" /> Play
                    </>
                  )}
                </Button>

                <Button variant="outline" size="sm" onClick={handleReset}>
                  <RotateCcw className="h-4 w-4 mr-1" /> Reset
                </Button>

                {/* Speed control */}
                <div className="inline-flex rounded-md shadow-sm ml-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setSpeed(200)}
                    className={
                      playbackSpeed === 200 ? "bg-blue-50 text-blue-600" : ""
                    }
                  >
                    0.5x
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setSpeed(100)}
                    className={
                      playbackSpeed === 100 ? "bg-blue-50 text-blue-600" : ""
                    }
                  >
                    1x
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setSpeed(50)}
                    className={
                      playbackSpeed === 50 ? "bg-blue-50 text-blue-600" : ""
                    }
                  >
                    2x
                  </Button>
                </div>
              </div>

              <Button
                variant="outline"
                size="sm"
                onClick={handleRotationToggle}
                className={
                  isRotating ? "bg-blue-50 text-blue-600 border-blue-200" : ""
                }
              >
                <MoveHorizontal className="h-4 w-4 mr-1" />
                {isRotating ? "Stop Rotation" : "Auto Rotate"}
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
