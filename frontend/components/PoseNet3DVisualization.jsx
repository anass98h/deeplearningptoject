"use client";

import React, { useRef, useEffect, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  CardFooter,
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

  // DOM refs
  const containerRef = useRef(null);

  // Three.js objects
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const controlsRef = useRef(null);
  const animationFrameRef = useRef(null);
  const timerRef = useRef(null);

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

  // Custom debug logger
  const logDebug = (message, data = null) => {
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
  };

  // Check WebGL capabilities
  const checkWebGLCapabilities = () => {
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
        maxViewportDims: gl.getParameter(gl.MAX_VIEWPORT_DIMS),
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
  };

  // Cleanup function for Three.js objects
  const cleanup = () => {
    logDebug("Cleaning up Three.js scene");

    // Cancel animation frame
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
      logDebug("Cancelled animation frame");
    }

    // Clear animation timer
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
      logDebug("Cleared animation timer");
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
      logDebug("Disposed keypoint meshes");

      // Dispose of bones
      bonesRef.current.forEach(({ line }) => {
        if (line.geometry) line.geometry.dispose();
        if (line.material) line.material.dispose();
        if (sceneRef.current) sceneRef.current.remove(line);
      });
      bonesRef.current = [];
      logDebug("Disposed bone lines");

      // Clear scene
      while (sceneRef.current.children.length > 0) {
        const object = sceneRef.current.children[0];
        sceneRef.current.remove(object);
      }
      logDebug("Cleared scene children");
    }

    // Remove renderer from DOM
    if (
      containerRef.current &&
      rendererRef.current &&
      rendererRef.current.domElement
    ) {
      try {
        containerRef.current.removeChild(rendererRef.current.domElement);
        logDebug("Removed renderer from DOM");
      } catch (e) {
        logDebug("Error removing renderer from DOM:", e.message);
      }
    }

    // Dispose of renderer
    if (rendererRef.current) {
      rendererRef.current.dispose();
      rendererRef.current = null;
      logDebug("Disposed renderer");
    }

    // Clear controls
    if (controlsRef.current) {
      controlsRef.current.dispose();
      controlsRef.current = null;
      logDebug("Disposed controls");
    }

    setIsInitialized(false);
    setVisibleKeypoints(0);
    setVisibleBones(0);
  };

  // Initialize Three.js scene
  const initializeScene = () => {
    if (!containerRef.current) {
      logDebug("Cannot initialize scene: container ref is null");
      return;
    }

    logDebug("Initializing Three.js scene");
    setSceneError(null);

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
      logDebug(`Container dimensions: ${width}x${height}`);

      // Create scene
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xf0f0f0);
      sceneRef.current = scene;
      logDebug("Created THREE.Scene");

      // Create camera
      const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
      camera.position.set(0, 1, 5);
      cameraRef.current = camera;
      logDebug("Created PerspectiveCamera");

      // Create renderer
      const renderer = new THREE.WebGLRenderer({
        antialias: true,
        powerPreference: "high-performance",
      });
      renderer.setSize(width, height);
      renderer.setPixelRatio(window.devicePixelRatio);
      renderer.shadowMap.enabled = true;

      // Check if container already has a canvas
      const existingCanvas = containerRef.current.querySelector("canvas");
      if (existingCanvas) {
        logDebug(
          "WARNING: Container already has a canvas element, removing it"
        );
        containerRef.current.removeChild(existingCanvas);
      }

      containerRef.current.appendChild(renderer.domElement);
      rendererRef.current = renderer;
      logDebug("Created WebGLRenderer and added to DOM");

      // Set up controls
      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.05;
      controls.autoRotate = isRotating;
      controls.autoRotateSpeed = 2.0;
      controlsRef.current = controls;
      logDebug("Created OrbitControls");

      // Add a grid helper
      const gridHelper = new THREE.GridHelper(10, 20, 0x555555, 0xcccccc);
      scene.add(gridHelper);
      logDebug("Added GridHelper");

      // Add lights
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
      scene.add(ambientLight);

      const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
      directionalLight.position.set(5, 5, 5);
      directionalLight.castShadow = true;
      directionalLight.shadow.mapSize.width = 1024;
      directionalLight.shadow.mapSize.height = 1024;
      scene.add(directionalLight);
      logDebug("Added lights");

      // Create keypoint meshes
      Object.keys(KEYPOINT_COLORS).forEach((keypoint) => {
        const geometry = new THREE.SphereGeometry(0.05, 16, 16);
        const material = new THREE.MeshStandardMaterial({
          color: KEYPOINT_COLORS[keypoint],
          metalness: 0.3,
          roughness: 0.7,
        });

        const mesh = new THREE.Mesh(geometry, material);
        mesh.visible = false; // Initially hidden
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        scene.add(mesh);
        keypointMeshesRef.current[keypoint] = mesh;
      });
      logDebug("Created keypoint meshes", {
        count: Object.keys(KEYPOINT_COLORS).length,
      });

      // Create bone connections
      SKELETON_CONNECTIONS.forEach(([start, end]) => {
        const material = new THREE.LineBasicMaterial({
          color: 0x0088ff,
          linewidth: 2,
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
      logDebug("Created bone connections", {
        count: SKELETON_CONNECTIONS.length,
      });

      // Setup animation loop
      const animate = () => {
        animationFrameRef.current = requestAnimationFrame(animate);

        if (controlsRef.current) {
          controlsRef.current.update();
        }

        if (rendererRef.current && cameraRef.current && sceneRef.current) {
          rendererRef.current.render(sceneRef.current, cameraRef.current);
        }
      };

      animate();
      logDebug("Started animation loop");

      // Add event listeners to debug OrbitControls
      renderer.domElement.addEventListener("mousedown", () => {
        logDebug("Canvas mousedown event");
      });

      renderer.domElement.addEventListener("mousemove", () => {
        if (!controlsRef.current) return;
        // Only log occasionally to avoid flooding
        if (Math.random() < 0.01) {
          logDebug("Canvas mousemove, camera position:", {
            x: cameraRef.current.position.x.toFixed(2),
            y: cameraRef.current.position.y.toFixed(2),
            z: cameraRef.current.position.z.toFixed(2),
          });
        }
      });

      setIsInitialized(true);
      logDebug("Scene initialization complete");
    } catch (error) {
      const errorMsg = `Error initializing scene: ${error.message}`;
      logDebug(errorMsg);
      setSceneError(errorMsg);
    }
  };

  // Attach resize handler
  useEffect(() => {
    const handleResize = () => {
      if (!containerRef.current || !rendererRef.current || !cameraRef.current) {
        logDebug("Cannot handle resize: missing refs");
        return;
      }

      const width = containerRef.current.clientWidth;
      const height = 500;
      logDebug(`Window resize: ${width}x${height}`);

      cameraRef.current.aspect = width / height;
      cameraRef.current.updateProjectionMatrix();

      rendererRef.current.setSize(width, height);
    };

    window.addEventListener("resize", handleResize);
    logDebug("Added window resize listener");

    return () => {
      window.removeEventListener("resize", handleResize);
      logDebug("Removed window resize listener");
    };
  }, []);

  // Initialize scene when component mounts
  useEffect(() => {
    logDebug("Component mounted");
    initializeScene();

    // Cleanup when component unmounts
    return () => {
      logDebug("Component unmounting");
      cleanup();
    };
  }, []); // Empty dependency array - only run once on mount

  // Update rotation state when isRotating changes
  useEffect(() => {
    if (controlsRef.current) {
      controlsRef.current.autoRotate = isRotating;
      logDebug(`Set autoRotate to ${isRotating}`);
    }
  }, [isRotating]);

  // Effect for initial data display and when poseData changes
  useEffect(() => {
    logDebug("poseData or initialization changed", {
      hasPoseData: !!poseData,
      poseDataLength: poseData?.length || 0,
      isInitialized: isInitialized,
    });

    if (poseData && poseData.length > 0 && isInitialized) {
      logDebug(`Visualization received ${poseData.length} frames of pose data`);

      // Check if we can extract keypoints from the first frame
      const firstFrame = poseData[0];
      logDebug("First frame raw data:", firstFrame);

      const keypoints = extractKeypoints(firstFrame);

      // Debug output to check data format
      logDebug("First frame data keys:", Object.keys(firstFrame).slice(0, 10));
      logDebug("Extracted keypoints:", Object.keys(keypoints));

      if (Object.keys(keypoints).length > 0) {
        const firstKeypointName = Object.keys(keypoints)[0];
        logDebug(
          `Keypoint example (${firstKeypointName}):`,
          keypoints[firstKeypointName]
        );
      } else {
        logDebug("WARNING: No keypoints extracted from frame data");
      }

      // Keypoint name mapping check
      const expectedKeypoints = Object.keys(KEYPOINT_COLORS);
      const foundKeypoints = Object.keys(keypoints);
      const missingKeypoints = expectedKeypoints.filter(
        (k) => !foundKeypoints.includes(k)
      );

      logDebug("Expected keypoints:", expectedKeypoints);
      logDebug("Found keypoints:", foundKeypoints);

      if (missingKeypoints.length > 0) {
        logDebug("Missing keypoints:", missingKeypoints);
      }

      // Continue with rendering
      setCurrentFrame(0);
      updatePoseFrame(0);
    }
  }, [poseData, isInitialized]);

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
  }, [isPlaying, poseData, playbackSpeed]);

  // Extract keypoints from frame data
  const extractKeypoints = (frameData) => {
    if (!frameData) {
      logDebug("extractKeypoints: No frame data provided");
      return {};
    }

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
  };

  // Update pose visualization for a specific frame
  const updatePoseFrame = (frameIndex) => {
    if (!poseData || poseData.length === 0) {
      logDebug("updatePoseFrame: No pose data available");
      return;
    }

    if (!sceneRef.current) {
      logDebug("updatePoseFrame: No scene available");
      return;
    }

    if (!isInitialized) {
      logDebug("updatePoseFrame: Scene not initialized");
      return;
    }

    try {
      const frameData = poseData[frameIndex];
      const keypoints = extractKeypoints(frameData);

      // Debug log for first frame
      if (frameIndex === 0) {
        logDebug("First frame keypoints:", keypoints);

        // Add detailed diagnostic information
        const xKeys = Object.keys(frameData).filter((k) => k.endsWith("_x"));
        const yKeys = Object.keys(frameData).filter((k) => k.endsWith("_y"));
        const zKeys = Object.keys(frameData).filter((k) => k.endsWith("_z"));

        logDebug("X coordinate keys:", xKeys);
        logDebug("Y coordinate keys:", yKeys);
        logDebug("Z coordinate keys:", zKeys);
      }

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
          if (frameIndex === 0) {
            logDebug(`Missing coordinate for keypoint ${name}:`, position);
          }
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
          if (frameIndex === 0) {
            logDebug(
              `Cannot draw line from ${start} to ${end}: missing coordinates`
            );
          }
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

      if (frameIndex === 0) {
        logDebug(
          `Visible elements: ${visiblePoints} keypoints, ${visibleLines} bones`
        );
      }
    } catch (error) {
      logDebug(`Error updating pose frame ${frameIndex}:`, error.message);
    }
  };

  // Event handlers for UI controls
  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const handleReset = () => {
    setIsPlaying(false);
    setCurrentFrame(0);
    updatePoseFrame(0);
  };

  const handleRotationToggle = () => {
    setIsRotating(!isRotating);
  };

  const handleFrameChange = (value) => {
    const frameIndex = value[0] || 0;
    setCurrentFrame(frameIndex);
    updatePoseFrame(frameIndex);
  };

  // Handle force re-initialize
  const handleForceInit = () => {
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
  };

  // Speed control function
  const setSpeed = (speed) => {
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
  };

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
