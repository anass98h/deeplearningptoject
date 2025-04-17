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
} from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { Play, Pause, RotateCcw, MoveHorizontal, Box } from "lucide-react";
import { Badge } from "@/components/ui/badge";

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

  // Initialize Three.js scene
  useEffect(() => {
    if (!containerRef.current) return;

    console.log("Initializing Three.js scene");

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

    // Create renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    containerRef.current.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Set up controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.autoRotate = isRotating;
    controls.autoRotateSpeed = 2.0;
    controlsRef.current = controls;

    // Add a grid helper
    const gridHelper = new THREE.GridHelper(10, 20, 0x555555, 0xcccccc);
    scene.add(gridHelper);

    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(5, 5, 5);
    directionalLight.castShadow = true;
    directionalLight.shadow.mapSize.width = 1024;
    directionalLight.shadow.mapSize.height = 1024;
    scene.add(directionalLight);

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

    // Setup animation loop
    const animate = () => {
      animationFrameRef.current = requestAnimationFrame(animate);

      if (controlsRef.current) {
        controlsRef.current.update();
      }

      renderer.render(scene, camera);
    };

    animate();

    // Handle window resize
    const handleResize = () => {
      if (!containerRef.current || !rendererRef.current || !cameraRef.current)
        return;

      const width = containerRef.current.clientWidth;
      const height = 500;

      cameraRef.current.aspect = width / height;
      cameraRef.current.updateProjectionMatrix();

      rendererRef.current.setSize(width, height);
    };

    window.addEventListener("resize", handleResize);

    // Cleanup function
    return () => {
      console.log("Cleaning up Three.js scene");

      // Remove event listeners
      window.removeEventListener("resize", handleResize);

      // Cancel animation frame
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }

      // Clear animation timer
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }

      // Dispose of Three.js objects
      Object.values(keypointMeshesRef.current).forEach((mesh) => {
        if (mesh.geometry) mesh.geometry.dispose();
        if (mesh.material) mesh.material.dispose();
      });

      bonesRef.current.forEach(({ line }) => {
        if (line.geometry) line.geometry.dispose();
        if (line.material) line.material.dispose();
      });

      // Remove renderer from DOM
      if (
        containerRef.current &&
        rendererRef.current &&
        rendererRef.current.domElement
      ) {
        try {
          containerRef.current.removeChild(rendererRef.current.domElement);
        } catch (e) {
          console.warn("Error removing renderer from DOM:", e);
        }
      }

      // Dispose of renderer
      if (rendererRef.current) {
        rendererRef.current.dispose();
      }
    };
  }, []);

  // Effect for initial data display
  useEffect(() => {
    if (poseData && poseData.length > 0) {
      console.log(`Got ${poseData.length} frames of pose data`);
      updatePoseFrame(0);
    }
  }, [poseData]);

  // Update rotation state
  useEffect(() => {
    if (controlsRef.current) {
      controlsRef.current.autoRotate = isRotating;
    }
  }, [isRotating]);

  // Animation playback control
  useEffect(() => {
    // Clear any existing timer
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    // Start new timer if playing
    if (isPlaying && poseData && poseData.length > 0) {
      console.log(`Starting animation playback at ${playbackSpeed}ms interval`);

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
      console.error("Error extracting keypoints:", error);
    }

    return keypoints;
  };

  // Update pose visualization for a specific frame
  const updatePoseFrame = (frameIndex) => {
    if (!poseData || poseData.length === 0 || !sceneRef.current) return;

    try {
      const frameData = poseData[frameIndex];
      const keypoints = extractKeypoints(frameData);

      // Scale factor for better visualization
      const scaleFactor = 1.0;

      // Update keypoint positions
      Object.entries(keypoints).forEach(([name, position]) => {
        const mesh = keypointMeshesRef.current[name];
        if (!mesh) return;

        if (
          position.x !== undefined &&
          position.y !== undefined &&
          position.z !== undefined
        ) {
          mesh.position.x = position.x * scaleFactor;
          mesh.position.y = position.y * scaleFactor;
          mesh.position.z = position.z * scaleFactor;
          mesh.visible = true;
        } else {
          mesh.visible = false;
        }
      });

      // Update bones
      bonesRef.current.forEach(({ line, start, end }) => {
        const startPoint = keypoints[start];
        const endPoint = keypoints[end];

        if (
          startPoint &&
          endPoint &&
          startPoint.x !== undefined &&
          startPoint.y !== undefined &&
          startPoint.z !== undefined &&
          endPoint.x !== undefined &&
          endPoint.y !== undefined &&
          endPoint.z !== undefined
        ) {
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
        } else {
          line.visible = false;
        }
      });
    } catch (error) {
      console.error("Error updating pose frame:", error);
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

          <Badge variant="outline" className="bg-blue-50 text-blue-700">
            Frame {currentFrame + 1} of {poseData?.length || 0}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* 3D viewport container */}
        <div
          ref={containerRef}
          className="w-full h-[500px] rounded-md overflow-hidden bg-gradient-to-b from-gray-50 to-gray-100 border"
        />

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
