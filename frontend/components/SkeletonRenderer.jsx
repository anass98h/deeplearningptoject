// SkeletonRenderer.jsx
"use client";

import React, { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import {
  POSE_CONNECTIONS,
  checkIfNeedsScaling,
  checkIfNeedsYFlip,
  useSkeletonContext,
} from "./SkeletonContext";

export function SkeletonRenderer({
  poseData,
  isGroundTruth = true,
  label = "Skeleton",
  onRenderComplete = () => {},
  comparisonPoseData = null, // Add option for comparison pose data
}) {
  // Get shared state from context
  const { currentFrame, autoRotate, setHasSkeletonData } = useSkeletonContext();

  // Refs for THREE.js objects
  const containerRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const skeletonRef = useRef(null);
  const requestRef = useRef(null);
  const planeRef = useRef(null);
  // Add a ref to track the current rotation angle
  const rotationAngleRef = useRef(0);

  // Initialize ThreeJS scene
  useEffect(() => {
    if (!containerRef.current || !poseData || poseData.length === 0) {
      console.log("Missing container or pose data");
      return;
    }

    // Clone the primary poseData for comparison if not provided separately
    const comparisonData = comparisonPoseData || poseData;

    // Get container dimensions
    const width = containerRef.current.clientWidth;
    const height = 600; // Fixed height

    // Clear container before appending to prevent duplicates
    while (containerRef.current.firstChild) {
      containerRef.current.removeChild(containerRef.current.firstChild);
    }

    // Create scene with dark blue background instead of light grey
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(isGroundTruth ? 0x1a2639 : 0x2a1a39); // Darker themed backgrounds
    sceneRef.current = scene;

    // Create camera with fixed position
    const camera = new THREE.PerspectiveCamera(25, width / height, 0.1, 1000);
    camera.position.set(0, 1.0, 20); // Fixed camera position
    camera.lookAt(0, 1, 0); // Look at the center of the skeleton
    cameraRef.current = camera;

    // Create renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = true;
    rendererRef.current = renderer;
    containerRef.current.appendChild(renderer.domElement);

    // Add simple lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
    directionalLight.position.set(5, 10, 5);
    scene.add(directionalLight);

    // Don't add floor plane or grid - remove these entirely
    // const planeGeometry = new THREE.PlaneGeometry(8, 8);
    // const planeMaterial = new THREE.MeshStandardMaterial({
    //   color: 0x95a5a6,
    //   side: THREE.DoubleSide,
    //   transparent: true,
    //   opacity: 0.5,
    // });
    // const plane = new THREE.Mesh(planeGeometry, planeMaterial);
    // plane.rotation.x = -Math.PI / 2;
    // plane.position.y = 0;
    // scene.add(plane);
    planeRef.current = null; // Still keep the ref for compatibility

    // Don't add grid either
    // const gridHelper = new THREE.GridHelper(8, 8);
    // scene.add(gridHelper);

    // Add simple controls with manual zoom
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.enableZoom = true; // Enable manual zooming
    controls.enablePan = true; // Enable panning
    controls.autoRotate = false;
    controls.zoomSpeed = 1.0; // Normal zoom speed
    controls.minDistance = 3; // Limit how close you can zoom
    controls.maxDistance = 30; // Limit how far you can zoom out
    controlsRef.current = controls;

    // Add label to identify view
    addLabel(scene, label, isGroundTruth);

    // Add color legend
    addColorLegend(scene);

    // Animation loop with forced rotation that works during playback
    const animate = () => {
      requestRef.current = requestAnimationFrame(animate);

      // Apply rotation to skeleton regardless of playback state
      if (autoRotate && skeletonRef.current) {
        // Update rotation angle based on time for smooth animation
        rotationAngleRef.current += 0.01; // Increment the angle for smooth rotation
        skeletonRef.current.rotation.y = rotationAngleRef.current;

        // Add class for external animation handling
        skeletonRef.current.userData.isRotating = true;
        skeletonRef.current.userData.className = "skeleton-group";
      }

      controls.update();
      renderer.render(scene, camera);
    };

    animate();

    // Initial skeleton update - update both primary and comparison skeletons
    updateSkeleton(currentFrame);

    // If comparison data exists, create a comparison skeleton
    if (comparisonPoseData) {
      updateComparisonSkeleton(currentFrame);
    }

    // Signal that rendering is complete
    onRenderComplete();

    // Handle window resize
    const handleResize = () => {
      if (!containerRef.current) return;
      const newWidth = containerRef.current.clientWidth;
      camera.aspect = newWidth / height;
      camera.updateProjectionMatrix();
      renderer.setSize(newWidth, height);
    };

    window.addEventListener("resize", handleResize);

    // Cleanup
    return () => {
      window.removeEventListener("resize", handleResize);
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
      }

      if (rendererRef.current && containerRef.current) {
        containerRef.current.removeChild(rendererRef.current.domElement);
      }

      if (scene) {
        scene.traverse((object) => {
          if (object instanceof THREE.Mesh) {
            if (object.geometry) object.geometry.dispose();
            if (object.material) {
              if (Array.isArray(object.material)) {
                object.material.forEach((material) => material.dispose());
              } else {
                object.material.dispose();
              }
            }
          }
        });
      }
    };
  }, [poseData, isGroundTruth, label, onRenderComplete]);

  // Update skeleton when the current frame changes - preserve camera position
  useEffect(() => {
    if (sceneRef.current) {
      // Store current camera position and rotation before updating
      let cameraPos, cameraRot;
      if (cameraRef.current) {
        cameraPos = cameraRef.current.position.clone();
        cameraRot = cameraRef.current.rotation.clone();
      }

      // Store current controls target
      let controlsTarget;
      if (controlsRef.current) {
        controlsTarget = controlsRef.current.target.clone();
      }

      // Store current skeleton rotation before updating
      let skeletonRotation;
      if (skeletonRef.current) {
        skeletonRotation = skeletonRef.current.rotation.clone();
      }

      // Update the skeleton
      updateSkeleton(currentFrame);

      // Restore camera position and controls after update
      if (cameraRef.current && cameraPos && cameraRot) {
        cameraRef.current.position.copy(cameraPos);
        cameraRef.current.rotation.copy(cameraRot);
      }

      if (controlsRef.current && controlsTarget) {
        controlsRef.current.target.copy(controlsTarget);
      }

      // Restore skeleton rotation after update if autoRotate is enabled
      if (autoRotate && skeletonRef.current && skeletonRotation) {
        skeletonRef.current.rotation.copy(skeletonRotation);
      }
    }
  }, [currentFrame, autoRotate]);

  // Update comparison skeleton when the current frame changes
  useEffect(() => {
    if (sceneRef.current && comparisonPoseData) {
      updateComparisonSkeleton(currentFrame);
    }
  }, [currentFrame, comparisonPoseData]);

  // Sync rotation between both skeletons
  useEffect(() => {
    if (autoRotate && skeletonRef.current && comparisonSkeletonRef.current) {
      // Apply the same rotation to the comparison skeleton
      comparisonSkeletonRef.current.rotation.copy(skeletonRef.current.rotation);
    }
  }, [autoRotate, rotationAngleRef.current]);

  // Function to add a text label to the scene
  const addLabel = (
    scene,
    text,
    isGroundTruth,
    position = { x: 0, y: 3.5, z: 0 }
  ) => {
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");
    canvas.width = 256;
    canvas.height = 64;

    // Draw background
    context.fillStyle = isGroundTruth
      ? "rgba(52, 152, 219, 0.7)"
      : "rgba(231, 76, 60, 0.7)";
    context.fillRect(0, 0, canvas.width, canvas.height);

    // Draw text
    context.font = "bold 24px Arial";
    context.fillStyle = "white";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(text, canvas.width / 2, canvas.height / 2);

    // Create texture
    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.MeshBasicMaterial({
      map: texture,
      transparent: true,
      side: THREE.DoubleSide,
    });

    // Create mesh
    const geometry = new THREE.PlaneGeometry(2, 0.5);
    const label = new THREE.Mesh(geometry, material);
    label.position.set(position.x, position.y, position.z); // Use provided position
    label.rotation.x = -Math.PI / 12; // Tilt slightly

    scene.add(label);

    return label;
  };

  // Add a color legend to explain the skeleton colors
  const addColorLegend = (scene) => {
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");
    canvas.width = 512;
    canvas.height = 100;

    // Draw background
    context.fillStyle = "rgba(25, 25, 25, 0.7)";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.strokeStyle = "rgba(200, 200, 200, 0.8)";
    context.lineWidth = 2;
    context.strokeRect(0, 0, canvas.width, canvas.height);

    // Draw legend title
    context.font = "bold 24px Arial";
    context.fillStyle = "white";
    context.textAlign = "center";
    context.textBaseline = "top";
    context.fillText("Legend", canvas.width / 2, 10);

    // Draw color samples and labels
    const drawColorSample = (color, label, x) => {
      // Draw color box
      context.fillStyle = color;
      context.fillRect(x - 60, 50, 30, 30);
      context.strokeStyle = "white";
      context.strokeRect(x - 60, 50, 30, 30);

      // Draw label
      context.font = "18px Arial";
      context.fillStyle = "white";
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(label, x, 65);
    };

    // Add color samples
    drawColorSample("rgb(52, 152, 219)", "True Sample", 170);
    drawColorSample("rgb(231, 76, 60)", "Prediction", 340);

    // Create texture
    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.MeshBasicMaterial({
      map: texture,
      transparent: true,
      side: THREE.DoubleSide,
    });

    // Create mesh
    const geometry = new THREE.PlaneGeometry(3, 0.75);
    const legend = new THREE.Mesh(geometry, material);
    legend.position.set(0, 3.8, 0); // Position above the scene
    legend.rotation.x = -Math.PI / 12; // Tilt slightly

    scene.add(legend);

    return legend;
  };

  // Update skeleton mesh based on frame data
  // Reference to the comparison skeleton
  const comparisonSkeletonRef = useRef(null);

  // Update primary skeleton
  const updateSkeleton = (frameIndex) => {
    if (!poseData || poseData.length === 0 || !sceneRef.current) return;

    // Make sure the frame index is valid
    const validFrameIndex = Math.min(frameIndex, poseData.length - 1);

    // Save the current rotation if a skeleton exists
    let currentRotation = null;
    if (skeletonRef.current) {
      currentRotation = skeletonRef.current.rotation.clone();
      sceneRef.current.remove(skeletonRef.current);
    }

    const frameData = poseData[validFrameIndex];
    const skeletonGroup = new THREE.Group();
    skeletonRef.current = skeletonGroup;

    // Generate joint spheres
    const joints = {};
    // Color joints differently for ground truth vs prediction
    const pointMaterial = new THREE.MeshPhongMaterial({
      color: isGroundTruth ? 0x3498db : 0xe74c3c, // Blue for ground truth, red for prediction
    });
    const jointRadius = 0.05; // Increased joint size for better visibility

    // Extract joints from frame data
    for (const key in frameData) {
      // Check if this is a joint coordinate
      if (key.endsWith("_x") || key.endsWith("_y") || key.endsWith("_z")) {
        const baseName = key.substring(0, key.length - 2);

        if (!joints[baseName]) {
          joints[baseName] = {};
        }

        const coord = key.charAt(key.length - 1);
        joints[baseName][coord] = frameData[key];
      }
    }

    // Get appropriate scale factor for the data
    const scaleFactor = checkIfNeedsScaling(joints);

    // Simple loop to create joint spheres
    let validJointCount = 0;
    Object.keys(joints).forEach((jointName) => {
      const joint = joints[jointName];

      // Skip if we don't have enough coordinate data
      if (joint.x === undefined || joint.y === undefined) {
        return;
      }

      validJointCount++;

      // Use z if available, otherwise use 0
      const x = joint.x * scaleFactor;
      const y = joint.y * scaleFactor;
      const z = (joint.z || 0) * scaleFactor;

      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(jointRadius, 8, 8),
        pointMaterial
      );

      sphere.position.set(x, y, z);
      skeletonGroup.add(sphere);
    });

    // Update debug info state if this is the first render of ground truth
    if (validFrameIndex === 0 && isGroundTruth) {
      setHasSkeletonData(validJointCount > 0);
    }

    // Create bone connections
    POSE_CONNECTIONS.forEach((connection) => {
      const [joint1Name, joint2Name] = connection;
      const joint1 = joints[joint1Name];
      const joint2 = joints[joint2Name];

      // Skip if any joint is missing
      if (!joint1 || !joint2) return;
      if (
        joint1.x === undefined ||
        joint1.y === undefined ||
        joint2.x === undefined ||
        joint2.y === undefined
      )
        return;

      const start = new THREE.Vector3(
        joint1.x * scaleFactor,
        joint1.y * scaleFactor,
        (joint1.z || 0) * scaleFactor
      );

      const end = new THREE.Vector3(
        joint2.x * scaleFactor,
        joint2.y * scaleFactor,
        (joint2.z || 0) * scaleFactor
      );

      // Calculate bone properties
      const direction = new THREE.Vector3().subVectors(end, start);
      const distance = direction.length();

      // Skip if points are too close
      if (distance < 0.01) return;

      // Choose color based on ground truth or prediction - single color scheme
      const boneColor = isGroundTruth ? 0x3498db : 0xe74c3c; // Blue for ground truth, red for prediction

      // Create bone cylinder
      const boneMaterial = new THREE.MeshPhongMaterial({ color: boneColor });
      const boneGeometry = new THREE.CylinderGeometry(0.03, 0.03, distance, 8); // Much thicker bones for better visibility

      // Move pivot point to start of cylinder
      boneGeometry.translate(0, distance / 2, 0);
      boneGeometry.rotateX(Math.PI / 2);

      const bone = new THREE.Mesh(boneGeometry, boneMaterial);
      bone.position.copy(start);
      bone.lookAt(end);
      skeletonGroup.add(bone);
    });

    // Add skeleton to scene - no positioning adjustments and add class for rotation
    sceneRef.current.add(skeletonGroup);

    // Add class for external style targeting
    skeletonGroup.userData.className = "skeleton-group";

    // Very simple Y-flip if needed
    const needsYFlip = checkIfNeedsYFlip(joints);
    if (needsYFlip) {
      skeletonGroup.scale.y = -1;
    }

    // Restore rotation if we had one and auto-rotate is enabled
    if (currentRotation && autoRotate) {
      skeletonGroup.rotation.copy(currentRotation);
    } else if (autoRotate) {
      // If this is the first time, use the current rotation angle from ref
      skeletonGroup.rotation.y = rotationAngleRef.current;
    }
  };

  // Update comparison skeleton
  const updateComparisonSkeleton = (frameIndex) => {
    if (
      !comparisonPoseData ||
      comparisonPoseData.length === 0 ||
      !sceneRef.current
    )
      return;

    // Make sure the frame index is valid
    const validFrameIndex = Math.min(frameIndex, comparisonPoseData.length - 1);

    // Save the current rotation if a comparison skeleton exists
    let currentRotation = null;
    if (comparisonSkeletonRef.current) {
      currentRotation = comparisonSkeletonRef.current.rotation.clone();
      sceneRef.current.remove(comparisonSkeletonRef.current);
    }

    const frameData = comparisonPoseData[validFrameIndex];
    const skeletonGroup = new THREE.Group();
    comparisonSkeletonRef.current = skeletonGroup;

    // Generate joint spheres
    const joints = {};
    // Use different colors for comparison skeleton
    const pointMaterial = new THREE.MeshPhongMaterial({
      color: 0xe74c3c, // Red for prediction/comparison
      opacity: 0.9,
      transparent: true,
    });
    const jointRadius = 0.05; // Increased joint size

    // Extract joints from frame data
    for (const key in frameData) {
      // Check if this is a joint coordinate
      if (key.endsWith("_x") || key.endsWith("_y") || key.endsWith("_z")) {
        const baseName = key.substring(0, key.length - 2);

        if (!joints[baseName]) {
          joints[baseName] = {};
        }

        const coord = key.charAt(key.length - 1);
        joints[baseName][coord] = frameData[key];
      }
    }

    // Get appropriate scale factor for the data
    const scaleFactor = checkIfNeedsScaling(joints);

    // Simple loop to create joint spheres
    let validJointCount = 0;
    Object.keys(joints).forEach((jointName) => {
      const joint = joints[jointName];

      // Skip if we don't have enough coordinate data
      if (joint.x === undefined || joint.y === undefined) {
        return;
      }

      validJointCount++;

      // Use z if available, otherwise use 0
      const x = joint.x * scaleFactor;
      const y = joint.y * scaleFactor;
      const z = (joint.z || 0) * scaleFactor;

      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(jointRadius, 8, 8),
        pointMaterial
      );

      sphere.position.set(x, y, z);
      skeletonGroup.add(sphere);
    });

    // Create bone connections
    POSE_CONNECTIONS.forEach((connection) => {
      const [joint1Name, joint2Name] = connection;
      const joint1 = joints[joint1Name];
      const joint2 = joints[joint2Name];

      // Skip if any joint is missing
      if (!joint1 || !joint2) return;
      if (
        joint1.x === undefined ||
        joint1.y === undefined ||
        joint2.x === undefined ||
        joint2.y === undefined
      )
        return;

      const start = new THREE.Vector3(
        joint1.x * scaleFactor,
        joint1.y * scaleFactor,
        (joint1.z || 0) * scaleFactor
      );

      const end = new THREE.Vector3(
        joint2.x * scaleFactor,
        joint2.y * scaleFactor,
        (joint2.z || 0) * scaleFactor
      );

      // Calculate bone properties
      const direction = new THREE.Vector3().subVectors(end, start);
      const distance = direction.length();

      // Skip if points are too close
      if (distance < 0.01) return;

      // Single color for comparison skeleton
      const boneColor = 0xe74c3c; // Red for prediction/comparison

      // Create bone cylinder
      const boneMaterial = new THREE.MeshPhongMaterial({
        color: boneColor,
        opacity: 0.9,
        transparent: true,
      });
      const boneGeometry = new THREE.CylinderGeometry(0.03, 0.03, distance, 8);

      // Move pivot point to start of cylinder
      boneGeometry.translate(0, distance / 2, 0);
      boneGeometry.rotateX(Math.PI / 2);

      const bone = new THREE.Mesh(boneGeometry, boneMaterial);
      bone.position.copy(start);
      bone.lookAt(end);
      skeletonGroup.add(bone);
    });

    // Add skeleton to scene
    sceneRef.current.add(skeletonGroup);

    // Add class for external style targeting
    skeletonGroup.userData.className = "comparison-skeleton-group";

    // Very simple Y-flip if needed
    const needsYFlip = checkIfNeedsYFlip(joints);
    if (needsYFlip) {
      skeletonGroup.scale.y = -1;
    }

    // Restore rotation if we had one and auto-rotate is enabled
    if (currentRotation && autoRotate) {
      skeletonGroup.rotation.copy(currentRotation);
    } else if (autoRotate && comparisonSkeletonRef.current) {
      // Match the comparison skeleton's rotation if it exists
      skeletonGroup.rotation.copy(comparisonSkeletonRef.current.rotation);
    } else if (autoRotate) {
      // If this is the first time, use the current rotation angle from ref
      skeletonGroup.rotation.y = rotationAngleRef.current;
    }
  };

  return (
    <div
      ref={containerRef}
      className="w-full rounded-lg overflow-hidden shadow-inner border border-gray-200"
      style={{ height: "600px" }}
    />
  );
}
