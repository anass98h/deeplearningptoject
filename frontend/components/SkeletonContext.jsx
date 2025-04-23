"use client";

import React, { createContext, useContext, useState } from "react";

// Create context for skeleton visualization state
const SkeletonContext = createContext();

// Define the skeleton connections for PoseNet data
export const POSE_CONNECTIONS = [
  // Upper body
  ["nose", "left_eye"],
  ["nose", "right_eye"],
  ["left_eye", "left_ear"],
  ["right_eye", "right_ear"],
  ["nose", "neck"],

  // Torso
  ["neck", "left_shoulder"],
  ["neck", "right_shoulder"],
  ["left_shoulder", "right_shoulder"],
  ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"],
  ["left_hip", "right_hip"],

  // Arms - standard PoseNet naming
  ["left_shoulder", "left_elbow"],
  ["right_shoulder", "right_elbow"],
  ["left_elbow", "left_wrist"],
  ["right_elbow", "right_wrist"],

  // Arms - alternative naming that might be in your CSV
  ["left_elbow", "left_hand"],
  ["right_elbow", "right_hand"],
  ["left_wrist", "left_hand"],
  ["right_wrist", "right_hand"],

  // Legs - standard PoseNet naming
  ["left_hip", "left_knee"],
  ["right_hip", "right_knee"],
  ["left_knee", "left_ankle"],
  ["right_knee", "right_ankle"],

  // Legs - alternative naming that might be in your CSV
  ["left_knee", "left_foot"],
  ["right_knee", "right_foot"],
  ["left_ankle", "left_foot"],
  ["right_ankle", "right_foot"],
];

// Helper functions
export function checkIfNeedsScaling(joints) {
  let maxX = 0,
    maxY = 0;

  Object.values(joints).forEach((joint) => {
    if (joint.x !== undefined && Math.abs(joint.x) > maxX)
      maxX = Math.abs(joint.x);
    if (joint.y !== undefined && Math.abs(joint.y) > maxY)
      maxY = Math.abs(joint.y);
  });

  // If maximum coordinates are small (less than 5), they're likely normalized and need scaling
  const needsScaling = maxX < 5 && maxY < 5;

  // Determine appropriate scale factor based on coordinate range
  let recommendedScale = 2.0; // default

  if (maxX < 1.1 && maxY < 1.1) {
    // Likely normalized 0-1 coordinates
    recommendedScale = 4.0;
  } else if (maxX < 10 && maxY < 10) {
    // Small coordinate range
    recommendedScale = 1.0;
  }

  console.log(
    `Coordinate range - Max X: ${maxX.toFixed(2)}, Max Y: ${maxY.toFixed(
      2
    )}, Recommended scale: ${recommendedScale}`
  );

  return needsScaling ? recommendedScale : 0.5; // Return actual scale factor instead of boolean
}

export function checkIfNeedsYFlip(joints) {
  if (Object.keys(joints).length === 0) return false;

  // Usually in skeletal systems, Y points up (head has higher Y than feet)
  // Check if nose/head is above hips/feet
  const headJoint = joints.nose || joints.head;
  const footJoint =
    joints.left_ankle ||
    joints.right_ankle ||
    joints.left_foot ||
    joints.right_foot;

  if (headJoint && footJoint && headJoint.y < footJoint.y) {
    // Head has lower Y value than feet, so we need to flip Y
    return true;
  }

  return false;
}

// Provider component
export function SkeletonProvider({ children }) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [autoRotate, setAutoRotate] = useState(false); // Default rotation off
  const [debug, setDebug] = useState(null);
  const [hasSkeletonData, setHasSkeletonData] = useState(false);

  // Toggle auto-rotation
  const toggleAutoRotate = () => {
    setAutoRotate(!autoRotate);
  };

  // Functions to handle playback
  const handleSpeedChange = (newSpeed) => {
    setSpeed(newSpeed[0]);
  };

  const value = {
    isPlaying,
    setIsPlaying,
    currentFrame,
    setCurrentFrame,
    speed,
    setSpeed,
    autoRotate,
    setAutoRotate,
    toggleAutoRotate,
    debug,
    setDebug,
    hasSkeletonData,
    setHasSkeletonData,
    handleSpeedChange,
  };

  return (
    <SkeletonContext.Provider value={value}>
      {children}
    </SkeletonContext.Provider>
  );
}

// Custom hook to use skeleton context
export function useSkeletonContext() {
  const context = useContext(SkeletonContext);
  if (context === undefined) {
    throw new Error(
      "useSkeletonContext must be used within a SkeletonProvider"
    );
  }
  return context;
}
