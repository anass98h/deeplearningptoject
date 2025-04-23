"use client";

import { useEffect } from "react";
import { useSkeletonContext } from "./SkeletonContext";

export function AnimationManager({
  poseData = [],
  predictedData = [],
  showSideBySide = false,
}) {
  // Get state from context
  const { isPlaying, setCurrentFrame, currentFrame, speed } =
    useSkeletonContext();

  // Setup animation loop
  useEffect(() => {
    if (
      !poseData ||
      poseData.length === 0 ||
      (showSideBySide && (!predictedData || predictedData.length === 0))
    ) {
      return;
    }

    let animationFrameId;
    let lastTime = 0;
    const frameTime = 1000 / 30; // 30 fps

    const updateFrame = (timestamp) => {
      if (!isPlaying) return;
      const elapsed = timestamp - lastTime;

      if (elapsed > frameTime / speed) {
        lastTime = timestamp;
        setCurrentFrame((prev) => {
          const maxFrames = showSideBySide
            ? Math.min(poseData.length, predictedData.length)
            : poseData.length;
          const nextFrame = (prev + 1) % maxFrames;
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
  }, [
    isPlaying,
    poseData,
    predictedData,
    speed,
    showSideBySide,
    setCurrentFrame,
  ]);

  // We don't need to render anything - this is just for managing animation state
  return null;
}
