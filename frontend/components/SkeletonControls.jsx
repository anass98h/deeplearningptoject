import React from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { PlayCircle, PauseCircle, RotateCcw } from "lucide-react";
import { useSkeletonContext } from "./SkeletonContext";

export function SkeletonControls({
  totalFrames = 0,
  showSideBySide = false,
  groundTruthFrames = 0,
  predictedFrames = 0,
}) {
  // Get state and functions from context
  const {
    isPlaying,
    setIsPlaying,
    currentFrame,
    setCurrentFrame,
    autoRotate,
    toggleAutoRotate,
  } = useSkeletonContext();

  // Calculate the maximum frame index
  const maxFrames = showSideBySide
    ? Math.min(groundTruthFrames, predictedFrames)
    : totalFrames;

  // Handle frame scrubbing
  const handleFrameChange = (value) => {
    const frameIndex = Math.min(Math.max(0, value[0]), maxFrames - 1);
    setCurrentFrame(frameIndex);
  };

  // Simplified controls with just play/pause and rotation toggle
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
              <PauseCircle className="h-5 w-5 text-blue-600" />
            ) : (
              <PlayCircle className="h-5 w-5 text-blue-600" />
            )}
          </Button>

          <Button
            variant="outline"
            onClick={toggleAutoRotate}
            className={`transition-all duration-200 ${
              autoRotate ? "bg-blue-100 text-blue-700" : "hover:bg-blue-50"
            }`}
          >
            <RotateCcw className="h-4 w-4 mr-2" />
            {autoRotate ? "Rotation On" : "Rotation Off"}
          </Button>
        </div>

        <div className="text-sm text-gray-600">
          Frame: {currentFrame + 1} / {maxFrames}
        </div>
      </div>

      {/* Frame slider */}
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
}
