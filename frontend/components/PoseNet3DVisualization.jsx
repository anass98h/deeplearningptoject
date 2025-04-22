// PoseNet3DVisualization.jsx
"use client";

import React, { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

// Import components
import { SkeletonProvider, useSkeletonContext } from "./SkeletonContext";
import { SkeletonRenderer } from "./SkeletonRenderer";
import { SkeletonControls } from "./SkeletonControls";
import { AnimationManager } from "./AnimationManager";

export function PoseNet3DVisualization({
  poseData = [],
  predictedData = [],
  showSideBySide = false,
  groundTruthLabel = "Sample Pose",
  predictedLabel = "Prediction",
}) {
  return (
    <SkeletonProvider>
      <PoseNetVisualizationContent
        poseData={poseData}
        predictedData={predictedData}
        showSideBySide={showSideBySide}
        groundTruthLabel={groundTruthLabel}
        predictedLabel={predictedLabel}
      />
    </SkeletonProvider>
  );
}

// Inner component with access to skeleton context
function PoseNetVisualizationContent({
  poseData,
  predictedData,
  showSideBySide,
  groundTruthLabel,
  predictedLabel,
}) {
  // Access shared skeleton state from context
  const { hasSkeletonData } = useSkeletonContext();

  // Track render completion for synchronized initialization
  const [leftRendered, setLeftRendered] = useState(false);
  const [rightRendered, setRightRendered] = useState(false);

  // Toggle for overlapping skeletons display mode
  const [showOverlapping, setShowOverlapping] = useState(!showSideBySide);

  return (
    <Card className="shadow-md border-blue-100">
      <CardContent className="p-6">
        <div className="space-y-4">
          {/* Animation manager (hidden component that handles animation logic) */}
          <AnimationManager
            poseData={poseData}
            predictedData={predictedData}
            showSideBySide={showSideBySide && !showOverlapping}
          />

          {/* Display mode toggle */}
          {predictedData && predictedData.length > 0 && (
            <div className="flex items-center space-x-2 mb-4">
              <Switch
                id="display-mode"
                checked={showOverlapping}
                onCheckedChange={setShowOverlapping}
                disabled={!showSideBySide}
              />
              <Label htmlFor="display-mode" className="cursor-pointer">
                {showOverlapping
                  ? "Showing overlapping skeletons"
                  : showSideBySide
                  ? "Showing side-by-side view"
                  : "Showing single skeleton"}
              </Label>
            </div>
          )}

          {/* Skeleton visualization area */}
          {showSideBySide && !showOverlapping ? (
            // Side-by-side view
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-center text-lg font-medium text-blue-700 mb-2">
                  {groundTruthLabel}
                </div>
                <SkeletonRenderer
                  poseData={poseData}
                  isGroundTruth={true}
                  label={groundTruthLabel}
                  onRenderComplete={() => setLeftRendered(true)}
                />
              </div>
              <div>
                <div className="text-center text-lg font-medium text-red-700 mb-2">
                  {predictedLabel}
                </div>
                <SkeletonRenderer
                  poseData={predictedData}
                  isGroundTruth={false}
                  label={predictedLabel}
                  onRenderComplete={() => setRightRendered(true)}
                />
              </div>
            </div>
          ) : (
            // Overlapping view or single skeleton view
            <div>
              <div className="text-center text-lg font-medium text-blue-700 mb-2">
                {predictedData && predictedData.length > 0 && showOverlapping
                  ? `${groundTruthLabel} vs ${predictedLabel}`
                  : groundTruthLabel}
              </div>
              <SkeletonRenderer
                poseData={poseData}
                isGroundTruth={true}
                label={groundTruthLabel}
                comparisonPoseData={showOverlapping ? predictedData : null}
                onRenderComplete={() => {
                  setLeftRendered(true);
                  setRightRendered(true); // Both are rendered in the same view
                }}
              />
            </div>
          )}

          {/* Warning if no skeleton data is found */}
          {!hasSkeletonData && poseData && poseData.length > 0 && (
            <Alert variant="warning" className="bg-yellow-50 border-yellow-100">
              <AlertCircle className="h-4 w-4 text-yellow-600" />
              <AlertDescription className="text-yellow-700">
                No skeleton data could be rendered. This might be due to:
                <ul className="list-disc pl-5 mt-2 space-y-1">
                  <li>
                    Joint names in your CSV don't match expected format (need
                    _x, _y, _z suffixes)
                  </li>
                  <li>Coordinates might need scaling or transformation</li>
                  <li>Missing key joint data</li>
                </ul>
              </AlertDescription>
            </Alert>
          )}

          {/* Skeleton color legend is now built into the 3D view */}
          {/* Removed separate legend here since it's built into the 3D view */}

          {/* Controls */}
          {leftRendered && (!showSideBySide || rightRendered) && (
            <SkeletonControls
              totalFrames={poseData?.length || 0}
              showSideBySide={showSideBySide && !showOverlapping}
              groundTruthFrames={poseData?.length || 0}
              predictedFrames={predictedData?.length || 0}
            />
          )}
        </div>
      </CardContent>
    </Card>
  );
}
