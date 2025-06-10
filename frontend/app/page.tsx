"use client";

import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import React, { useState } from "react";
import { Upload } from "lucide-react";
import { VideoTrimmer } from "@/components/VideoTrimmer";

export default function MLDashboard() {
  const [activeTab, setActiveTab] = useState<string>("video-trimmer");

  return (
    <div
      className="
      min-h-screen
      flex items-center justify-center p-6
      bg-gradient-to-br from-purple-900 via-indigo-900 to-pink-700
      text-white
    "
    >
      <div className="w-full max-w-7xl">
        {/* Header */}
        <div className="text-center mb-8">
          <img
            src="/logo.png"
            alt="SquatCheck Logo"
            className="mx-auto mb-4 h-48 w-48"
          />
          <h1 className="text-5xl font-extrabold tracking-tight">
            ML Prediction Dashboard
          </h1>
          <p className="mt-3 text-lg text-white/70">
            Upload data and run machine learning predictions with ease
          </p>
        </div>

        {/* Tabs container */}

        {/* Video Trimmer Content */}
        {activeTab === "video-trimmer" && (
          <div className="bg-[#1C1534] rounded-2xl p-6 border border-white/10 shadow-lg">
            {/* Header for Video Trimmer */}
            <div className="mb-6 text-center">
              <h3 className="text-2xl font-semibold text-white">
                AI-Powered Exercise Evaluation Platform
              </h3>
              <p className="mt-1 text-sm text-gray-300">
                Upload a video to process
              </p>
            </div>

            {/* Video Trimmer Component */}
            <VideoTrimmer />
          </div>
        )}
      </div>
    </div>
  );
}
