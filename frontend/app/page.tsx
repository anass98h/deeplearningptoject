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
      <div className="w-full max-w-4xl">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-extrabold tracking-tight">
            ML Prediction Dashboard
          </h1>
          <p className="mt-3 text-lg text-white/70">
            Upload data and run machine learning predictions with ease
          </p>
        </div>

        {/* Tabs container */}
        <Tabs
          defaultValue="video-trimmer"
          onValueChange={(value: string) => setActiveTab(value)}
        >
          <TabsList
            className="
            flex space-x-4 mb-6
            bg-black/20
            p-1 rounded-full
          "
          >
            <TabsTrigger
              value="video-trimmer"
              className={`
              flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-colors duration-200
              ${activeTab === "video-trimmer"
                  ? "bg-gradient-to-br from-purple-600 to-pink-500 text-white"
                  : "text-white/70 hover:bg-white/10"}
            `}
            >
              <Upload className="h-4 w-4" />
              Video Trimmer
            </TabsTrigger>
          </TabsList>
        </Tabs>

        {/* Video Trimmer Content */}
        {activeTab === "video-trimmer" && (
          <div className="bg-[#1C1534] rounded-2xl p-6 border border-white/10 shadow-lg">
            {/* Header for Video Trimmer */}
            <div className="mb-6 text-center">
              <h3 className="text-2xl font-semibold text-white">Video Trimmer</h3>
              <p className="mt-1 text-sm text-gray-300">
                Upload a video to process
              </p>
            </div>

            {/* Video Trimmer Component */}
            {/* This is where the VideoTrimmer component will be rendered */}
            <VideoTrimmer />
          </div>
        )}
      </div>
    </div>
  );

}
