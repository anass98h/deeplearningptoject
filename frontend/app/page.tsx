import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { CSVPrediction } from "@/components/csv-prediction";
import PoseNetPrediction from "@/components/posenet-prediction";
import { FrameTrimmer } from "@/components/FrameTrimmer";
import { VideoFrameTrimmer } from "@/components/VideoFrameTrimmer";


export const dynamic = "force-dynamic";
export const revalidate = 0;

export default function Dashboard() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-gray-50 to-gray-100">
      <div className="container mx-auto py-10 px-4 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold bg-gradient-to-r from-gray-800 to-gray-900 bg-clip-text text-transparent">
            ML Prediction Dashboard
          </h1>
          <p className="text-gray-600 mt-2">
            Upload data and run machine learning predictions with ease
          </p>
        </div>

        <Tabs defaultValue="csv" className="w-full">
          <TabsList className="grid w-full max-w-xl grid-cols-4 mb-8 p-1 bg-gray-100 rounded-lg">
            <TabsTrigger
              value="csv"
              className="rounded-md data-[state=active]:bg-white data-[state=active]:shadow-md transition-all duration-200"
            >
              Predictions
            </TabsTrigger>
            <TabsTrigger
              value="posenet"
              className="rounded-md data-[state=active]:bg-white data-[state=active]:shadow-md transition-all duration-200"
            >
              PoseNet
            </TabsTrigger>
            <TabsTrigger
              value="trimmer"
              className="rounded-md data-[state=active]:bg-white data-[state=active]:shadow-md transition-all duration-200"
            >
              Frame Trimmer
            </TabsTrigger>
            <TabsTrigger
              value="video"
              className="rounded-md data-[state=active]:bg-white data-[state=active]:shadow-md transition-all duration-200"
            >
              Video Analysis
            </TabsTrigger>
          </TabsList>

          <TabsContent value="csv" className="mt-0">
            <Card className="border-0 shadow-xl overflow-hidden">
              <CardHeader className="border-b border-gray-100 bg-white px-6 py-4">
                <CardTitle className="text-xl font-semibold text-gray-800">
                  CSV-based Predictions
                </CardTitle>
                <CardDescription className="text-gray-500 mt-1">
                  Upload a CSV file to perform regression or classification
                  predictions.
                </CardDescription>
              </CardHeader>
              <CardContent className="p-0">
                <CSVPrediction />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="posenet" className="mt-0">
            <Card className="border-0 shadow-xl overflow-hidden">
              <CardHeader className="border-b border-gray-100 bg-white px-6 py-4">
                <CardTitle className="text-xl font-semibold text-gray-800">
                  PoseNet Predictions
                </CardTitle>
                <CardDescription className="text-gray-500 mt-1">
                  Use your webcam for real-time pose detection and predictions.
                </CardDescription>
              </CardHeader>
              <CardContent className="p-0">
                <PoseNetPrediction />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="trimmer" className="mt-0">
            <Card className="border-0 shadow-xl overflow-hidden">
              <CardHeader className="border-b border-gray-100 bg-white px-6 py-4">
                <CardTitle className="text-xl font-semibold text-gray-800">
                  Frame Trimmer
                </CardTitle>
                <CardDescription className="text-gray-500 mt-1">
                  Upload a CSV file to remove unwanted frames and visualize the
                  results.
                </CardDescription>
              </CardHeader>
              <CardContent className="p-6">
                <FrameTrimmer />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="video" className="mt-0">
            <Card className="border-0 shadow-xl overflow-hidden">
              <CardHeader className="border-b border-gray-100 bg-white px-6 py-4">
                <CardTitle className="text-xl font-semibold text-gray-800">
                  Video Analysis
                </CardTitle>
                <CardDescription className="text-gray-500 mt-1">
                  Upload a video file for pose estimation and frame analysis.
                </CardDescription>
              </CardHeader>
              <CardContent className="p-6">
                <VideoFrameTrimmer />
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
