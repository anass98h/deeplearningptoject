import type React from "react"
import "./globals.css"
import { Inter } from "next/font/google"

const inter = Inter({ subsets: ["latin"] })

export const metadata = {
  title: "ML Prediction Dashboard",
  description: "Upload data and run machine learning predictions with ease",
}

export default function RootLayout({ children }: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={inter.className}>
      <body
        className="
          bg-gradient-to-b from-[#1F0C3A] via-[#2A0F4D] to-[#130B26]
          text-white min-h-screen"
      >
        {children}
      </body>
    </html>
  )
}
