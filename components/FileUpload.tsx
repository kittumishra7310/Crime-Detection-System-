"use client"

import React, { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Upload, FileVideo, FileImage, X, CheckCircle, AlertTriangle } from "lucide-react"
import { apiService } from "@/services/api"

interface DetectionResult {
  detection_id?: number
  crime_type?: string
  confidence?: number | null
  severity?: string
  message: string
  filename?: string
  success?: boolean
}

interface FileUploadProps {
  onDetectionComplete?: (result: DetectionResult) => void
}

export function FileUpload({ onDetectionComplete }: FileUploadProps) {
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])
  const [uploading, setUploading] = useState(false)
  const [results, setResults] = useState<DetectionResult[]>([])
  const [error, setError] = useState<string>("")

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setUploadedFiles(prev => [...prev, ...acceptedFiles])
    setError("")
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpg', '.jpeg', '.png', '.gif', '.bmp'],
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    },
    maxSize: 100 * 1024 * 1024, // 100MB
    multiple: true
  })

  const removeFile = (index: number) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index))
  }

  const uploadFiles = async () => {
    if (uploadedFiles.length === 0) return

    setUploading(true)
    setError("")
    const newResults: DetectionResult[] = []

    try {
      for (const file of uploadedFiles) {
        try {
          const response = await apiService.uploadFile(file, 1);
      
          if (response) {
            setResults(prev => [...prev, {
              filename: file.name,
              message: response.message,
              crime_type: response.detection?.crime_type,
              confidence: response.detection?.confidence ? parseFloat(response.detection.confidence) / 100 : null,
              severity: response.detection?.severity,
              detection_id: response.detection?.id,
              success: response.success
            }]);
          }
        } catch (err: any) {
          newResults.push({
            message: `Failed to process ${file.name}: ${err.message}`
          })
        }
      }

      setUploadedFiles([])
    } catch (err: any) {
      setError(err.message || "Upload failed")
    } finally {
      setUploading(false)
    }
  }

  const getFileIcon = (file: File) => {
    if (file.type.startsWith('video/')) {
      return <FileVideo className="h-8 w-8 text-blue-500" />
    }
    return <FileImage className="h-8 w-8 text-green-500" />
  }

  const getSeverityColor = (severity?: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-500'
      case 'high': return 'bg-orange-500'
      case 'medium': return 'bg-yellow-500'
      case 'low': return 'bg-blue-500'
      default: return 'bg-gray-500'
    }
  }

  return (
    <div className="space-y-6">
      <Card className="bg-slate-800 border-slate-700">
        <CardHeader>
          <CardTitle className="text-slate-100 flex items-center gap-2">
            <Upload className="h-5 w-5" />
            Crime Detection Upload
          </CardTitle>
          <CardDescription className="text-slate-400">
            Upload images or videos to analyze for criminal activity
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Drop Zone */}
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
              isDragActive
                ? 'border-blue-500 bg-blue-500/10'
                : 'border-slate-600 hover:border-slate-500 bg-slate-700/50'
            }`}
          >
            <input {...getInputProps()} />
            <Upload className="h-12 w-12 text-slate-400 mx-auto mb-4" />
            {isDragActive ? (
              <p className="text-blue-400">Drop the files here...</p>
            ) : (
              <div className="text-slate-400">
                <p className="mb-2">Drag & drop files here, or click to select</p>
                <p className="text-sm">Supports: JPG, PNG, MP4, AVI, MOV (max 100MB)</p>
              </div>
            )}
          </div>

          {/* File List */}
          {uploadedFiles.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-slate-200 font-medium">Selected Files:</h4>
              {uploadedFiles.map((file, index) => (
                <div key={index} className="flex items-center justify-between bg-slate-700 p-3 rounded-lg">
                  <div className="flex items-center gap-3">
                    {getFileIcon(file)}
                    <div>
                      <p className="text-slate-200 text-sm font-medium">{file.name}</p>
                      <p className="text-slate-400 text-xs">
                        {(file.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => removeFile(index)}
                    className="text-slate-400 hover:text-red-400"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              ))}
            </div>
          )}

          {/* Upload Button */}
          {uploadedFiles.length > 0 && (
            <Button
              onClick={uploadFiles}
              disabled={uploading}
              className="w-full bg-blue-600 hover:bg-blue-700"
            >
              {uploading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                  Processing Files...
                </>
              ) : (
                `Analyze ${uploadedFiles.length} File${uploadedFiles.length > 1 ? 's' : ''}`
              )}
            </Button>
          )}

          {/* Error Display */}
          {error && (
            <Alert className="bg-red-900/20 border-red-800">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription className="text-red-400">{error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Results */}
      {results.length > 0 && (
        <Card className="bg-slate-800 border-slate-700">
          <CardHeader>
            <CardTitle className="text-slate-100 flex items-center gap-2">
              <CheckCircle className="h-5 w-5 text-green-500" />
              Detection Results
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {results.map((result, index) => (
              <div key={index} className="bg-slate-700 p-4 rounded-lg">
                <div className="flex items-start justify-between mb-2">
                  <p className="text-slate-200 font-medium">{result.message}</p>
                  {result.severity && (
                    <Badge className={`${getSeverityColor(result.severity)} text-white`}>
                      {result.severity.toUpperCase()}
                    </Badge>
                  )}
                </div>
                
                {result.crime_type && (
                  <div className="space-y-2 mt-3">
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-400">Crime Type:</span>
                      <span className="text-slate-200">{result.crime_type}</span>
                    </div>
                    {result.confidence && (
                      <div className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span className="text-slate-400">Confidence:</span>
                          <span className="text-slate-200">{(result.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <Progress 
                          value={result.confidence * 100} 
                          className="h-2 bg-slate-600"
                        />
                      </div>
                    )}
                    {result.detection_id && (
                      <div className="flex justify-between text-sm">
                        <span className="text-slate-400">Detection ID:</span>
                        <span className="text-slate-200">#{result.detection_id}</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  )
}
