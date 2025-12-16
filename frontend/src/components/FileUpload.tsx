// frontend/src/components/FileUpload.tsx

import { useState } from "react";
import { Upload, FileText, LoaderCircle, CheckCircle, AlertCircle } from "lucide-react";
import { apiUrl } from "../config";

interface FileUploadProps {
  onTextExtracted: (text: string, qualityData?: any, medicalAnalysis?: any) => void;
}

export function FileUpload({ onTextExtracted }: FileUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    handleFiles(files);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    handleFiles(files);
  };

  const handleFiles = async (files: File[]) => {
    if (files.length === 0) return;

    // Filter for supported file types
    const supportedTypes = ["application/pdf", "image/jpeg", "image/png", "image/tiff"];
    const supportedFiles = files.filter(file => supportedTypes.includes(file.type));
    if (supportedFiles.length !== files.length) {
      setError("Please upload only PDF, JPEG, PNG, or TIFF files.");
      return;
    }

    setUploadedFiles(supportedFiles);
    setError(null);
    setIsLoading(true);

    try {
      const formData = new FormData();
      supportedFiles.forEach(file => {
        formData.append("files", file);
      });

      const response = await fetch(apiUrl("/api/v1/upload"), {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Upload failed");
      }

      const data = await response.json();
      onTextExtracted(data.extracted_text, {
        extraction_quality: data.extraction_quality,
        overall_quality: data.overall_quality,
        time_taken: data.time_taken
      }, data.medical_analysis);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("An unknown error occurred.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Upload Drug Reports</h2>
        <p className="text-gray-600">Upload PDF files to extract Drug information</p>
      </div>

      {/* Upload Area */}
      <div
        className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
          isDragOver
            ? "border-red-400 bg-red-50"
            : "border-gray-300 hover:border-red-400 hover:bg-red-50"
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          type="file"
          multiple
          accept=".pdf,.jpg,.jpeg,.png,.tiff,.tif"
          onChange={handleFileSelect}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          disabled={isLoading}
        />
        
        <div className="space-y-4">
          <div className="flex justify-center">
            {isLoading ? (
              <LoaderCircle className="h-12 w-12 text-red-600 animate-spin" />
            ) : (
              <Upload className="h-12 w-12 text-red-600" />
            )}
          </div>
          
          <div>
            <p className="text-lg font-semibold text-gray-800 mb-2">
              {isLoading ? "Processing files..." : "Drag & drop medical documents here"}
            </p>
            <p className="text-gray-600">
              {isLoading ? "Please wait while we extract text from your documents" : "or click to browse"}
            </p>
          </div>
        </div>
      </div>

      {/* Uploaded Files */}
      {uploadedFiles.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-gray-800 mb-3">Uploaded Files</h3>
          <div className="space-y-2">
            {uploadedFiles.map((file, index) => (
              <div key={index} className="flex items-center space-x-3 p-3 bg-gray-50 rounded-lg">
                <FileText className="h-5 w-5 text-red-600" />
                <span className="text-gray-700 flex-1">{file.name}</span>
                <span className="text-sm text-gray-500">
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <AlertCircle className="h-5 w-5 text-red-600" />
            <span className="text-red-800 font-medium">Error</span>
          </div>
          <p className="text-red-700 mt-1">{error}</p>
        </div>
      )}

      {/* Success Message */}
      {!isLoading && !error && uploadedFiles.length > 0 && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <CheckCircle className="h-5 w-5 text-green-600" />
            <span className="text-green-800 font-medium">Success</span>
          </div>
          <p className="text-green-700 mt-1">
            {uploadedFiles.length} file(s) uploaded successfully. Text extraction completed.
          </p>
        </div>
      )}

      {/* Instructions */}
      <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
        <h3 className="text-orange-800 font-semibold mb-2">Instructions</h3>
        <ul className="text-orange-700 text-sm space-y-1">
          <li>• Upload PDF, JPEG, PNG, or TIFF format medical documents</li>
          <li>• Multiple files can be uploaded at once</li>
          <li>• Supported: Medical summaries, prescriptions, medical reports, patient records</li>
          <li>• Handwritten and printed documents are both supported</li>
        </ul>
      </div>
    </div>
  );
}
