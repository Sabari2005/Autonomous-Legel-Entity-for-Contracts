import React, { useState, useEffect, useRef, useContext } from "react";
import axios from "axios";
import Notification from "./Notification"; // Importing the notification component
import { Viewer } from "@react-pdf-viewer/core";
import "@react-pdf-viewer/core/lib/styles/index.css";
import ExtractedTextComponent from "./ExtractedTextComponent";
import Loading from "../assets/loading.json";
import Lottie from "lottie-react";
import { ExtractedTextContext } from "../ExtractedTextContext";
import ProcessLoader from "./ProcessLoader";
import { AuthContext } from "../AuthContext";
function RiskAnalysisLeft({ setFileUrl }) {
  const link = "https://8000-01jrcj02963afp5r8dxtpsxdqz.cloudspaces.litng.ai";
  const [selectedFile, setSelectedFile] = useState(null);
  const [progress, setProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const { isUploaded, setIsUploaded } = useContext(ExtractedTextContext);
  const { userDetails } = useContext(AuthContext);
  const [notification, setNotification] = useState({ message: "", color: "" });
  const [isDragging, setIsDragging] = useState(false);
  const [pdfUrl, setPdfUrl] = useState(""); // Store PDF URL for rendering
  const { originalTextUrl, setOriginalTextUrl } = useContext(ExtractedTextContext);
  const fileInputRef = useRef(null);
  const abortControllerRef = useRef(null);

  const {
    extractedExplainedText,
    setExtractedExplainedText,
    extractedOriginalText,
    setExtractedOriginalText,
    extractedModifiedText,
    setExtractedModifiedText,
    KgContent,setKgContent, setSidebarOperation
  } = useContext(ExtractedTextContext);

  const reset = () => {
    setSelectedFile(null);
    setIsUploaded(false);
    setProgress(0);
    setExtractedExplainedText("");
    setExtractedOriginalText("");
    setExtractedModifiedText("");
    setNotification({ message: "", color: "" });
    setPdfUrl("");
    setOriginalTextUrl("");
    setKgContent("");

  };

  const handleFileChange = (event) => {
    if (isUploading) return;
  
    const file = event.target.files[0];
    if (!file) return null;
  
    const allowedExtensions = ["pdf", "docx","png","jpg","jpeg"];
    const fileExtension = file.name.split(".").pop().toLowerCase();
  
    if (!allowedExtensions.includes(fileExtension)) {
      setNotification({ message: "Unsupported file format!", color: "red" });
      return null;
    }
  
    setSelectedFile(file);
    return file;
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => setIsDragging(false);

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);

    const file = event.dataTransfer.files[0];
    if (!file) return;

    const allowedExtensions = ["pdf", "docx","png","jpg","jpeg"];
    const fileExtension = file.name.split(".").pop().toLowerCase();

    if (!allowedExtensions.includes(fileExtension)) {
      setNotification({ message: "Unsupported file format!", color: "red" });
      return;
    }

    setSelectedFile(file);
  };

  const handleUpload = async (file) => {
    if (!file) {
      setNotification({ message: "Please select a file first.", color: "orange" });
      return;
    }
  
    setUploadStatus("Processing..");
    setIsUploading(true);
    setProgress(0);
    setSidebarOperation(true);
    abortControllerRef.current = new AbortController();
  
    const formData = new FormData();
    formData.append("file", file); // use the passed file
  
    try {
      const response = await axios.post(`${link}/upload/`, formData, {
        headers: { "Content-Type": "multipart/form-data",'userid': userDetails.sub },
        signal: abortControllerRef.current.signal,
        onUploadProgress: (progressEvent) => {
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setProgress(percent);
        },
        
      });
      if (response.status === 200) {
        setUploadStatus("Upload Complete");
        setIsUploaded(true);
        setSidebarOperation(false);
        const { explained_text, original_text, modified_text, htmlUrl, kg_out } = response.data;
  
        setExtractedExplainedText(explained_text);
        setExtractedOriginalText(original_text);
        setExtractedModifiedText(modified_text);
        if (kg_out) {
          const fullKgUrl = `${link}${kg_out}`;
          console.log("Full kg URL:", fullKgUrl);
          // setFileUrl(fullFileUrl);
          // setPdfUrl(fullFileUrl);
          // setOriginalTextUrl(fullFileUrl);
          setKgContent(fullKgUrl);
        } else {
          setNotification({ message: "Error: No file URL received!", color: "red" });
        }
  
        if (htmlUrl) {
          const fullFileUrl = `${link}${htmlUrl}`;
          console.log("Full file URL:", fullFileUrl);
          setFileUrl(fullFileUrl);
          setPdfUrl(fullFileUrl);
          setOriginalTextUrl(fullFileUrl);
        } else {
          setNotification({ message: "Error: No file URL received!", color: "red" });
        }
  
        setNotification({ message: "File uploaded successfully!", color: "green" });
      } else {
        throw new Error("Server responded with an error");
      }
    } catch (error) {
      if (axios.isCancel(error)) {
        setNotification({ message: "Upload canceled!", color: "gray" });
      } else {
        setNotification({ message: "Error in server. Please try again.", color: "red" });
      }
    } finally {
      setIsUploading(false);
      setSidebarOperation(false);
    }
  };
  
  const handleFileSelect = (event) => {
    if (isUploading) return;
  
    const file = event.target.files[0];
    if (!file) return;
  
    const allowedExtensions = ["pdf", "docx","png","jpg","jpeg"];
    const fileExtension = file.name.split(".").pop().toLowerCase();
  
    if (!allowedExtensions.includes(fileExtension)) {
      setNotification({ message: "Unsupported file format!", color: "red" });
      return;
    }
  
    handleUpload(file); // Directly pass the valid file
  };
  
  return (
    <div className="RiskAnalysisLeft" style={{overflow:'scroll'}}>
      
      
      {notification.message && (
        <Notification
          message={notification.message}
          color={notification.color}
          onClose={() => setNotification({ message: "", color: "" })}
        />
      )}

      {!isUploaded ? (
        <>
          {!isUploading && (
            <div
              className={`upload-container ${isDragging ? "dragging" : ""}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <input
  type="file"
  ref={fileInputRef}
  onChange={handleFileSelect}
  style={{ display: "none" }}
  disabled={isUploading}
/>
              <div className="UploadSuperContainer">
                <span>Upload</span>
                <div
                  className={`upload-box ${isUploading ? "disabled" : ""}`}
                  onClick={() => !isUploading && fileInputRef.current.click()}
                >
                  <img src="/upload.svg" alt="upload" height="90" width="90" />
                  <span style={{ fontSize: "13px", color: "grey" }}>
                    {isDragging ? (
                      <span style={{ color: "green" }}>Drop the file here...</span>
                    ) : (
                      <>
                        <span style={{ color: "blue" }}>Click here</span> or drag & drop your file to analyze risk!
                      </>
                    )}
                  </span>
                </div>
                <span className="file-name" style={{width:"100%",textAlign:"center"}}>
                  Selected File: {selectedFile ? selectedFile.name : "None"}
                </span>
              </div>

              

              {/* {!isUploading && (
                <button className="upload_btn" onClick={handleUpload}>
                  Upload
                </button>
              )} */}
            </div>
          )}

          {isUploading && (
            <div className="progress-container">
              <ProcessLoader />
              <span>{uploadStatus} {progress < 100 ? `${progress}%` : ""}</span>
              {/* <button className="upload_btn cancel-btn" onClick={() => abortControllerRef.current.abort()}>
                Cancel
              </button> */}
            </div>
          )}
        </>
      ) : (

        
        <ExtractedTextComponent
          fileUrl={pdfUrl}
          selectedFile={selectedFile}
          originalText={extractedOriginalText}
          modifiedText={extractedModifiedText}
          reset={reset}
          link={link}
        />
      )}
    </div>
  );
}

export default RiskAnalysisLeft;
