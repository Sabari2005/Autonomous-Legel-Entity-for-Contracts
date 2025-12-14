import React, { useEffect, useState, useContext } from "react";
import ReactMarkdown from "react-markdown";
import { ExtractedTextContext } from "../ExtractedTextContext";
import "./Style.css";
import MarkdownRenderer from "./Markdown";
import ImageGallery from "./ImageGallery";
import { FiCopy, FiCheck } from 'react-icons/fi'; // Feather Icons
import { ArrowLeftOutlined } from '@ant-design/icons';
import CaseDetails from "./CAseDetails";
import ChatProcessLoader from "./ChatProcessLoader";
import { FiDownload } from "react-icons/fi";
import { DownloadOutlined } from '@ant-design/icons';
import { FaDownload } from "react-icons/fa";

function CompanyDeligenceReport() {
  const baseLink = "https://8000-01jrcj02963afp5r8dxtpsxdqz.cloudspaces.litng.ai";
  const { CpyDelReportPath,setCpyDelReportPath, CpyDelImgPaths,setCpyDelImgPaths, FinalReport } = useContext(ExtractedTextContext);
  const [markdownContent, setMarkdownContent] = useState("");
  const [selectedImage, setSelectedImage] = useState(null);
  const [copied, setCopied] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const handleMessage = (event) => {
      if (event.data.iframeHeight) {
        const iframe = document.getElementById('report-iframe');
        if (iframe) {
          iframe.style.height = event.data.iframeHeight + 'px';
        }
      }
    };
    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, []);
  
  const handleCopy = () => {
    navigator.clipboard.writeText(markdownContent);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
    // alert("Report copied to clipboard!");
  };
  const handleReset = () => {
    setCpyDelReportPath("");
    setCpyDelImgPaths([]);
  
    // Optional: reset context values if needed
    // If context doesn't support reset, navigate away or reload
  };
  const handleDownload = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${baseLink}/download-report?file_path=${encodeURIComponent(CpyDelReportPath)}`);
      const blob = await response.blob();
      const link = document.createElement("a");
      link.href = window.URL.createObjectURL(blob);
      link.download = "report.pdf";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (error) {
      console.error("Download failed:", error);
    } finally {
      setLoading(false);
    }
  };
  

  return (
    <div className="CompanyDeligenceReport">
      <div className="CompanyDeligenceReportContent">
      <iframe
  id="report-iframe"
  src={`${baseLink}${CpyDelReportPath}`}
  title="Risk Report"
  scrolling="no"
  style={{ width: "100%", border: "none" }}
/>
      </div>

      {/* <ImageGallery imgPaths={CpyDelImgPaths} baseLink={baseLink} /> */}



      <div className="CompanyDeligenceReportTools">
        
        <button className="back-btn" onClick={handleReset}>
            <ArrowLeftOutlined  />
        </button>
        <div style={{ display: "flex", justifyContent: "center", alignItems:"center",gap:10 }}>
          {/* <button onClick={handleCopy} className="copy-btn">
            <span className="icon">{copied ? <FiCheck color="white" /> : <FiCopy />}</span>
            <span className="text">Copy</span>
          </button> */}
          {/* <button className="copy-btn" onClick={handleDownload}>Download</button> */}
          <button className="copy-btn" onClick={handleDownload} disabled={loading}>
            {loading ? (
              <><ChatProcessLoader width="20px" height="20px" />
              Downloading ...</>
            ) : (
              <>
              <FaDownload />  
              Download
              </>
            )}
          </button>
        
        </div>
      </div>


      
      {/* <CaseDetails /> */}
      {/* <button className="back-btn" onClick={handleReset}>Try Another Company</button> */}
    </div>
  );
}

export default CompanyDeligenceReport;
