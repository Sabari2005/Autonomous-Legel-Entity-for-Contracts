import React, { useState, useEffect, useRef, useContext } from "react";
import ExplainedTextAccordion from "./ExplainedTextAccordian";
import { pdfjs } from "pdfjs-dist";
import * as docxPreview from "docx-preview";
import { ExtractedTextContext } from "../ExtractedTextContext";
import "./Style.css";
import { CloseOutlined } from '@ant-design/icons'

function RiskAnalysisRight({ extractedExplainedText, extractedOriginalText, extractedModifiedText, fileUrl }) {
  const [openAccordion, setOpenAccordion] = useState("explained");
  const [loading, setLoading] = useState(false);
  const [fileType, setFileType] = useState(null);
  const docxContainerRef = useRef(null);
  const [isPopupOpen, setIsPopupOpen] = useState(false);
  const handleVisualize = () => {
    setIsPopupOpen(true);
  };

  // ✅ Extract values from Context
  const {
    extractedOriginalText: storedOriginal,
    setExtractedOriginalText,
    extractedModifiedText: storedModified,
    setExtractedModifiedText,
    extractedExplainedText: storedExplained,
    setExtractedExplainedText,
    KgContent
  } = useContext(ExtractedTextContext);
  const { originalTextUrl, setOriginalTextUrl } = useContext(ExtractedTextContext);

  // ✅ Update extracted text in Context ONLY when props change
  useEffect(() => {
    if (extractedOriginalText && extractedOriginalText !== storedOriginal) {
      setExtractedOriginalText(extractedOriginalText);
    }
    if (extractedModifiedText && extractedModifiedText !== storedModified) {
      setExtractedModifiedText(extractedModifiedText);
    }
    if (extractedExplainedText && extractedExplainedText !== storedExplained) {
      setExtractedExplainedText(extractedExplainedText);
    }
  }, [extractedOriginalText, extractedModifiedText, extractedExplainedText, storedOriginal, storedModified, storedExplained]);

  // ✅ Handle File Upload & Type Detection (Optimized)
  useEffect(() => {
    if (!originalTextUrl) return;

    setLoading(true);
    const fileExtension = originalTextUrl.split(".").pop().toLowerCase();
    setFileType(fileExtension);

    if (fileExtension === "pdf") {
      renderPDF(originalTextUrl);
    } else if (fileExtension === "docx") {
      renderDOCX(originalTextUrl);
    } else {
      setExtractedOriginalText("Unsupported file format.");
      setLoading(false);
    }

    return () => {
      // Cleanup: Reset loading and clear previous content
      setLoading(false);
    };
  }, [fileUrl]);

  // ✅ Detect Large Responses & Delay State Update for Smooth UI
  useEffect(() => {
    const isLargeResponse = [extractedOriginalText, extractedModifiedText, extractedExplainedText]
      .some((text) => text && text.length > 1000); // Adjust threshold
  
    if (isLargeResponse) {
      setIsPopupOpen(true);
      setTimeout(() => setIsPopupOpen(false), 2000);
    }
  }, [extractedOriginalText, extractedModifiedText, extractedExplainedText]);
  

  

  // ✅ Handle Accordion Toggle
  const toggleAccordion = (accordion) => {
    setOpenAccordion((prev) => (prev === accordion ? null : accordion));
  };

  const [pdfRendered, setPdfRendered] = useState(false);

useEffect(() => {
  if (openAccordion === "original" && !pdfRendered) {
    setPdfRendered(true);
  }
}, [openAccordion]);

  return (
    <div className="RiskAnalysisRight">
      

      <div className={`accordion ${openAccordion === "original" ? "expanded" : "collapsed"}`}>
        <button
          className="accordion-btn original"
          style={{ background: "#E1D39C", color: "#997A00" }}
          onClick={() => toggleAccordion("original")}
        >
          Original Text <span className={`arrow ${openAccordion === "original" ? "rotate" : ""}`}>▼</span>
        </button>
        {openAccordion === "original" ? (
          <div className="accordion-content">
            <iframe
                id="pdfViewer"
                src={pdfRendered ? originalTextUrl : ""}
                width="100%"
                height="570px"
                style={{ display: pdfRendered ? "block" : "none" }}
              />
          </div>
        ) : (
          <div className="preview-text">{""}</div>
        )}
      </div>

      <div className={`accordion ${openAccordion === "modified" ? "expanded" : "collapsed"}`}>
        <button
          className="accordion-btn modified"
          style={{ background: "#D1FAE5", color: "#2F855A" }}
          onClick={() => toggleAccordion("modified")}
        >
          Modified Text <span className={`arrow ${openAccordion === "modified" ? "rotate" : ""}`}>▼</span>
        </button>
        {openAccordion === "modified" ? (
          <div className="accordion-content">
            <div dangerouslySetInnerHTML={{ __html: storedModified }} />
          </div>
        ) : (
          <div className="preview-text">
            {(typeof storedModified === "string" ? storedModified.replace(/<\/?[^>]+(>|$)/g, "").slice(0, 50) : "") + ""}
          </div>
        )}
      </div>

      <div className={`accordion explained ${openAccordion === "explained" ? "expanded" : "collapsed"}`}>
        <button
          className="accordion-btn explained"
          style={{ display:"flex",justifyContent:"space-between",background: "lightblue", color: "black" }}
          onClick={() => toggleAccordion("explained")}
        >
          <div>Explained Text <span className={`arrow ${openAccordion === "explained" ? "rotate" : ""}`}>▼</span></div>
          {storedExplained && (
            <div className="knowledgeGraph">
              <button className="wantHover" onClick={handleVisualize}>Visualize</button>
            </div>
          )}
          </button>
        
        {openAccordion === "explained" ? (
          <ExplainedTextAccordion extractedExplainedText={storedExplained || ""} />
        ) : (
          <div className="preview-text">
            {""}
          </div>
        )}
      </div>
      {isPopupOpen && (
        <div className="popup-overlay">
          <div className="popup-content">
            <div style={{width:"100%",display:"flex",justifyContent:"flex-end"}}><button className="wantHover" onClick={() => setIsPopupOpen(false)}>
  <CloseOutlined /> 
</button></div>  
            <div className="visualize">
            <iframe
                id="fkgViewer"
                src={KgContent}
                width="100%"
                height="500px"
                style={{ width:"100%", height:"100%"}}
              />

          
            </div>
            
          </div>
        </div>
      )}
    </div>
  );
}

export default RiskAnalysisRight;
