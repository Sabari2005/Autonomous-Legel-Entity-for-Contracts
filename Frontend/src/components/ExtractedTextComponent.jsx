import { useState, useRef, useEffect, useContext } from "react";
import Quill from "quill";
import "quill/dist/quill.snow.css";
import {motion} from 'framer-motion'
import Notification from "./Notification"; // Import Notification Component
import { ExtractedTextContext } from "../ExtractedTextContext";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"; // Tooltip
import { AuthContext } from "../AuthContext";
import { FaEdit } from 'react-icons/fa'
import { FaDownload } from "react-icons/fa";
import { FaSave } from 'react-icons/fa'
import { blue } from "@mui/material/colors";
const ExtractedTextComponent = ({ fileUrl, selectedFile, modifiedText, reset, link }) => {
  const [modifiedContent, setModifiedContent] = useState(modifiedText || "No text extracted yet.");
  const [notification, setNotification] = useState(null);
  const [isEditing, setIsEditing] = useState(false); // Track edit mode
  const [isModVisible, setIsModVisible] = useState(false); // ✅ Track if "mod" is visible
  const editorRef = useRef(null);
  const quillRef = useRef(null);
  const [activeSection, setActiveSection] = useState("original");
  const [pdfUrl, setPdfUrl] = useState(""); 
  const { originalTextUrl, setOriginalTextUrl } = useContext(ExtractedTextContext);
  const { userDetails } = useContext(AuthContext);
  useEffect(() => {
    if (editorRef.current && !quillRef.current) {
      quillRef.current = new Quill(editorRef.current, { theme: "snow" });

      quillRef.current.on("text-change", () => {
        setModifiedContent(quillRef.current.root.innerHTML);
      });
      if (modifiedContent) {
        quillRef.current.clipboard.dangerouslyPasteHTML(0, modifiedContent);
      }

      // quillRef.current.root.innerHTML = modifiedContent;
      quillRef.current.enable(false); // Initially disable editing
    }
  }, []);

  const handleScrollToSection = (sectionId) => {
    const section = document.getElementById(sectionId);
    if (section) {
      section.scrollIntoView({ behavior: "smooth" });
      if (sectionId === "mod") {
        setIsModVisible(true); // ✅ Show Edit/Save button when "mod" is visible
      } else {
        setIsModVisible(false);
      }
    }
  };

  const handleDownload = async () => {
    if (!quillRef.current) return;

    const editedHtml = quillRef.current.root.innerHTML;

    try {
      const response = await fetch(`${link}/convert`, {
        method: "POST",
        headers: { "Content-Type": "application/json",userid:userDetails.sub },
        body: JSON.stringify({ html_content: editedHtml }),
      });

      if (!response.ok) {
        throw new Error("Failed to generate PDF");
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "modified-text.pdf";
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);

      setNotification({ message: "Download successful!", color: "green" });
    } catch (error) {
      console.error("Error generating PDF:", error);
      setNotification({ message: "Download failed!", color: "red" });
    }
  };

  const handleEditSave = () => {
    if (!quillRef.current) return;
  
    if (isEditing) {
      // If editing, save content in a variable
      const editedHtml = quillRef.current.root.innerHTML;
      setModifiedContent(editedHtml);
  
      setNotification({ message: "Content saved successfully!", color: "green" });
    }
  
    // Toggle edit mode
    setIsEditing(!isEditing);
    quillRef.current.enable(!isEditing);
  };
  
  return (
    <div style={{ padding: "5px", background: "white",height:"100%", borderRadius: "15px", position: "relative",backgroundColor:"white" }}>
      {notification && <Notification message={notification.message} color={notification.color} onClose={() => setNotification(null)} />}

      <div
        style={{
          position: "sticky",
          top: "0",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          zIndex: 100,
          background: "white",
          padding: "10px",
          borderRadius: "10px",
          boxShadow: "0px 4px 6px rgba(0, 0, 0, 0.1)",
        }}
      >
        <div style={{ display: "flex", gap: 10, background: "white", borderRadius: "30px", padding: "5px" }}>
        <motion.div
          initial={{ x: activeSection === "original" ? "5px" : "135px" }}
          animate={{ x: activeSection === "original" ? "5px" : "135px" }}
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
          style={{
            position: "absolute",
            width: "110px",
            height: "4px",
            bottom: "15px",
            background: "black",
            borderRadius: "30px",
            zIndex: 0,
          }}
        />
          <button
           onClick={() => {
            setActiveSection("original");
            handleScrollToSection("original");
          }}
            style={{
              width: "120px",
              height: "40px",
              borderRadius: "30px",
              background: "transparent",
              border: "none",
              color: "#000",
              fontWeight: "bold",
              cursor: "pointer",
            }}
          >
            Original Text
          </button>
          <button
            onClick={() => {
            setActiveSection("modified");
            handleScrollToSection("mod");
          }}
            style={{
              width: "120px",
              height: "40px",
              borderRadius: "30px",
              background: "transparent",
              border: "none",
              color: "#000",
              fontWeight: "bold",
              cursor: "pointer",
            }}
          >
            Modified Text
          </button>
        </div>
        {isModVisible && (
          <div>
            <button
              className="download-btn"
              onClick={handleEditSave}
              style={{ background: isEditing ? "green" : "#D1FAE5", color: "white", marginRight: 10 }}
            >
              {isEditing ? <div>

                <TooltipProvider>
                <Tooltip>
                    <TooltipTrigger asChild>
                           
                            <FaSave />
                        
                    </TooltipTrigger>
                    <TooltipContent>Save</TooltipContent>
                </Tooltip>
            </TooltipProvider>
              </div> : 
              <div>
<TooltipProvider>
                <Tooltip>
                    <TooltipTrigger asChild>
                        <FaEdit color="black"/>
                    </TooltipTrigger>
                    <TooltipContent>Edit</TooltipContent>
                </Tooltip>
            </TooltipProvider>

              </div>
              }
            </button>

            
            <button className="download-btn" onClick={handleDownload}>
            <TooltipProvider>
                <Tooltip>
                    <TooltipTrigger asChild>
                        <FaDownload />
                    </TooltipTrigger>
                    <TooltipContent>Download</TooltipContent>
                </Tooltip>
            </TooltipProvider>
            </button>
          </div>
        )}
      </div>

      <div style={{ paddingBottom: "0px", marginTop: "10px", height: "80%", overflow: "hidden"}}>
        <div id="original" style={{ height: "100%"}}>
          {originalTextUrl ? <iframe id="pdfViewer" src={originalTextUrl} width="100%" height="100%" /> : <p>No file selected</p>}
        </div>

        <div id="mod"  style={{ height:"100%",paddingBottom:50 }}>
          <div ref={editorRef} className="modified-text-container" style={{ height: "100%", overflowY: "scroll", padding: "10px" }}>
          </div>
        </div>
      </div>

      <button
        className="wantHover"
        onClick={reset}
        style={{
          marginTop: 10,
          width: "100%",
          height: "40px",
          borderRadius: "10px",
          background: "#FAD1D1",
          fontSize: "16px",
          color: "red",
        }}
      >
        {console.log("###",originalTextUrl) }
        Close {selectedFile || "File"}
      </button>
    </div>
  );
};

export default ExtractedTextComponent;
