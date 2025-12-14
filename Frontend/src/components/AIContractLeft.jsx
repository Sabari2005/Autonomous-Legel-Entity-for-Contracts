import React, { useContext, useState, useRef, useEffect } from "react";
import { motion } from 'framer-motion';
import "./Style.css";
import Sender from "./Sender";
import Receiver from "./Receiver";
import Notification from "./Notification";
import { ExtractedTextContext } from "../ExtractedTextContext";
import AIContracterChatLoader from "./AIContracterChatLoader";
import Quill from "quill";
import "quill/dist/quill.snow.css";
import { AuthContext } from "../AuthContext";
import { FaEdit } from 'react-icons/fa'
import { FaDownload } from "react-icons/fa";
import { FaSave } from 'react-icons/fa';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"; // Tooltip
import UserDetails from "./UserDetails";


function AIContractLeft() {
    const { messages, processingFile, contractContent, setContractContent, loading, showDetailsPopup, setShowDetailsPopup } = useContext(ExtractedTextContext);
    const [expanded, setExpanded] = useState(false);
    const chatContainerRef = useRef(null);
    const lastMessageRef = useRef(null);
    const quillRef = useRef(null);
    const quillInstance = useRef(null);
    const [isEditable, setIsEditable] = useState(false);
    const {userDetails}=useContext(AuthContext);
    const [downloading, setDownloading] = useState(false);
    const [quillKey, setQuillKey] = useState(0);
    const resetQuillEditor = (htmlContent) => {
        setContractContent(null); // Optional: clear current content
        if (quillInstance.current) {
            quillInstance.current = null;
        }
        setTimeout(() => {
            setContractContent(htmlContent);
            setQuillKey(prev => prev + 1); // Force remount
        }, 0);
    };
    



// const quillRef = useRef(null);

    useEffect(() => {
        if (lastMessageRef.current) {
            lastMessageRef.current.scrollIntoView({ behavior: "smooth" });
        }
    }, [messages]);

    useEffect(() => {
        if (quillRef.current && !quillInstance.current) {
            quillInstance.current = new Quill(quillRef.current, {
                theme: "snow",
                placeholder: "",
                modules: {
                    toolbar: [
                        [{ header: "1" }, { header: "2" }, { font: [] }],
                        [{ list: "ordered" }, { list: "bullet" }],
                        ["bold", "italic", "underline"],
                        ["link"],
                        ["blockquote", "code-block"],
                        [{ align: [] }],
                    ],
                },
            });

            if (contractContent) {
                quillInstance.current.root.innerHTML = contractContent;
            }

            quillInstance.current.on("text-change", () => {
                const html = quillInstance.current.root.innerHTML;
                setContractContent(html);
            });
        }
    }, [contractContent,setContractContent]);
    useEffect(() => {
        if (quillInstance.current) {
            quillInstance.current.enable(isEditable);
        }
    }, [isEditable]);

    useEffect(() => {
        if (quillRef.current) {
            quillInstance.current = new Quill(quillRef.current, {
                theme: "snow",
                placeholder: "",
                modules: {
                    toolbar: [
                        [{ header: "1" }, { header: "2" }, { font: [] }],
                        [{ list: "ordered" }, { list: "bullet" }],
                        ["bold", "italic", "underline"],
                        ["link"],
                        ["blockquote", "code-block"],
                        [{ align: [] }],
                    ],
                },
            });
    
            if (contractContent) {
                quillInstance.current.root.innerHTML = contractContent;
            }
    
            quillInstance.current.on("text-change", () => {
                const html = quillInstance.current.root.innerHTML;
                setContractContent(html);
            });
        }
    }, [quillKey]);
    

    const toggleExpand = () => {
        setExpanded(!expanded);
    };

    const handleDownloadPDF = async () => {
        try {
            setDownloading(true); // Start loading
            const response = await fetch("https://8000-01jrcj02963afp5r8dxtpsxdqz.cloudspaces.litng.ai/convert", {
                method: "POST",
                headers: { 
                    "Content-Type": "application/json",
                    userid: userDetails.sub
                },
                body: JSON.stringify({ html_content: contractContent })
            });
    
            if (!response.ok) {
                throw new Error("Failed to generate PDF");
            }
    
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "contract.pdf";
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        } catch (error) {
            console.error("Error downloading PDF:", error);
        } finally {
            setDownloading(false); // Stop loading
        }
    };
    

    return (
        <div className="AIContractLeft">
            {showDetailsPopup && (
              <div className="user-details-popup-overlay">
                <div className="user-details-popup">
                  <button className="close-button" onClick={() => setShowDetailsPopup(false)}>×</button>
                  <UserDetails />
                </div>
              </div>
            )}

            <div className="Chatting_contain" ref={chatContainerRef}>
                {messages.map((msg, index) => (
                    <motion.div
                        key={index}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.3 }}
                    >
                        {msg.sender === "user" ? (
                            <Sender text={msg.text} />
                        ) : (
                            <Receiver text={msg.text} />
                        )}
                    </motion.div>
                ))}

                {loading && (
                    <motion.div className="typing-indicator" transition={{ duration: 0.5, repeat: Infinity, ease: "easeInOut" }}>
                        <AIContracterChatLoader />
                    </motion.div>
                )}

                <div ref={lastMessageRef}></div>

                {!processingFile && contractContent && (
                    <div style={{ display: "flex", justifyContent: "center", width: "100%" }}>
<div className={`processing-box ${expanded ? "expanded" : ""}`} onClick={!expanded ? toggleExpand : null}>
    {expanded && (
        <div style={{ display: "flex", flexDirection: "row", alignItems: "center", justifyContent: "space-between" }}>
            <h2>Contract Document</h2>
            <div style={{ display: "flex", flexDirection: "row", gap: 10 }}>
                <button className="processing-closing-btn"
                    onClick={(e) => {
                        e.stopPropagation();
                        setExpanded(false);
                    }}>
                    close
                </button>
                <button className="processing-closing-btn"
                    style={{
                        backgroundColor:"lightcyan",
                        color:"black"
                    }}
                    onClick={(e) => {
                        e.stopPropagation();
                        setIsEditable((prev) => !prev); // Toggle edit mode
                    }}>
                    {isEditable ? <FaSave /> : <FaEdit />}
                </button>
                <button
    className="processing-closing-btn"
    style={{ backgroundColor: "grey", color: "white" }}
    onClick={(e) => {
        e.stopPropagation();
        handleDownloadPDF();
    }}
    disabled={downloading}
>
    {downloading ? "Downloading..." : "Download"}
</button>

            </div>
        </div>
    )}

    <div style={{ height: "90%", marginTop: 10 }}>
    <div key={quillKey} ref={quillRef} style={{ backgroundColor: "white", height: "96%", marginBottom: "1rem" }} />

    </div>
</div>
                    </div>
                )}
            </div>
        </div>
    );
}

export default AIContractLeft;
