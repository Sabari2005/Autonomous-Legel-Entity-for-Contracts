import React, { useState } from "react";
import { motion } from "framer-motion"; // Import Framer Motion
import SideBar from "./components/SideBar.jsx";
import "./App.css";
import RiskAnalysisLeft from "./components/RiskAnalysisLeft.jsx";
import RiskAnalysisRight from "./components/RiskAnalysisRight.jsx";

function RiskAnalysis() {
    const [extractedExplainedText, setExtractedExplainedText] = useState("");
    const [extractedOriginalText, setExtractedOriginalText] = useState("");
    const [extractedModifiedText, setExtractedModifiedText] = useState("");
    const [fileUrl, setFileUrl] = useState(""); // Store file URL

    return (
        <motion.div
        className="RiskAnalysisSection"
        initial={{ opacity: 0, filter: "blur(10px)" }}
        animate={{ opacity: 1, filter: "blur(0px)" }}
        exit={{ opacity: 0, filter: "blur(10px)" }}
        transition={{ duration: 0.6, ease: "easeOut" }}
        >
            <SideBar />
            <div className="Risk_right">
                <RiskAnalysisLeft 
                    setFileUrl={setFileUrl}
                    setExtractedExplainedText={setExtractedExplainedText}
                    setExtractedOriginalText={setExtractedOriginalText}
                    setExtractedModifiedText={setExtractedModifiedText} 
                />
                <RiskAnalysisRight 
                    extractedExplainedText={extractedExplainedText}
                    extractedOriginalText={extractedOriginalText}
                    extractedModifiedText={extractedModifiedText}
                    fileUrl={fileUrl} // Pass file URL
                />
            </div>
        </motion.div>
    );
}

export default RiskAnalysis;
