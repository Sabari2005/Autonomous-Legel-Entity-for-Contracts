import React, { useState,useContext } from "react";
import SideBar from "./components/SideBar";
import "./components/Style.css";
import AIContractIntro from "./components/AIContractIntro";
import AIContractLeft from "./components/AIContractLeft";
import AIContractRight from "./components/AIContractRight";
import SummerizerTop from "./components/SummerizerTop";
import AIContracterHistory from "./components/AIContracterHistory";
import AIContracterTop from "./components/AIContracterNav";
import { ExtractedTextContext } from "./ExtractedTextContext";
// import AIContracterTop from "./components/AIContraxterTop";
function AIContract() {
    const { messages, sendMessage, loading } = useContext(ExtractedTextContext);
    return (
        <div className="AIContract">
            <SideBar /> 
            <AIContracterTop />
            <div className="aicontractcontain">
                {messages.length === 0 ? (
                    <>
                    <AIContractIntro sendMessage={sendMessage} /></>
                ) : (
                    <>        
                    
                    <AIContractLeft messages={messages} /></>
                )}
                <AIContractRight sendMessage={sendMessage} loading={loading} />
                
            </div>

                
        </div>
    );
}

export default AIContract;
