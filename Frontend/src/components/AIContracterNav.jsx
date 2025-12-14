import React, { useContext, useState, useEffect } from "react";
import "./Style.css";
import { ChevronRight, History, Plus } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { motion } from "framer-motion";
import { ExtractedTextContext } from "../ExtractedTextContext";
import { AuthContext } from "../AuthContext";
import { Info } from 'lucide-react';
import UserDetails from "./UserDetails";
function AIContracterTop() {
    const [isOpen, setIsOpen] = useState(false);
    const [sessions, setSessions] = useState([]);
    const { handleNewChat, loadChatHistory,setShowDetailsPopup } = useContext(ExtractedTextContext);
    const {userDetails} =useContext(AuthContext);
        
    // Fetch all chat sessions
    const fetchSessions = async () => {
        try {
            const response = await fetch("https://8000-01jrcj02963afp5r8dxtpsxdqz.cloudspaces.litng.ai/get_sessions/",{
                method:"GET",
                headers:{
                    "content-type":"application/JSON",
                    userid:userDetails.sub
                }
            });
            const data = await response.json();
            setSessions(data);
        } catch (error) {
            console.error("Error fetching sessions:", error);
        }
    };

    useEffect(() => {
        if (isOpen) {
            fetchSessions();  // Load sessions when history panel opens
        }
    }, [isOpen]);

    return (
        <div className="ContracterTop" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px 20px" }}>
            {/* <span style={{ background: "#F2F9FF", padding: "5px", borderRadius: "10px", marginTop: 10, fontSize: "1.2rem", fontWeight: 700 }}>
                AI Contracter
            </span> */}
            <span></span>
            {/* <UserDetails /> */}
            

            <div style={{ display: "flex", gap: "10px" }}>
                <TooltipProvider>
                    <Tooltip>
                        <TooltipTrigger asChild>
                            <motion.button 
                                className="open-btn" 
                                onClick={() => setShowDetailsPopup(true)}
                                style={{
                                    right:170
                                }}
                            >
                                <Info size={24} />
                            </motion.button>
                        </TooltipTrigger>
                        <TooltipContent>User Information</TooltipContent>
                    </Tooltip>
                </TooltipProvider>
                <TooltipProvider>
                    <Tooltip>
                        <TooltipTrigger asChild>
                            <motion.button 
                                className="open-btn" 
                                onClick={() => setIsOpen(!isOpen)} 
                                style={{
                                    right:130
                                }}
                            >
                                <History size={24} />
                            </motion.button>
                        </TooltipTrigger>
                        <TooltipContent>History</TooltipContent>
                    </Tooltip>
                </TooltipProvider>

                {/* <TooltipProvider>
                    <Tooltip>
                        <TooltipTrigger asChild>
                            <motion.button 
                                className="open-btn" 
                                onClick={handleNewChat}
                                whileHover={{ scale: 1.1 }}
                                whileTap={{ scale: 0.9 }}
                            >
                                <Plus size={24} />
                            </motion.button>
                        </TooltipTrigger>
                        <TooltipContent>New Chat</TooltipContent>
                    </Tooltip>
                </TooltipProvider> */}
                <button
                    style={{
                        backgroundColor:"black",
                        padding: "5px 15px",
                        color:"white",
                        borderRadius: "10px",
                        cursor:"pointer"
                    }}
                    onClick={handleNewChat}
                >New Chat</button>
            </div>

            {/* History Panel */}
            <motion.div
                className="historyContent"
                initial={{ x: 300 }}
                animate={{ x: isOpen ? 0 : 300 }}
                transition={{ duration: 0.5, ease: "easeInOut" }}
                style={{
                    position: "absolute",
                    right: "0",
                    top: "0",
                    backgroundColor: "white",
                    height: "100vh",
                    width: "300px",
                    boxShadow: "-2px 0 10px rgba(0,0,0,0.1)",
                    zIndex: 9000
                }}
            >
                <button className="close-btn" onClick={() => setIsOpen(false)} style={{ position: "absolute", left: 0 }}>
                    <ChevronRight size={24} />
                </button>

                <span style={{ position: "absolute", right: 50, top: 15 }}>History</span>

                <div className="history-content">
                {sessions.length === 0 ? <p>No previous sessions</p> : (
    sessions.map((session) => (
        <div 
            key={session.session_id} 
            className="history_name_container" 
            onClick={() => loadChatHistory(session.session_id)}
            style={{ cursor: "pointer", padding: "10px", borderBottom: "1px solid #ccc", color: "black" }}
        >
            <span>{session.first_message}</span>
        </div>
    ))
)}

</div>

            </motion.div>
        </div>
    );
}

export default AIContracterTop;
