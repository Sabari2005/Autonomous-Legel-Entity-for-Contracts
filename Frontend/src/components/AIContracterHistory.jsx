import React, { useState } from "react";
import { ChevronLeft, History } from "lucide-react"; // Icons
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"; // Tooltip

function AIContracterHistory({ setMessages }) {
    const [isOpen, setIsOpen] = useState(false); // Start closed

    return (
        <div >
            {/* Button to toggle history panel */}
            <TooltipProvider>
                <Tooltip>
                    <TooltipTrigger asChild>
                        <button 
                            className="open-btn" 
                            onClick={() => setIsOpen(!isOpen)}
                            style={{
                                position: "absolute", 
                                right: isOpen ? "200px" : "0", 
                                top: "10px",
                                transition: "all 0.3s"
                            }}
                        >   
                            <History size={24} />
                        </button>
                    </TooltipTrigger>
                    <TooltipContent>Toggle History</TooltipContent>
                </Tooltip>
            </TooltipProvider>

            {/* History Panel */}
            {isOpen && (
                <div className="historyContent">
                    <button 
                        className="close-btn" 
                        onClick={() => setIsOpen(false)}
                        style={{
                            position: "absolute", 
                            left: "-30px", 
                            top: "10px",
                            background: "white", 
                            border: "none",
                            cursor: "pointer"
                        }}
                    >
                        <ChevronLeft size={24} />
                    </button>
                    <p style={{ padding: "20px" }}>History Content Goes Here</p>
                </div>
            )}
        </div>
    );
}

export default AIContracterHistory;
