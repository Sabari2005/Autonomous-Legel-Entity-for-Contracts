import React, { useState, useEffect, useContext, useRef } from "react";
import axios from "axios";
import { PlusCircleOutlined,ArrowRightOutlined,PaperClipOutlined ,CaretRightOutlined, SendOutlined, CloudUploadOutlined } from "@ant-design/icons";
import "./Style.css";
import SummeryMsgContain from "./SummeryMsgComtain";
import { ExtractedTextContext } from "../ExtractedTextContext";
import ChatProcessLoader from "./ChatProcessLoader";
import Notification from "./Notification";
import TextareaAutosize from "react-textarea-autosize";
import { yellow } from "@mui/material/colors";
import { AuthContext } from "../AuthContext";
function SummerizerLeft({ onReceiveResponse, locatedMsgIndex, setLocatedMsgIndex }) {
    const leftChatRef = useRef(null);
    const [notification, setNotification] = useState(null);

    const [inputText, setInputText] = useState("");
    const [messages, setMessages] = useState([]); 
    const [file, setFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const { userDetails } = useContext(AuthContext);
    const { conversation, addMessageToConversation, updateServerMessage, setFileName, KgContent, setKgContent, setSidebarOperation } = useContext(ExtractedTextContext);
    const fileInputRef = useRef(null); // Ref for file input
    const link="https://8000-01jrcj02963afp5r8dxtpsxdqz.cloudspaces.litng.ai";
    const eventSourceRef = useRef(null); // Store SSE reference
    const textAreaRef = useRef(null);

    const [value, setValue] = useState("");

    const handleKeyDown = (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          sendMessage();
        }
      };




    // ✅ Function to Send Text Message
    // const sendMessage = async () => {
    //     if (loading) {
    //         setNotification({ message: "Processing another request, please wait...", color: "red" });
    //         return;
    //     }
        
    //     if (!inputText.trim()) return;
        
    //     addMessageToConversation("user", {
    //         text: inputText,
    //         isFile: false,
    //         timestamp: new Date().toISOString()
    //     });

    //     setLoading(true);
    //     setInputText(""); 

    //     try {
    //         const response = await axios.post(
    //             "http://127.0.0.2:8000/summerizer_text", 
    //             { text: inputText },     
    //             { headers: { "Content-Type": "application/json" } }
    //         );

    //         console.log("Server Response:", response.data.response);

    //         addMessageToConversation("server", {
    //             text: response.data.response,
    //             isFile: false,
    //             timestamp: new Date().toISOString()
    //         });
    //     } catch (error) {
    //         console.error("Error sending message:", error);
    //         addMessageToConversation("server", {
    //             text: "Cannot process...",
    //             isFile: false,
    //             timestamp: new Date().toISOString()
    //         });
    //         setNotification({ message: "Error sending message", color: "red" });
    //     } finally {
    //         setLoading(false);
    //     }
    // };

    // const sendMessage = async () => {
    //     if (loading) {
    //         setNotification({ message: "Processing another request, please wait...", color: "red" });
    //         return;
    //     }
    
    //     if (!inputText.trim()) return;
    
    //     addMessageToConversation("user", {
    //         text: inputText,
    //         isFile: false,
    //         timestamp: new Date().toISOString()
    //     });
    
    //     setLoading(true);
    //     setInputText("");
    
    //     try {
    //         const eventSource = new EventSource(`https://1ae8e974156258.lhr.life/summarizer_text_sse?text=${encodeURIComponent(inputText)}`);
    
    //         eventSource.onmessage = (event) => {
    //             console.log("SSE Response:", event.data);
                
    //             // Dynamically update the serverMessage as the message is being received
    //             if (event.data.includes("#3jk*24")) {
    //                 console.log("🔴 SSE Connection Closed by Server");
    //                 eventSource.close();
    //                 setLoading(false);
    //             }
    //             updateServerMessage(event.data);
    
    //             // Once the full message is received, add it to the conversation
    //             // addMessageToConversation("server", {
    //             //     text: event.data,
    //             //     isFile: false,
    //             //     timestamp: new Date().toISOString()
    //             // });

    //         };
    
    //         eventSource.onerror = (error) => {
    //             console.error("SSE error:", error);
    //             setNotification({ message: "SSE connection error", color: "red" });
    //             eventSource.close();  // Close the connection if an error occurs
    //             setLoading(false);
    //         };
    
    //         eventSource.onopen = () => {
    //             console.log("SSE Connection Opened");
    //         };
    
    //     } catch (error) {
    //         console.error("Error sending message:", error);
    //         setNotification({ message: "Failed to connect to SSE", color: "red" });
    //         setLoading(false);
    //     }
    // };
    
    const sendMessage = async () => {
        if (loading) {
            setNotification({ message: "Processing another request, please wait...", color: "yellow" });
            return;
        }
    
        if (!inputText.trim()) return;
    
        addMessageToConversation("user", {
            text: inputText,
            isFile: false,
            timestamp: new Date().toISOString()
        });
    
        setLoading(true);
        setSidebarOperation(true);
        setInputText(" ");
        if (textAreaRef.current) {
            textAreaRef.current.style.height = "auto"; // reset height manually
        }
    
        try {
            eventSourceRef.current = new EventSource(`${link}/summarizer_text_sse?text=${encodeURIComponent(inputText)}`);
    
            eventSourceRef.current.onmessage = (event) => {
                console.log("SSE Response:", event.data);
                
                // Dynamically update the serverMessage as the message is being received
                if (event.data.includes("#3jk*24")) {
                    console.log("🔴 SSE Connection Closed by Server");
                    eventSourceRef.current.close();
                    setLoading(false);
                    setSidebarOperation(false);
                }else{
                updateServerMessage(event.data);
            }
                // Once the full message is received, add it to the conversation
                // addMessageToConversation("server", {
                //     text: event.data,
                //     isFile: false,
                //     timestamp: new Date().toISOString()
                // });

            };
    
            eventSourceRef.current.onerror = (error) => {
                console.error("SSE error:", error);
                setNotification({ message: "SSE connection error", color: "red" });
                eventSourceRef.current.close();  // Close the connection if an error occurs
                setLoading(false);
                setSidebarOperation(false);
            };
    
            eventSourceRef.current.onopen = () => {
                console.log("SSE Connection Opened");
            };
    
        } catch (error) {
            console.error("Error sending message:", error);
            setNotification({ message: "Failed to connect to SSE", color: "red" });
            setLoading(false);
            setSidebarOperation(false);
        }
    };
    const cancelProcess = () => {
        if (eventSourceRef.current) {
            eventSourceRef.current.close();
            console.log("Process Cancelled");
            setNotification({ message: "Process Cancelled", color: "red" });
        }
        setLoading(false);
        setSidebarOperation(false);
    };
    const handleInput = () => {
        const textarea = textAreaRef.current;
        textarea.style.height = "auto"; // Reset height
        textarea.style.height = Math.min(textarea.scrollHeight, 150) + "px"; // Max height limit (150px)
      };
    
    // ✅ Function to Handle File Upload
    const handleFileUpload = async (event) => {
        if (loading) {
            setNotification({ message: "Processing another request, please wait...", color: "red" });
            return;
        }

        const uploadedFile = event.target.files[0];
        // setNotification({ message: "Processing file", color: "yellow" });
        if (uploadedFile) {
            setFileName(uploadedFile.name); // ⬅️ Store the filename
            processFileUpload(uploadedFile);
        }
    };
    
    const handleFileDrop = (event) => {
        event.preventDefault();
        if (loading) {
            setNotification({ message: "Processing another request, please wait...", color: "red" });
            return;
        }

        const droppedFile = event.dataTransfer.files[0];
        if (droppedFile) processFileUpload(droppedFile);
    };
        
        const processFileUpload = async (file) => {
            if (!file) return;
        
            const allowedExtensions = ["pdf", "docx","png","jpg","jpeg"];
            const fileExtension = file.name.split(".").pop().toLowerCase();
        
            if (!allowedExtensions.includes(fileExtension)) {
                setNotification({ message: "Unsupported file format!", color: "red" });
                return;
            }
        
            addMessageToConversation("user", {
                text: file.name,
                isFile: true,
                timestamp: new Date().toISOString(),
            });
        
            setLoading(true);
            setSidebarOperation(true);
            setFile(file);
            
            const formData = new FormData();
            formData.append("file", file);
        
            try {
                const response = await axios.post(
                    `${link}/summerizer_upload_file`, 
                    formData, 
                    { headers: { "Content-Type": "multipart/form-data",userid:userDetails.sub } }
                );
        
                console.log("File Response:", response.data);
        
                const fileId = response.data.file_id;  // Assuming backend returns a file_id
        
                
        
                // ✅ SSE Connection for File Processing
                eventSourceRef.current = new EventSource(`${link}/summarizer_file_sse?file_id=${fileId}`);
        
                eventSourceRef.current.onmessage = (event) => {
                    console.log("SSE File Response:", event.data);
                    
                    // Dynamically update server message as it's received
                    if (event.data.includes("#3jk*24")) {
                        console.log("SSE Connection Closed by Server");
                        eventSourceRef.current.close();
                        setLoading(false);
                        setSidebarOperation(false);
                        return;
                    }else{
                        updateServerMessage(event.data);
                    }
        
                    
                };
        
                eventSourceRef.current.onerror = (error) => {
                    console.error("SSE error:", error);
                    setNotification({ message: "SSE connection error", color: "red" });
                    eventSourceRef.current.close(); 
                    setLoading(false);
                    setSidebarOperation(false);
                };
        
                eventSourceRef.current.onopen = () => {
                    console.log("SSE Connection Opened for File Processing");
                };
        
            } catch (error) {
                console.error("Error uploading file:", error);
                addMessageToConversation("server", {
                    text: "Cannot process...",
                    isFile: true,
                    timestamp: new Date().toISOString(),
                });
                setNotification({ message: "Error uploading file", color: "red" });
                setLoading(false);
                setSidebarOperation(false);
            } finally {
                setFile(null);
            }
        };
    

    return (
        <div className="summerizerLeft">
            {notification && <Notification message={notification.message} color={notification.color} onClose={() => setNotification(null)} />}
            <div style={{ height: 50, width: "100%", borderBottom: "1px solid rgb(200, 200, 200)", padding:15 }}>
                <h1 style={{marginLeft:"0px"}}><b>Source</b></h1>
            </div>
<div style={{
    width:"100%",
    height:"93%",
    display:"flex",
    justifyContent:"space-between",
    alignItems:"center",
    flexDirection:"column"
}}>
                <div className="SummerizerChatLeftContain" ref={leftChatRef}>
                    {conversation
                        .filter((msg) => msg.sender === "user")
                        .map((msg, index) => (
                            <SummeryMsgContain 
                                key={index} 
                                msg={msg.text} 
                                index={index} 
                                locatedMsgIndex={locatedMsgIndex} 
                                setLocatedMsgIndex={setLocatedMsgIndex}
                                isFile={msg.isFile}  
                                timestamp={msg.timestamp}
                            />
                        ))}
                </div>

                <div className="SummerizerInputContain">
                    <PaperClipOutlined   
                    fontSize={20}
                        className="cludUpload"
                        style={{
                        display: "flex", justifyContent: "center", alignItems: "center",
                        padding: 0, borderRadius: "100%", width: 30, height: 30,marginBottom:10
                    }}
                        onClick={() => fileInputRef.current.click()} // Directly triggers the file input
                    />
    {/* 
                    <input
                        type="text"
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && sendMessage()}
                    /> */}
                        <textarea
            ref={textAreaRef}
            value={inputText}
            onChange={(e) => {
            setInputText(e.target.value);
            handleInput();
            }}
            onKeyDown={handleKeyDown}
            className="custom-textarea"
            placeholder="Type your message..."
        />

    <button 
                    onClick={loading ? cancelProcess : sendMessage}
                    className="sendInputHover" 
                    style={{
                        display: "flex", justifyContent: "center", alignItems: "center",
                        padding: 5, borderRadius: "100%", background: "black", width: 30, height: 30,marginBottom:8
                    }}
                >
                    {loading ? <ChatProcessLoader width="20px" height="20px" /> : <ArrowRightOutlined style={{ fontSize: 15, color: "white" }} />}
                </button>
                </div>
                </div>

            <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileUpload} 
                style={{ display: "none" }} // Hidden input to trigger file selection on button click
            />
        </div>
    );
}

export default SummerizerLeft;
