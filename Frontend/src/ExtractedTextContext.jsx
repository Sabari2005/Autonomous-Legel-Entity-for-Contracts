import React, { createContext, useState,useEffect, useRef, useContext } from "react";
import Notification from "./components/Notification";
// ✅ Create and export the Context
import { AuthContext } from "./AuthContext";
export const ExtractedTextContext = createContext(null);


// ✅ Define the Provider component
export const ExtractedTextProvider = ({ children }) => {
  const [extractedOriginalText, setExtractedOriginalText] = useState("");
  const [extractedModifiedText, setExtractedModifiedText] = useState("");
  const [extractedExplainedText, setExtractedExplainedText] = useState("");
  const [isUploaded, setIsUploaded] = useState(false);
  const [originalTextUrl, setOriginalTextUrl] = useState(""); // ✅ Store URL
  const [conversation, setConversation] = useState([]); // Stores [{ sender: "user", text: "..." }, { sender: "server", text: "..." }]
  const [processingFile, setProcessingFile] = useState(false);
  const [contractContent, setContractContent] =useState("");
  const {userDetails} =useContext(AuthContext);
  // State to hold the current server message (response in progress)
  const [serverMessage, setServerMessage] = useState(""); 
  const [messages, setMessages] = useState([]);
    const [loading, setLoading] = useState(false);
    const [contractKey, setContractKey] = useState(0);
    const [fileName, setFileName] = useState("");
    const[KgContent, setKgContent] = useState("");
      const [formData, setFormData] = useState({
        sender_email: "",
        receiver_email: "",
        application_type: "",
        counter_party_name: "",
        counter_party_role: "",
        counter_party_companyname: "",
        send_party_name: "",
        sender_party_role: "",
        sender_party_company: ""
      });
      const [isEditable, setIsEditable] = useState(true);
  const [CpyDelReportPath, setCpyDelReportPath] = useState("");
  const [CpyDelImgPaths, setCpyDelImgPaths] = useState([]);
  const [CaseDetails, setCaseDetails] = useState([]);
  const [FinalReport, setFinalReport]=useState("");
  const [SidebarOperation, setSidebarOperation] = useState(false);
  const [showDetailsPopup, setShowDetailsPopup] = useState(false);
   //   const sendMessage = async (inputText) => {
  //     if (!inputText.trim() || loading) return;

  //     const newMessage = { sender: "user", text: inputText };
  //     setMessages((prevMessages) => [...prevMessages, newMessage]); // Add user message to state
  //     setLoading(true);

  //     try {
  //         const response = await fetch("http://127.0.0.1:8000/send_message", {
  //             method: "POST",
  //             headers: { "Content-Type": "application/json" },
  //             body: JSON.stringify({ message: inputText }),
  //         });

  //         if (!response.ok) {
  //             throw new Error(`Server responded with status ${response.status}`);
  //         }

  //         const data = await response.json();
  //         const botMessage = { sender: "bot", text: data.message };
  //         console.log(data.message);
  //         setMessages((prevMessages) => [...prevMessages, botMessage]); // Add bot response
  //     } catch (error) {
  //         console.error("Error sending message:", error);
  //     }

  //     setLoading(false);
  // };

  // ✅ Function to add a new message (user or server)
  const [sessionId, setSessionId] = useState(null)  ;  // Store session ID
  const [notification, setNotification] = useState({ message: "", color: "", isVisible: false });
  const showNotification = (message, color) => {
    setNotification({ message, color, isVisible: true });

    // Hide after 3 seconds
    setTimeout(() => {
        setNotification({ message: "", color: "", isVisible: false });
    }, 3000);
};
  const loadChatHistory = async (session_id) => {
    try {
        const response = await fetch(`https://8000-01jrcj02963afp5r8dxtpsxdqz.cloudspaces.litng.ai/get_messages/${session_id}`);
        const data = await response.json();

        if (data.messages) {
            setSessionId(session_id);
            setMessages(data.messages);
        }
    } catch (error) {
      showNotification("Error loading chat history", "red");
        console.error("Error loading chat history", error);
    }
};
  const handleNewChat = async () => {
    try {
        const response = await fetch("https://8000-01jrcj02963afp5r8dxtpsxdqz.cloudspaces.litng.ai/create_session/", {
            method: "POST",
            headers: { "Content-Type": "application/json", userid:userDetails.sub },
        });

        const data = await response.json();
        console.log("New Session ID:", data.session_id);
        
        if (data.session_id) {
            setSessionId(data.session_id);  // Store session ID in state
            showNotification("New Session Created", "green");
            setMessages([]);
            return data.session_id;  // ✅ Return the new session ID
        } else {
            throw new Error("Session ID is missing in response");
        }
    } catch (error) {
        console.error("Error creating session:", error);
        showNotification("Error creating session", "red");
        return null;  // ✅ Return null to indicate failure
    }
};

    const sendMessage = async (inputText) => {
      if (!inputText.trim() || loading) return;
      
      let currentSessionId = sessionId;

      if (!currentSessionId) {
        console.error("Session ID is missing. Creating a new chat session...");
    
        currentSessionId = await handleNewChat();  // Wait for new session
    
        if (!currentSessionId) {
          showNotification("Failed to create a new session. Cannot send message.", "red");
            console.error("Failed to create a new session. Cannot send message.");
            return;
        }
    }
    
  
      const newMessage = { sender: "user", text: inputText };
    setMessages((prevMessages) => [...prevMessages, newMessage]);
    if (inputText==="/process"){
      showNotification("Processing File", "blue");
    }
      setLoading(true);
  
      try {
        const requestBody = {
          session_id: currentSessionId,
          message: inputText,
          userid: userDetails.sub,
          FullDetails: formData
        };
        if (contractContent && typeof contractContent === "string") {
          requestBody.contract = contractContent;
        }
          const response = await fetch("https://8000-01jrcj02963afp5r8dxtpsxdqz.cloudspaces.litng.ai/send_message/", {
            method: "POST",
            headers: { 
              "Content-Type": "application/json", 
              "userid": userDetails.sub 
            },
            body: JSON.stringify(requestBody),
          });
  
          const data = await response.json();
          console.log("API Response:", data);
  
          if (data.error) {
            showNotification(`${data.error}`, "yellow");
              throw new Error(data.error);  // Handle backend errors properly
          }
          // if (data.message.includes("Kindly Message /process to process the document")) {
          //   setProcessingFile(true);
          // }
          if (data.file) {
            setProcessingFile(false); // Stop showing "Processing file..." 
            // const fileMessage = { sender: "bot", text: "processing file please wait." };
            try {
              const contractResponse = await fetch("https://8000-01jrcj02963afp5r8dxtpsxdqz.cloudspaces.litng.ai/get_contract/",{
                method:"GET",
                headers:{
                  "content-type":"application/json",
                 
                }
              });
              if (!contractResponse.ok) {
                  throw new Error("Failed to fetch contract");
              }
              const contractHtml = await contractResponse.text(); // ✅ Gets full HTML content
        console.log("Contract HTML:", contractHtml); // Debugging output
        setContractContent(null); // Unmount first
        setTimeout(() => {
          setContractContent(contractHtml);  // Then remount
          setContractKey(prev => prev + 1); // change key to trigger remount
        }, 0);
        // setContractContent(contractHtml); // ✅ Correctly storing full HTML content
            } catch (error) {
              console.error("Error fetching contract:", error);
          }


            // setMessages((prevMessages) => [...prevMessages, fileMessage]);
          }
  
          if (!data.message) {
              throw new Error("Invalid response structure: 'message' field missing");
          }
          if (data.message=="this is file"){
              const botMessage = { sender: "bot", text: "Below is the contract processed." };
          
          setMessages((prevMessages) => [...prevMessages, botMessage]);
          showDetailsPopup(true);
        }
        else{
          const botMessage = { sender: "bot", text: data.message };
          
          setMessages((prevMessages) => [...prevMessages, botMessage]);
        }
      } catch (error) {
          console.error("Error sending message:", error);
      }
  
      setLoading(false);
  };


  const addMessageToConversation = (sender, message) => {
    setConversation((prevConversation) => [
      ...prevConversation, 
      { sender, 
        text: message.text, 
        isFile: message.isFile || false, 
        timestamp: new Date().toISOString() }
    ]);
  };

  const updateServerMessage = (message) => {
    setConversation((prevConversation) => {
        const lastMessage = prevConversation[prevConversation.length - 1];

        if (lastMessage?.sender === "server") {
            return [
                ...prevConversation.slice(0, -1),
                { ...lastMessage, text: lastMessage.text + message } // Append chunks
            ];
        } else {
            return [
                ...prevConversation,
                { sender: "server", text: message, isFile: false, timestamp: new Date().toISOString() }
            ];
        }
    });
};

// 
  return (
    <ExtractedTextContext.Provider
      value={{
        extractedExplainedText,
        setExtractedExplainedText,
        extractedOriginalText,
        setExtractedOriginalText,
        extractedModifiedText,
        setExtractedModifiedText,
        isUploaded,
        setIsUploaded,
        originalTextUrl, // ✅ Provide URL state
        setOriginalTextUrl,
        conversation,
        addMessageToConversation,
        serverMessage,
        setFileName,fileName,
        updateServerMessage,contractContent, // Expose the update function for SSE
        messages, sendMessage, loading,handleNewChat, loadChatHistory,processingFile,
        KgContent, setKgContent,
        CpyDelImgPaths, setCpyDelImgPaths,
        CpyDelReportPath, setCpyDelReportPath,
        CaseDetails, setCaseDetails,FinalReport, setFinalReport,
        SidebarOperation,setSidebarOperation,setContractContent,showDetailsPopup,setShowDetailsPopup,
        formData,setFormData,isEditable,setIsEditable,contractKey,setContractKey,
      }}
    >
      {children}
    </ExtractedTextContext.Provider>
  );
};
