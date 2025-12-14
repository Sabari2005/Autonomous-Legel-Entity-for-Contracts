import React, { useState,useContext } from "react";
import "./Style.css";
import Notification from "./Notification";
import { PlusCircleOutlined, SendOutlined, CloudUploadOutlined } from "@ant-design/icons";
import ChatProcessLoader from "./ChatProcessLoader";
import { ExtractedTextContext } from "../ExtractedTextContext";
function AIContractRight() {
    // const [inputText, setInputText] = useState("");
    const [notification, setNotification] = useState(null);
    const { sendMessage, loading } = useContext(ExtractedTextContext);
    const [inputText, setInputText] = useState("");
    const handleSendMessage = () => {
        if (inputText.trim()) {
            sendMessage(inputText);
            setInputText("");
        }
    };
    const handleKeyDown = (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault(); // prevent newline
            if (!loading) handleSendMessage();
        }
    };


    return (
        <>
            {notification && (
                <Notification
                    message={notification.message}
                    color={notification.color}
                    onClose={() => setNotification(null)}
                />
            )}
            <div className="AIContractInputContain">
            <textarea
    ref={(el) => {
        if (el) {
            el.style.height = "auto"; // reset
            el.style.height = `${Math.min(el.scrollHeight, 200)}px`; // max height: 200px
        }
    }}
    value={inputText}
    onChange={(e) => {
        setInputText(e.target.value);
        const el = e.target;
        el.style.height = "auto";
        el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
    }}
    onKeyDown={handleKeyDown}
    disabled={loading}
    placeholder="Type your message..."
    className="growing-textarea"
/>

                <button
                    className="ContractInputSend"
                    onClick={handleSendMessage}
                    disabled={loading}
                >
                    {loading ? <ChatProcessLoader height="30px" width="30px" /> : <SendOutlined style={{ fontSize: 24, color: "white" }} />}
                </button>
            </div>
        </>
    );
}

export default AIContractRight;
