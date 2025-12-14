import React from "react";
import "./AIContracterChatLoader.css";

const AIContracterChatLoader = () => {
    return (
        <>
        <img src="public/logo.svg" />
            <div className="typing-container">
                <div className="typing-bar" />
                <div className="typing-bar short" />
            </div>
        </>
    );
};

export default AIContracterChatLoader;
