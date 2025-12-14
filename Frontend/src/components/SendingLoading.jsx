import React from "react";
import "./Loading.css";

function SendingLoading() {
    return (
        <div className="ReceiverContain">
            <img src="/logo.svg" alt="Chatbot Logo" />
            <div className="Receiverchatcontent" style={{ borderTopLeftRadius: 0 }}>
                <div className="loading-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        </div>
    );
}

export default SendingLoading;
