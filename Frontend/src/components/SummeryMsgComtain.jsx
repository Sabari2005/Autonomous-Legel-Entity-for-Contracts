import React, { useEffect, useRef, useState } from "react";
import MarkdownRenderer from "./Markdown";

function SummeryMsgContain({ msg, index, locatedMsgIndex, setLocatedMsgIndex, isFile, timestamp, isServer }) {
    const messageRef = useRef(null);
    const [isHovered, setIsHovered] = useState(false);
    const [copied, setCopied] = useState(false);

    useEffect(() => {
        if (index === locatedMsgIndex && messageRef.current) {
            messageRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
            messageRef.current.classList.add("glow-effect");
            setTimeout(() => {
                messageRef.current.classList.remove("glow-effect");
            }, 2000);
        }
    }, [locatedMsgIndex]);

    const formattedTime = new Date(timestamp).toLocaleTimeString("en-US", {
        hour: "2-digit",
        minute: "2-digit",
        hour12: true
    });

    const handleCopyToFile = () => {
        const blob = new Blob([msg], { type: "text/plain;charset=utf-8" });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `summary_${index + 1}.txt`;
        a.click();
        URL.revokeObjectURL(a.href);
    };



    const handleCopyToClipboard = async () => {
        try {
            await navigator.clipboard.writeText(msg);
            setCopied(true); // ✅ Show copied message
            setTimeout(() => setCopied(false), 2000); // ✅ Hide after 2 seconds
        } catch (err) {
            console.error("Failed to copy!", err);
        }
    };
    return (
        <div
            className="message-container"
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => setIsHovered(false)}
        >
            <div
                // ref={messageRef}
                // id={`msg-${index}`}
                // className="message-box"
                style={{ backgroundColor: "transparent"}}
            >
                {isServer ? (
                    <div className="message-box"
                    ref={messageRef}
                    id={`msg-${index}`}
                    style={{ border:"1px solid transparent", width:"100%", whiteSpace:"pre-wrap", wordWrap:"break-word"}}
                    >         
                        <MarkdownRenderer markdownText={msg} />
                        <div className="message-footer">
                            <span><b>{formattedTime}</b></span>

                            {isHovered && (
                                <div>
                                    <span
                                        className="locate-btn"
                                        style={{ marginLeft: "10px", cursor: "pointer" }}
                                        onClick={() => {
                                            setLocatedMsgIndex(null);
                                            setTimeout(() => setLocatedMsgIndex(index), 0);
                                        }}
                                    >
                                        Locate
                                    </span>
                                    <span
                                        className="locate-btn"
                                        style={{ marginLeft: "10px", cursor: "pointer" }}
                                        onClick={handleCopyToClipboard}
                                    >
                                        {copied ? "Copied!" : "Copy"}
                                    </span>
                                </div>
                            )}
                        </div>
                    </div>
                   
                ) : (
                    <div style={{padding:10,borderBottom:"0.5px dashed grey",whiteSpace:"pre-wrap", wordWrap:"break-word"}}
                    ref={messageRef}
                    id={`msg-${index}`}>
                        <span>{msg}</span>
                        <div className="message-footer">
                            <span><b>{formattedTime}</b></span>

                            {isHovered && (
                                <div>
                                    <span
                                        className="locate-btn"
                                        style={{ marginLeft: "10px", cursor: "pointer" }}
                                        onClick={() => {
                                            setLocatedMsgIndex(null);
                                            setTimeout(() => setLocatedMsgIndex(index), 0);
                                        }}
                                    >
                                        Locate
                                    </span>
                                    <span
                                        className="locate-btn"
                                        style={{ marginLeft: "10px", cursor: "pointer" }}
                                        onClick={handleCopyToClipboard}
                                    >
                                        {copied ? "Copied!" : "Copy"}
                                    </span>
                                </div>
                            )}
                        </div>
                    </div>

                )}

                {/* <div className="message-footer">
                    <span>{formattedTime}</span>

                    {isHovered && (
                        <div>
                            <span
                                className="locate-btn"
                                style={{ marginLeft: "10px", cursor: "pointer" }}
                                onClick={() => {
                                    setLocatedMsgIndex(null);
                                    setTimeout(() => setLocatedMsgIndex(index), 0);
                                }}
                            >
                                Locate
                            </span>
                            <span
                                className="locate-btn"
                                style={{ marginLeft: "10px", cursor: "pointer" }}
                                onClick={handleCopyToClipboard}
                            >
                                {copied ? "Copied!" : "Copy"}
                            </span>
                        </div>
                    )}
                </div> */}
            </div>
        </div>
    );
}

export default SummeryMsgContain;
