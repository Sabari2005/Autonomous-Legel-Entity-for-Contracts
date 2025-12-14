import React, {useContext,useEffect,useState,useRef} from "react";
import { ExtractedTextContext } from "../ExtractedTextContext";
import SummeryMsgContain from "./SummeryMsgComtain";

function SummerizerRight({ serverResponses, locatedMsgIndex, setLocatedMsgIndex }) {
    const rightChatRef = useRef(null);

    // const [locatedMsgIndex, setLocatedMsgIndex] = useState(null);
    // const { conversation } = useContext(ExtractedTextContext); // Use context
    const [streamingResponse, setStreamingResponse] = useState("");
    const { serverMessage, conversation } = useContext(ExtractedTextContext);
    // useEffect(() => {
        
    //     if (rightChatRef.current) {
    //         rightChatRef.current.scrollTop = rightChatRef.current.scrollHeight;
    //     }
    // }, [conversation]); // Runs whenever conversation updates

    return (
        <div className="summerizerRight">
            <div style={{ height: 50, width: "100%", borderBottom: "1px solid rgb(200, 200, 200)", padding:15 }}>
                <h1 style={{marginLeft:"0px"}}><b>Summary</b></h1>
            </div>
            {/* Display multiple server responses */}
            <div className="SummerizerChatRightContain" ref={rightChatRef}>
            {conversation
                .filter((msg) => msg.sender === "server")
                .map((msg, index) => (
                    <SummeryMsgContain 
                        key={index} 
                        msg={msg.text} 
                        index={index} 
                        isFile={msg.isFile}  
                        timestamp={msg.timestamp}
                        setLocatedMsgIndex={setLocatedMsgIndex}
                        locatedMsgIndex={locatedMsgIndex} 
                        isServer={msg.sender === "server"}
                    />
                ))}

            {/* Display the server message dynamically */}
            {/* {conversation.map((msg, index) => (
    <SummeryMsgContain 
        key={index}  // Use dynamic key to prevent overwriting
        msg={msg.text} 
        isFile={msg.isFile} 
        timestamp={msg.timestamp}
        setLocatedMsgIndex={setLocatedMsgIndex}
        locatedMsgIndex={locatedMsgIndex} 
    />
))} */}

            </div>
        </div>
    );
}

export default SummerizerRight;
