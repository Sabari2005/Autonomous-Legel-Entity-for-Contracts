import React, {useState} from "react";
import '../src/components/Style.css';
import Sidebar from "./components/SideBar";
import SummerizerTop from "./components/SummerizerTop";
import SummerizerRight from "./components/SummerizerRight";
import SummerizerLeft from "./components/SummerizerLeft";
import Testing from "./components/Testing";
function Summerrizer(){
    const [serverResponses, setServerResponses] = useState([]); // Store multiple responses
    const [locatedMsgIndex, setLocatedMsgIndex] = useState(null); // ✅ Shared state for locating messages

    const handleReceiveResponse = (newResponse) => {
        setServerResponses((prevResponses) => [...prevResponses, newResponse]); // Append new response
    };
    
    return(
        <div style={{display:"flex",justifyContent:"flex-end",height:"100vh",width:"100%"}}>
        <Sidebar />
            <div className="Summerizer">
                <SummerizerTop />
                <div className="summerizerContain">
                <SummerizerLeft 
                        onReceiveResponse={handleReceiveResponse} 
                        locatedMsgIndex={locatedMsgIndex} 
                        setLocatedMsgIndex={setLocatedMsgIndex} // ✅ Pass the function
                    />
                    <SummerizerRight 
                        serverResponses={serverResponses} 
                        locatedMsgIndex={locatedMsgIndex} 
                        setLocatedMsgIndex={setLocatedMsgIndex} // ✅ Pass the function
                    />
                </div>
            </div>

        </div>
    )
}
export default Summerrizer;