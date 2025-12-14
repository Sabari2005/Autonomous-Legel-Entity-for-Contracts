import React,{useContext} from "react";
import './Style.css';
import { ExtractedTextContext } from "../ExtractedTextContext";
function SummerizerTop(){
    const { fileName } = useContext(ExtractedTextContext);
    return(
        <div className="SummerizerTop">
            <span>
                {fileName ? fileName : "Untitled Summary"}
            </span>
        </div>
    )
}
export default SummerizerTop;