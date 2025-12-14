import React from "react";
import {motion} from 'framer-motion'
import './Style.css'
function Sender({text}){
    return(
        <div className="SenderContain">
            <div className="Senderchatcontent" style={{borderTopRightRadius:0}}>
                <span>{text}</span>
            </div>
            <img src="/user.svg"></img>
        </div>
    )
}

export default Sender