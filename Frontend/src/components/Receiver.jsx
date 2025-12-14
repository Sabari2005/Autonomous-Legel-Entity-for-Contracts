// import React from "react";
// import './Style.css';

// function Receiver({ text }) {  // Receive the 'text' prop
//     return (
//         <div className="ReceiverContain">
//             <img src="/logo.svg" alt="Chatbot Logo" />
//             <div className="Receiverchatcontent" style={{ borderTopLeftRadius: 0 }}>
//                 <span   >{text}</span> 
//             </div>
//         </div>
//     );
// }

// export default Receiver;

import React from "react";
import { marked } from "marked";
import './Style.css';

function Receiver({ text }) {
    const renderMarkdown = (markdownText) => {
        return { __html: marked(markdownText) };
    };

    return (
        <div className="ReceiverContain">
            <img src="/logo.svg" alt="Chatbot Logo" />
            <div className="Receiverchatcontent" style={{ borderTopLeftRadius: 0 }}>
                <span dangerouslySetInnerHTML={renderMarkdown(text)} />
            </div>
        </div>
    );
}

export default Receiver;

