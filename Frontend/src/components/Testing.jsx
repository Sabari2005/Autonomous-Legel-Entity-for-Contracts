import { useState } from "react";

const Testing = () => {
    const [messages, setMessages] = useState([]);
    const [inputText, setInputText] = useState("");
    const [loading, setLoading] = useState(false);
    const sendMessage = () => {
        if (!inputText.trim()) return;
    
        setLoading(true);
        setMessages([]); // Clear old messages before new request
    
        const eventSource = new EventSource(`http://127.0.0.1:8000/summarizer_text_sse?text=${inputText}`);
    
        eventSource.onopen = () => console.log("✅ SSE Connection Opened");
    
        eventSource.onmessage = (event) => {
            setMessages(prevMessages => {
                const lastMessage = prevMessages.length ? prevMessages[prevMessages.length - 1] : "";
                return [...prevMessages.slice(0, -1), lastMessage + event.data];
            });
        
            if (event.data.includes("Connection closed")) {
                console.log("🔴 SSE Connection Closed by Server");
                eventSource.close();
                setLoading(false);
            }
        };
    
        eventSource.onerror = (error) => {
            console.error("❌ SSE Error:", error);
            eventSource.close(); // Ensure proper cleanup
            setLoading(false);
        };
    };
    

    return (
        <div>
            <input 
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Enter text"
            />
            <button onClick={sendMessage} disabled={loading}>Send</button>

            <div>
                <h3>Responses:</h3>
                {messages.map((msg, index) => (
                    <p key={index}>{msg}</p>
                ))}
            </div>
        </div>
    );
};

export default Testing;
