import React, { useEffect, useState } from "react";

function ExplainedTextAccordion({ extractedExplainedText }) {
    const [loading, setLoading] = useState(true); // State to track loading

    useEffect(() => {
        if (extractedExplainedText) {
          // Simulate content rendering delay
          const scripts = document.querySelectorAll(".accordion-content script");
          scripts.forEach((oldScript) => {
            const newScript = document.createElement("script");
            newScript.textContent = oldScript.textContent; // Use textContent instead of innerHTML
            newScript.type = "text/javascript"; // Ensure correct type
            newScript.async = false; // Prevent async execution affecting layout
            oldScript.replaceWith(newScript);
          });
          // Set loading to false after the content is rendered
          setLoading(false);
        }
    }, [extractedExplainedText]);

    return (
        <div className="accordion-content">
            {loading ? (
                <div></div> // Display loading message while content is being rendered
            ) : (
                <div dangerouslySetInnerHTML={{ __html: extractedExplainedText || "" }} />
            )}
        </div>
    );
}

export default ExplainedTextAccordion;
