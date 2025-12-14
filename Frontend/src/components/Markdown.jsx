import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const formatMarkdown = (text) => {
    return text
        .replace(/(?<!\n)##/g, "\n\n##") // Ensure headers start on a new line
        .replace(/(?<!\n)- /g, "\n- ")   // Ensure list items are on new lines
        .replace(/\.\s*##/g, ".\n\n##")  // Ensure section separation
        .replace(/\| --\n+\s*\| --/g, "\n|--|--|") // Fix broken table separators
        .replace(/\|\s*\|/g, "|\n|") // Ensure each row starts properly
        .replace(/\n\n+/g, "\n\n"); // Remove excessive newlines
};

const MarkdownRenderer = ({ markdownText }) => {
    const formattedMarkdown = formatMarkdown(markdownText);

    return (
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {formattedMarkdown}
        </ReactMarkdown>
    );
};

export default MarkdownRenderer;
