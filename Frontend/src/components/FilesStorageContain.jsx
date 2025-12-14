import React, { useEffect, useState, useContext } from "react";
import "./Style.css";
import { FaFileDownload } from "react-icons/fa";
import { AuthContext } from "../AuthContext";

function FilesStorageContain() {
    const [files, setFiles] = useState([]);
    const { userDetails } = useContext(AuthContext);
    const link = "https://8000-01jrcj02963afp5r8dxtpsxdqz.cloudspaces.litng.ai";

    useEffect(() => {
        if (userDetails) {
            fetch(`${link}/api/files`, {
                headers: {
                    'userid': userDetails.sub
                }
            })
                .then(res => {
                    if (!res.ok) {
                        throw new Error(`HTTP error! status: ${res.status}`);
                    }
                    return res.json();
                })
                .then(data => {
                    if (Array.isArray(data)) {
                        setFiles(data);
                    } else {
                        console.error("Expected array but got:", data);
                        setFiles([]);
                    }
                })
                .catch(err => console.error("Error fetching files:", err));
        }
    }, [userDetails]);

    function get_files() {
        if (!userDetails || !userDetails.id) return;
        fetch(`${link}/api/files`, {
            headers: {
                'userid': userDetails.sub
            }
        })
            .then(res => {
                if (!res.ok) {
                    throw new Error(`HTTP error! status: ${res.status}`);
                }
                return res.json();
            })
            .then(data => {
                if (Array.isArray(data)) {
                    setFiles(data);
                } else {
                    console.error("Expected array but got:", data);
                    setFiles([]);
                }
            })
            .catch(err => console.error("Error fetching files:", err));
    }

    const getFileIcon = (type) => {
        if (!type) return "unknown-icon.png";
        type = type.toLowerCase();
        if (type.includes("pdf")) return "pdf-icon.png";
        if (type.includes("word") || type.includes("doc")) return "doc-icon.png";
        if (type.includes("text") || type.includes("txt")) return "txt-icon.png";
        return "unknown-icon.png";
    };

    const convertBytesToMB = (bytes) => {
        return (bytes / 1024).toFixed(2);
    };

    const groupFilesByDate = () => {
        const groupedFiles = {
            today: [],
            yesterday: [],
            older: []
        };

        if (!Array.isArray(files)) return groupedFiles;

        files.forEach((file) => {
            const fileDate = new Date(new Date(file.lastModified).toLocaleString("en-US", { timeZone: "Asia/Kolkata" }));
            const today = new Date(new Date().toLocaleString("en-US", { timeZone: "Asia/Kolkata" }));
            const yesterday = new Date(today);
            yesterday.setDate(today.getDate() - 1);

            if (fileDate.toDateString() === today.toDateString()) {
                groupedFiles.today.push(file);
            } else if (fileDate.toDateString() === yesterday.toDateString()) {
                groupedFiles.yesterday.push(file);
            } else {
                groupedFiles.older.push(file);
            }
        });

        return groupedFiles;
    };

    const groupedFiles = groupFilesByDate();

    return (
        <div className="FilesList">
            <div className="FilesMetaDataDetails">
                <span style={{ width: "40%" }}>Name</span>
                <span style={{ width: "20%" }}>Last Modified</span>
                <span style={{ width: "20%" }}>File Size</span>
                <button className="refreshBtn" onClick={get_files}>Refresh</button>
            </div>

            <div className="FilesListContain">
                {["today", "yesterday", "older"].map((key) => (
                    groupedFiles[key].length > 0 && (
                        <div key={key}>
                            <span style={{ marginLeft: "20px" }}>
                                {key === "today" ? "Today" : key === "yesterday" ? "Yesterday" : "Older Files"}
                            </span>
                            {groupedFiles[key].map((file, index) => (
                                <div className="FilesTemplate" key={index}>
                                    <img
                                        src={`public/${getFileIcon(file.type)}`}
                                        height={30}
                                        width={30}
                                        style={{ objectFit: "contain", marginLeft: "10px" }}
                                        alt="file-icon"
                                    />
                                    <div className="FilesData">
                                        <span style={{ width: "52%", marginLeft: "10px" }}>{file.name}</span>
                                        <span style={{ width: "28.5%" }}>{new Date(file.lastModified).toLocaleString()}</span>
                                        <span style={{ width: "10%" }}>{convertBytesToMB(file.size)} KB</span>
                                    </div>
                                    <div className="FilesTools">
                                        <a href={`${link}${file.url}`} download className="Files-download-btn">
                                            <FaFileDownload />
                                        </a>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )
                ))}
            </div>
        </div>
    );
}

export default FilesStorageContain;
