import { useState } from "react";

function tesTing() {
  const [progress, setProgress] = useState([]);
  const [file, setFile] = useState(null);
  const [ws, setWs] = useState(null);

  // ✅ Handle File Selection
  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  // ✅ Upload and Send File Over WebSocket
  const handleUpload = () => {
    if (!file) return alert("Please select a file!");

    const socket = new WebSocket("ws://4cb2-44-202-31-68.ngrok-free.app/ws/upload");

    socket.onopen = async () => {
      console.log("WebSocket connected");

      // ✅ Send file metadata first
      socket.send(JSON.stringify({ file_name: file.name }));

      // ✅ Read file as Base64
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
        const base64Data = reader.result.split(",")[1]; // Extract Base64 part
        socket.send(base64Data); // Send file as Base64 string
      };

      setWs(socket);
    };

    socket.onmessage = (event) => {
      const parsedData = JSON.parse(event.data);
      if (parsedData.status === "done") {
        socket.close();
      } else {
        setProgress((prev) => [...prev, parsedData]);
      }
    };

    socket.onerror = () => {
      console.error("WebSocket error");
      socket.close();
    };
  };

  return (
    // <div>
    //   <h1>Upload File via WebSocket</h1>
    //   <input type="file" accept=".pdf, .docx" onChange={handleFileChange} />
    //   <button onClick={handleUpload}>Upload</button>

    //   <h2>Live Risk Analysis</h2>
    //   {progress.map((update, index) => (
    //     <div key={index}>
    //       <p><b>Chunk {update.chunk_index}</b></p>
    //       <p>Risk Level: {update.risk_level}</p>
    //       <p>Risk Category: {update.risk_category}</p>
    //       <div dangerouslySetInnerHTML={{ __html: update.explanation }} />
    //       <hr />
    //     </div>
    //   ))}
    // </div>
    <h1>hello</h1>
  );
}

export default tesTing;
