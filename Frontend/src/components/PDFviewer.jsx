import { Worker, Viewer } from "@react-pdf-viewer/core";
import "@react-pdf-viewer/core/lib/styles/index.css";

const PdfViewer = ({ pdfFile }) => {
  const fullPdfUrl = `http://127.0.0.1:8000/pdf/${pdfFile}`;

  return (
    <div style={{ height: "600px", border: "1px solid #ddd", padding: "10px" }}>
      <Worker workerUrl={`https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.12.313/pdf.worker.min.js`}>
        <Viewer fileUrl={fullPdfUrl} />
      </Worker>
    </div>
  );
};

export default PdfViewer;
