import React, { useContext } from "react";
import "./Style.css";
import { ExtractedTextContext } from "../ExtractedTextContext";

function CaseDetails() {
  const { CaseDetails } = useContext(ExtractedTextContext);

  const caseList = CaseDetails?.case_details || [];

  return (
    <div className="CaseDetailsContain">
      <h2>Case Details</h2>
      {caseList.length > 0 ? (
        <table className="case-table">
          <thead>
            <tr>
              <th>Case Title</th>
              <th>Date</th>
              <th>Jurisdiction</th>
              <th>Risk Level</th>
              <th>Source</th>
            </tr>
          </thead>
          <tbody>
            {caseList.map((item, index) => (
              <tr key={index}>
                <td>{item.case_title || "N/A"}</td>
                <td>{item.date_recorded || "N/A"}</td>
                <td>{item.origin || "N/A"}</td>
                <td>{item.risk_level || "N/A"}</td> {/* If available */}
                <td>
                  {item.source_url ? (
                    <a href={item.source_url} target="_blank" rel="noopener noreferrer">
                      Link
                    </a>
                  ) : (
                    "N/A"
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p>No case details available.</p>
      )}
    </div>
  );
}

export default CaseDetails;
