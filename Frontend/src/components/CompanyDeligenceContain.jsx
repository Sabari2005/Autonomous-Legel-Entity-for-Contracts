import React, { useState, useContext, useEffect, useRef } from "react";
import Notification from "./Notification";
import CustomTooltip from "./CustomToolTip";
import { ExtractedTextContext } from "../ExtractedTextContext";
import { motion, AnimatePresence } from "framer-motion";

import "./Style.css";
import CompanyDeligenceReport from "./CompanyDeligenceReport";
import CompanyLoader from "./CompanyLoader";
import ProcessLoader from "./ProcessLoader";

function CompanyDeligenceContain() {
  const { setCpyDelReportPath, CpyDelReportPath, setCpyDelImgPaths, CpyDelImgPaths, setCaseDetails, setFinalReport, setSidebarOperation } =
    useContext(ExtractedTextContext);
    const [timerText, setTimerText] = useState("");
const [timerExpired, setTimerExpired] = useState(false);

  const [formData, setFormData] = useState({
    ticker_symbol: "",
    ticker_name: "",
    corporation: "",
    jurisdiction: "",
    period: "",
    region: "",
  });

  const [notification, setNotification] = useState(null);
  const [invalidFields, setInvalidFields] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;

    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));

    if (invalidFields.includes(name) && value.trim() !== "") {
      setInvalidFields((prev) => prev.filter((field) => field !== name));
    }
  };

  const showNotification = (message, color) => {
    setNotification({ message, color });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const emptyFields = Object.entries(formData)
      .filter(([key, value]) => value.trim() === "")
      .map(([key]) => key);

    if (emptyFields.length > 0) {
      setInvalidFields(emptyFields);
      showNotification("Please fill in all the fields.", "#e63946");
      return;
    }

    setLoading(true);
    setSidebarOperation(true);

    try {
      const response = await fetch("https://8000-01jrcj02963afp5r8dxtpsxdqz.cloudspaces.litng.ai/CpyDelData", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      if (response.ok) {
        const data = await response.json();
        showNotification("Company details submitted successfully!", "#2a9d8f");

        setFormData({
          ticker_symbol: "",
          ticker_name: "",
          corporation: "",
          jurisdiction: "",
          period: "",
          region: "",
        });

        setInvalidFields([]);
        setCpyDelReportPath(data.report_path);
        setCpyDelImgPaths(data.image_paths);
        setCaseDetails(data.json_file);
        setFinalReport(data.final_report);
        console.log(data.image_paths, data.report_path);
      } else {
        const errorText = await response.text();
        showNotification(`Error: ${errorText}`, "#e76f51");
      }
    } catch (error) {
      showNotification(`Error connecting to server.`, "#e76f51");
      console.error("Submission error:", error);
    } finally {
      setLoading(false);
      setSidebarOperation(false);
    }
  };

  const inputStyle = (name) =>
    invalidFields.includes(name) ? { border: "2px solid red", outline: "none" } : {};

  const isReportReady = CpyDelReportPath || (CpyDelImgPaths && CpyDelImgPaths.length > 0);
  const timerRef = useRef(null); // for cleanup

useEffect(() => {
  if (loading) {
    let timeLeft = 420; // 5 minutes = 300 seconds
    setTimerExpired(false); // reset message
    setTimerText("07:00");

    timerRef.current = setInterval(() => {
      timeLeft -= 1;
      const minutes = String(Math.floor(timeLeft / 60)).padStart(2, '0');
      const seconds = String(timeLeft % 60).padStart(2, '0');
      setTimerText(`${minutes}:${seconds}`);

      if (timeLeft <= 0) {
        clearInterval(timerRef.current);
        setTimerExpired(true);
      }
    }, 1000);
  }

  return () => clearInterval(timerRef.current);
}, [loading]);
  return (
    <div style={{ display: "flex", justifyContent: "center", alignItems: "center", height: "100vh", width: "95%", backgroundColor:"#E3E9EE"}}>
      {notification && (
        <Notification
          message={notification.message}
          color={notification.color}
          onClose={() => setNotification(null)}
        />
      )}
      <AnimatePresence mode="wait">
      {loading ? (
        <motion.div 
        key="loader"
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.9 }}
        transition={{ duration: 0.3 }}
          style={{ display: "flex", justifyContent: "center", alignItems: "center", height: "100vh",flexDirection:"column" }}>
          {/* <div> */}
  <ProcessLoader />
  <span style={{ marginTop: "100px", fontSize: "18px", color: timerExpired ? "green" : "#333" }}>
    {timerExpired ? "It's taking longer than expected... Please wait" : `Time left: ${timerText}`}
  </span>
  <span style={{color:"black"}}>Please do not refresh the web page</span>
{/* </div> */}
        </motion.div>
      ) : !isReportReady ? (
        <motion.div
        key="form"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        transition={{ duration: 0.3 }} 
        className="ComapnyDeligenceContain">
          <span style={{ fontSize: "2rem", fontWeight: 700, marginTop: 50 }}>
            Verify Company's Background
          </span>

          <div className="form-container">
            <form className="form-box" onSubmit={handleSubmit}>
              <div style={{ display: "flex", justifyContent: "space-around", gap: 100 }}>
                <div className="form-contain">
                  <div className="form-group">
                    <label htmlFor="ticker_symbol">
                      Ticker Symbol
                      <CustomTooltip tooltipText="Unique stock identifier used in financial markets (e.g., AAPL for Apple Inc.).">
                        <span className="question">?</span>
                      </CustomTooltip>
                    </label>
                    <input
                      type="text"
                      id="ticker_symbol"
                      name="ticker_symbol"
                      value={formData.ticker_symbol}
                      onChange={handleChange}
                      style={inputStyle("ticker_symbol")}
                      placeholder="Enter ticker symbol"
                    />
                  </div>

                  <div className="form-group">
                    <label htmlFor="ticker_name">
                      Ticker Name
                      <CustomTooltip tooltipText="Full name or abbreviation of the traded security.">
                        <span className="question">?</span>
                      </CustomTooltip>
                    </label>
                    <input
                      type="text"
                      id="ticker_name"
                      name="ticker_name"
                      value={formData.ticker_name}
                      onChange={handleChange}
                      style={inputStyle("ticker_name")}
                      placeholder="Enter ticker name"
                    />
                  </div>

                  <div className="form-group">
                    <label htmlFor="corporation">
                      Corporation
                      <CustomTooltip tooltipText="Enter the legal name of the company.">
                        <span className="question">?</span>
                      </CustomTooltip>
                    </label>
                    <input
                      type="text"
                      id="corporation"
                      name="corporation"
                      value={formData.corporation}
                      onChange={handleChange}
                      style={inputStyle("corporation")}
                      placeholder="Enter corporation"
                    />
                  </div>
                </div>

                <div className="form-contain">
                  <div className="form-group">
                    <label htmlFor="jurisdiction">
                      Jurisdiction
                      <CustomTooltip tooltipText="Legal authority governing the company's operations or filings.">
                        <span className="question">?</span>
                      </CustomTooltip>
                    </label>
                    <input
                      type="text"
                      id="jurisdiction"
                      name="jurisdiction"
                      value={formData.jurisdiction}
                      onChange={handleChange}
                      style={inputStyle("jurisdiction")}
                      placeholder="Enter jurisdiction"
                    />
                  </div>

                  <div className="form-group">
                    <label htmlFor="period">
                      Period
                      <CustomTooltip tooltipText="Reporting period for the financial or legal data (e.g., Q4 2023).">
                        <span className="question">?</span>
                      </CustomTooltip>
                    </label>
                    <input
                      type="text"
                      id="period"
                      name="period"
                      value={formData.period}
                      onChange={handleChange}
                      style={inputStyle("period")}
                      placeholder="Enter period (e.g. 2023 or Q4 2023)"
                    />
                  </div>

                  <div className="form-group">
                    <label htmlFor="region">
                      Region
                      <CustomTooltip tooltipText="Input the region code for localization or data filtering.">
                        <span className="question">?</span>
                      </CustomTooltip>
                    </label>
                    <input
                      type="text"
                      id="region"
                      name="region"
                      value={formData.region}
                      onChange={handleChange}
                      style={inputStyle("region")}
                      placeholder="Enter region (e.g. kr)"
                    />
                  </div>
                </div>
              </div>

              <button type="submit" className="submit-btn">
                Analyze
              </button>
            </form>
          </div>
        </motion.div>
      ) : (

          <CompanyDeligenceReport />

      )}
      </AnimatePresence>
    </div>
  );
}

export default CompanyDeligenceContain;
