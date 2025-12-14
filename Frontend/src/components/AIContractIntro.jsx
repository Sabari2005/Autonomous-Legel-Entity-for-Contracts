import React,{useContext} from "react";
import "./Style.css";
import { motion } from "framer-motion";
import { ExtractedTextContext } from "../ExtractedTextContext";
import UserDetails from "./UserDetails";
const Card = ({ text, onCardClick }) => {
    return (
        <motion.div 
            className="card-template"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            whileHover={{ scale: 1.1 }}
            transition={{ duration: 0.5 }}
            onClick={onCardClick} // ✅ Each card has its own function
        >
            <motion.div 
                style={{ display: "flex", justifyContent: "flex-end", alignItems: "center", width: "100%", height: "40px" }}
            >
                <motion.img 
                    src="/arrow1.png" 
                    height={40} 
                    width={40} 
                    style={{ paddingRight: "10px" }}

                />
            </motion.div>
            <div className="content_card">
                <span>{text}</span>
            </div>
        </motion.div>
    );
};

function AIContractIntro({ sendMessage }) {  
    // ✅ Separate onClick functions for each card
    const handleCloudServiceClick = () => sendMessage("cloud_services");
    const handleSLAclick = () => sendMessage("Service_Level_Agreement");
    const handleSoftwareDevClick = () => sendMessage("Software_development_agreement");
    const handleSoftwareLicenseClick = () => sendMessage("Software_license_agreement");
    const handleSoftwareMaintenanceClick = () => sendMessage("Software_maintenance_agreement");
    const {setShowDetailsPopup,showDetailsPopup}=useContext(ExtractedTextContext)
    return (
        <div className="IntroAIContract">
             {showDetailsPopup && (
              <div className="user-details-popup-overlay">
                <div className="user-details-popup">
                  <button className="close-button" onClick={() => setShowDetailsPopup(false)}>×</button>
                  <UserDetails />
                </div>
              </div>
            )}
            <div className="heading1" style={{ justifyContent: "flex-start", marginTop: 100, height: "70vh" }}>
                <div className="Master_logo" style={{ marginTop: "0px" }}>
                    <div className="Master_Logo_layer1">
                        <img src="/logowhite.svg" alt="Logo" />
                    </div>   
                </div>
                <div className="heading_content">
                    <span>Ask Anything To Draft.</span>
                </div>
                <div className="intro-corousel">
                    <Card text="Create a Cloud Service Agreement" onCardClick={handleCloudServiceClick} />
                    <Card text="Create a Service Level Agreement" onCardClick={handleSLAclick} />
                    <Card text="Create a Software Development Agreement" onCardClick={handleSoftwareDevClick} />
                    <Card text="Create a Software License Agreement" onCardClick={handleSoftwareLicenseClick} />
                    <Card text="Create a Software Maintenance Agreement" onCardClick={handleSoftwareMaintenanceClick} />
                </div>
            </div>
        </div>
    );
}

export default AIContractIntro;
