import React from "react";
import Sidebar from "./components/SideBar";
import CompanyDeligenceContain from "./components/CompanyDeligenceContain";

function Company_Deligence() {
  return (
    <div className="Company_Deligence">
        <Sidebar />
        <div style={{width:"100%",height:"100vh",display:"flex",justifyContent:"flex-end",alignItems:"center", color:"white"}}>
          <CompanyDeligenceContain />
        </div>
    </div>
  );
}

export default Company_Deligence;