import React,{useContext} from "react";
import SideBar from "./components/SideBar";
import "./components/Style.css";
import AccountContent from "./components/AccountContent";

function Account(){
    return(
        <div className="Account">
            <SideBar />
            <div className="Account-content">
                <AccountContent />
            </div>
        </div>
    )
}
export default Account