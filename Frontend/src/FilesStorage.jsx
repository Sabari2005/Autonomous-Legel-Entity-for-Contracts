import React from "react";
import SideBar from "./components/SideBar";
import "./components/Style.css"
import FilesStorageContain from "./components/FilesStorageContain";
function FilesStorage(){
    return(
        <div className="FilesStorage">
            <SideBar />
            <div className="FilesStorage-content">
                <div className="FilesStorage-header">
                    <span>Files Storage</span>
                </div>
                <FilesStorageContain />
            </div>
        </div>
    )
}

export default FilesStorage