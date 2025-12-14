import React, { useContext } from "react";
import { AuthContext } from "../AuthContext"; // Make sure path is correct
import "./Style.css";

function AccountContent() {
  const { userDetails, setIsSignedIn, setUserDetails } = useContext(AuthContext);

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("userDetails");
    localStorage.setItem("isSignedIn", false);
    setIsSignedIn(false);
    setUserDetails(null);
    window.location.href = "/"; // or use navigate("/")
  };

  return (
    <div className="Account-contain">
      <img
        className="profile_image"
        src={userDetails?.picture || "default_profile.png"}
        alt="profile"
      />
      <span style={{ fontWeight: 700, fontSize: "2.5rem", marginTop: 50 }}>
        Welcome to Alec
      </span>
      <span style={{ fontWeight: 700, fontSize: "2rem", marginTop: 10 }}>
        {userDetails?.name || "username"}
      </span>
      <button className="wantHover" style={{padding:10,marginTop:50,backgroundColor:"grey",borderRadius:20}} onClick={handleLogout}>
        Logout
      </button>
    </div>
  );
}

export default AccountContent;
