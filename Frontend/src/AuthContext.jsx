// AuthContext.jsx
import React, { createContext, useState, useEffect } from "react";

const AuthContext = createContext();

const AuthProvider = ({ children }) => {
    const [isSignedIn, setIsSignedIn] = useState(() => localStorage.getItem("isSignedIn") === "true");
    const [userDetails, setUserDetails] = useState(() => {
        const storedUser = localStorage.getItem("userDetails");
        return storedUser ? JSON.parse(storedUser) : null;
    });

    const [recaptchaVerified, setRecaptchaVerified] = useState(() => {
        return localStorage.getItem("recaptcha_verified") === "true";
    });

    useEffect(() => {
        localStorage.setItem("isSignedIn", isSignedIn);
        if (userDetails) {
            localStorage.setItem("userDetails", JSON.stringify(userDetails));
        }
    }, [isSignedIn, userDetails]);

    useEffect(() => {
        localStorage.setItem("recaptcha_verified", recaptchaVerified);
    }, [recaptchaVerified]);
  return (
    <AuthContext.Provider
      value={{
        isSignedIn, 
            setIsSignedIn, 
            userDetails, 
            setUserDetails,
            recaptchaVerified,
            setRecaptchaVerified 
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export { AuthContext, AuthProvider };
