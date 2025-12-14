import { useContext, useEffect } from "react";
import { Navigate } from "react-router-dom";
import { AuthContext } from "./AuthContext";

const ProtectedRoute = ({ children }) => {
  const { isSignedIn,  recaptchaVerified} = useContext(AuthContext);

  if (isSignedIn && recaptchaVerified) {
    return children;
  }

  // redirect to "/" and simulate login btn click via localStorage
  useEffect(() => {
    localStorage.setItem("triggerLogin", "true");
  }, []);

  return <Navigate to="/" replace />;
};

export default ProtectedRoute;
