import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App.jsx";
import { ExtractedTextProvider } from "./ExtractedTextContext"; // ✅ Import correctly
import { AuthContext, AuthProvider } from "./AuthContext.jsx";
ReactDOM.createRoot(document.getElementById("root")).render(
  <BrowserRouter> 
  <AuthProvider>  
    <ExtractedTextProvider> {/* ✅ Wrapped correctly */}
      
        <App />
      
    </ExtractedTextProvider>
    </AuthProvider>
  </BrowserRouter>
);
