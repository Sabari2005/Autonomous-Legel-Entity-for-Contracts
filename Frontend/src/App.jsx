import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "./Home.jsx";
import "./index.css";
import RiskAnalysis from "./RiskAnalysis.jsx";
import AIContract from "./AIcontract.jsx";
import FilesStorage from "./FilesStorage.jsx";
import Account from "./Account.jsx";
import Settings from "./Settings.jsx";
import Summerrizer from "./Summerizer.jsx";
import ProtectedRoute from "./ProtectedRout.jsx";
import Company_Deligence from "./Company_Deligence.jsx";

function App() {
  return (
    <Routes>
      {/* Public route */}
      <Route path="/" element={<Home />} />

      {/* Protected routes */}
      <Route
        path="/RSk"
        element={
          <ProtectedRoute>
            <RiskAnalysis />
          </ProtectedRoute>
        }
      />
      <Route
        path="/AiC"
        element={
          <ProtectedRoute>
            <AIContract />
          </ProtectedRoute>
        }
      />
      <Route
        path="/Fs"
        element={
          <ProtectedRoute>
            <FilesStorage />
          </ProtectedRoute>
        }
      />
      <Route
        path="/Acct"
        element={
          <ProtectedRoute>
            <Account />
          </ProtectedRoute>
        }
      />
      <Route
        path="/Sts"
        element={
          <ProtectedRoute>
            <Settings />
          </ProtectedRoute>
        }
      />
      <Route
        path="/Smz"
        element={
          <ProtectedRoute>
            <Summerrizer />
          </ProtectedRoute>
        }
      />
      <Route
        path="/Cde"
        element={
          <ProtectedRoute>
            <Company_Deligence />
          </ProtectedRoute>
        }
      />
    </Routes>
  );
}

export default App;
