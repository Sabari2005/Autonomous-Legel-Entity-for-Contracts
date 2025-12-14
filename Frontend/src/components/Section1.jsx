import React, { useState, useRef, useEffect,useContext } from "react";
import { useNavigate } from "react-router-dom";
import ReCAPTCHA from "react-google-recaptcha";
import { motion } from "framer-motion";
import { AuthContext } from "../AuthContext";
import axios from "axios";
import { jwtDecode } from "jwt-decode"; // ✅ Correct way
// import axios from "axios";
import "./Style.css";

function Section1() {

  const navigate = useNavigate();
  const recaptchaRef = useRef();
  const [showCaptcha, setShowCaptcha] = useState(false);
  
  const {
    isSignedIn,
    setIsSignedIn,
    setUserDetails,
    setRecaptchaVerified,
  } = useContext(AuthContext);

  const handleLoginClick = () => {
    if (isSignedIn) {
      navigate("/Rsk");
    } else {
      setShowCaptcha(true);
    }
  };
    const onRecaptchaChange = async (token) => {
        if (!token) return;
        try {
            const res = await axios.post("https://8000-01jrcj02963afp5r8dxtpsxdqz.cloudspaces.litng.ai/verify-recaptcha", { token });

            if (res.data.success) {
                setRecaptchaVerified(true); // ✅ Context state
                setShowCaptcha(false);
                window.google.accounts.id.prompt(); // ✅ Trigger Google login
            } else {
                alert("reCAPTCHA failed. Try again.");
            }
        } catch (err) {
            console.error("Recaptcha error:", err);
        }
    };
  
  useEffect(() => {
    if (window.google) {
      window.google.accounts.id.initialize({
        client_id:
          "984182788509-kh9no830dpn2kngk4hfk6efsco5iber9.apps.googleusercontent.com",
        callback: (response) => {
          const userObject = jwtDecode(response.credential);
          setIsSignedIn(true);
          setUserDetails(userObject);
          navigate("/Rsk");
        },
      });
    }
  }, []);

  return (
    <div className="section1">
      {/* Floating lines */}
      <div className="lines"></div>
      <div className="lines" style={{ left: "300px", width: "150px", top: -200 }}></div>
      <div className="lines" style={{ width: "300px", top: -300, left: 450 }}></div>
      <div className="lines" style={{ width: "200px", top: -600, left: 700, height: "140vh" }}></div>

      {/* Cloud animation */}
      <div className="background">
        <motion.img
          src="./cloud.png"
          animate={{
            x: [0, 20, 0],
            y: [0, 15, 0],
          }}
          transition={{
            duration: 10,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
      </div>

      {/* Main content */}
      <div className="content1">
        <div className="heading1">
          <div className="Master_logo">
            <div className="Master_Logo_layer1">
              <img src="logowhite.svg" alt="Logo" />
            </div>
          </div>
          <div className="heading_content">
            <span>Automate. Engage. Convert. Powered by AI.</span>
          </div>
          <span
            style={{
              fontSize: "1rem",
              minWidth: "300px",
              maxWidth: "400px",
              textAlign: "center",
            }}
          >
            Your journey to AI-powered documents starts here
          </span>
          <div className="btn-contain">
            <button className="GetStarted" onClick={handleLoginClick}>
              Get started
            </button>
            <button className="LearnMore">Learn more</button>
          </div>
        </div>
      </div>

      {/* reCAPTCHA popup modal */}
      {showCaptcha && (
        <div className="recaptcha-popup-backdrop">
          <div className="recaptcha-popup">
            <ReCAPTCHA
              sitekey="6LefvhUrAAAAAAkn3VWk86DQ7LnfBJKHhpEU1uaq"
              ref={recaptchaRef}
              onChange={onRecaptchaChange}
            />
          </div>
        </div>
      )}

    </div>
  );
}

export default Section1;
