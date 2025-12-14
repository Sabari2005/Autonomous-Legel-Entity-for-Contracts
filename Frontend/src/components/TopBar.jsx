import React, { useState,useEffect, useRef,useContext } from "react";
import "./Style.css";
import ReCAPTCHA from "react-google-recaptcha";
import {motion} from 'framer-motion'
import { useNavigate } from "react-router-dom";
import { jwtDecode } from "jwt-decode"; // ✅ Correct way
import axios from "axios";
import { AuthContext } from "../AuthContext";
function Topbar() {
    const { isSignedIn, setIsSignedIn, userDetails, setUserDetails, setRecaptchaVerified } = useContext(AuthContext);
      const recaptchaRef = useRef();
      const [showCaptcha, setShowCaptcha] = useState(false);

    const [menuOpen, setMenuOpen] = useState(false);
    const [scrolled, setScrolled] = useState(false);
    const navigate = useNavigate();
    // const navigate = useNavigate();
    const googleBtnRef = useRef(null);

    useEffect(() => {
        const handleScroll = () => {
            setScrolled(window.scrollY > 200); // True when scrolled past 200px
        };

        window.addEventListener("scroll", handleScroll);
        return () => window.removeEventListener("scroll", handleScroll);
    }, []);

    // useEffect(() => {
    //     // ✅ Initialize Google OAuth client
    //     if (window.google) {
    //         window.google.accounts.id.initialize({
    //             client_id: "984182788509-kh9no830dpn2kngk4hfk6efsco5iber9.apps.googleusercontent.com",
    //             callback: handleCredentialResponse,
    //         });
    //     }
    // }, []);
    useEffect(() => {
        if (window.google) {
            window.google.accounts.id.initialize({
                client_id: "984182788509-kh9no830dpn2kngk4hfk6efsco5iber9.apps.googleusercontent.com",
                callback: handleCredentialResponse,
                auto_select: false, // 👈 don't auto-login previous user
                prompt_parent_id: "recaptcha-popup", // optional: target container
            });
        }
    }, []);
    const handleCredentialResponse = (response) => {
        const userObject = jwtDecode(response.credential);
        console.log("✅ User Info:", userObject);

        setIsSignedIn(true);
        setUserDetails(userObject);

        navigate("/Rsk");
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
    const handleLoginClick = () => {
        if (isSignedIn) {
            navigate("/Rsk");
        } else {
            setShowCaptcha(true); // ⬅️ Show captcha first
        }
    };
    // const handleLoginClick = () => {
    //     if (isSignedIn) {
    //         console.log("✅ Already signed in. Routing to /Rsk...");
    //         navigate("/Rsk");
    //     } else if (window.google) {
    //         window.google.accounts.id.prompt(); // trigger login popup
    //     }
    // };
    return (
        <>
        <motion.div className={`Topbar ${scrolled?"scrolled":""}`} >
            <div className="Logo" >
                <img src="logo.svg" alt="Logo" />
                <span>Alec</span>
            </div>

            {/* ✅ Mobile Menu Icon */}
            <div className="menuIcon" onClick={() => setMenuOpen(!menuOpen)}>
                {menuOpen ? "✖" : "☰"}
            </div>

            {/* ✅ Navigation Menu */}
            <div className={`navMenu ${menuOpen ? "open" : ""}`}>
                <div className="navLinks">
                    <span>Feature</span>
                    <span>Pricing</span>
                    <span>Changelog</span>
                    <span>Contact</span>
                </div>

                {/* ✅ "Get Started" button appears only inside menu on mobile */}
                <button className="Special" onClick={handleLoginClick}>
                    <span>Login/Signup</span>
                </button>
            </div>

            {/* ✅ Keep "Get Started" button outside on larger screens */}
            <button className="Special desktopOnly" onClick={handleLoginClick}>
                <span>Login</span>
            </button>
        </motion.div>
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
              </>
    );
}

export default Topbar;