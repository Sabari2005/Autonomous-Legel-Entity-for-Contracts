import React, { useEffect } from "react";
import "./Style.css";

const Notification = ({ message, color, onClose }) => {
  useEffect(() => {
    // Auto-hide notification after 3 seconds
    const timer = setTimeout(() => {
      onClose();
    }, 3000);
    return () => clearTimeout(timer);
  }, [onClose]);

  return (
    <div className="notification" style={{ backgroundColor: color, zIndex:100000 }}>
      {message}
    </div>
  );
};

export default Notification;
