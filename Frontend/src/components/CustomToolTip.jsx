import React, { useState } from "react";
import { motion } from "framer-motion"; // Import Framer Motion
import "./CustomToolTip.css"; // Import the custom CSS

const CustomTooltip = ({ children, tooltipText }) => {
  const [showTooltip, setShowTooltip] = useState(false);

  return (
    <div
      className="tooltip-wrapper"
      onMouseEnter={() => setShowTooltip(true)}
      onMouseLeave={() => setShowTooltip(false)}
    >
      {children}
      {showTooltip && (
        <motion.div
          className="tooltip"
          initial={{ opacity: 0, y: -10 }} // Tooltip starts invisible and slightly above
          animate={{ opacity: 1, y: 0 }} // Tooltip fades in and comes into place
          exit={{ opacity: 0, y: 10 }} // Tooltip fades out and moves slightly down
          transition={{ duration: 0.3 }} // Transition duration for smooth animation
        >
          {tooltipText}
        </motion.div>
      )}
    </div>
  );
};

export default CustomTooltip;
