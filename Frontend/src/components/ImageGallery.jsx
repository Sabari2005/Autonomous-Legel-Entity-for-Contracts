import React, { useState } from 'react';
import './ImageGallery.css';

const ImageGallery = ({ imgPaths, baseLink }) => {
  const [selectedImage, setSelectedImage] = useState(null);
  console.log("imgPaths:", imgPaths);

  return (
    <div className="CompanyDeligenceReportImages">
      <h3>Report Visuals</h3>
      <div className="image-grid">
        {imgPaths.map((imgPath, index) => {
          const fullImgPath = `${baseLink}${imgPath}`;
          const imageName = imgPath.split('/').pop();  // Extract file name from path

          return (
            <div key={index} className="image-card" onClick={() => setSelectedImage(fullImgPath)}>
              <img
                src={fullImgPath}
                alt={`report_image_${index}`}
              />
              <p className="image-name">{imageName}</p> {/* Display the filename */}
            </div>
          );
        })}
      </div>

      {/* Modal Popup */}
      {selectedImage && (
        <div className="image-modal" onClick={() => setSelectedImage(null)}>
          <img src={selectedImage} alt="Zoomed In" />
        </div>
      )}
    </div>
  );
};

export default ImageGallery;
