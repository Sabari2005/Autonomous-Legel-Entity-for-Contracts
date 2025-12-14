import React, { useState, useContext } from "react";
import "./UserDetailsStyle.css";
import { ExtractedTextContext } from "../ExtractedTextContext";

function UserDetails() {
  const { formData, setFormData,isEditable,setIsEditable } = useContext(ExtractedTextContext);


  const handleChange = (e) => {
    if (!isEditable) return;
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (isEditable) {
      // On first submit
      console.log(formData);
      setIsEditable(false); // Lock inputs
    } else {
      // On 'Change Data' click
      setIsEditable(true); // Unlock inputs
    }
  };

  return (
    <div className="UserDetails">
      <form onSubmit={handleSubmit}>
        <label>
          Sender Email
          <input
            type="email"
            name="sender_email"
            value={formData.sender_email}
            onChange={handleChange}
            disabled={!isEditable}
          />
        </label>

        <label>
          Receiver Email
          <input
            type="email"
            name="receiver_email"
            value={formData.receiver_email}
            onChange={handleChange}
            disabled={!isEditable}
          />
        </label>

        <label>
          Application Type
          <input
            name="application_type"
            value={formData.application_type}
            onChange={handleChange}
            disabled={!isEditable}
          />
        </label>

        <label>
          Counter Party Name
          <input
            name="counter_party_name"
            value={formData.counter_party_name}
            onChange={handleChange}
            disabled={!isEditable}
          />
        </label>

        <label>
          Counter Party Role
          <input
            name="counter_party_role"
            value={formData.counter_party_role}
            onChange={handleChange}
            disabled={!isEditable}
          />
        </label>

        <label>
          Counter Party Company Name
          <input
            name="counter_party_companyname"
            value={formData.counter_party_companyname}
            onChange={handleChange}
            disabled={!isEditable}
          />
        </label>

        <label>
          Sender Name
          <input
            name="send_party_name"
            value={formData.send_party_name}
            onChange={handleChange}
            disabled={!isEditable}
          />
        </label>

        <label>
          Sender Role
          <input
            name="sender_party_role"
            value={formData.sender_party_role}
            onChange={handleChange}
            disabled={!isEditable}
          />
        </label>

        <label>
          Sender Company
          <input
            name="sender_party_company"
            value={formData.sender_party_company}
            onChange={handleChange}
            disabled={!isEditable}
          />
        </label>

        <button type="submit">
          {isEditable ? "Submit" : "Change Data"}
        </button>
      </form>
    </div>
  );
}

export default UserDetails;
