# 2025-Legal-Document-Analysis

#### Solution by **CODE BLENDERS**

## Table of contents

- [Overview](#overview)
<!-- - [Screenshot](#screenshot) -->
- [Architecture Diagram](#architecture-diagram)
- [Services we provide](#services-we-provide)
- [Built with](#built-with)
<!-- - [Installation](#installation) -->
<!-- - [Project structure](#project-structure) -->
<!-- - [Risk Analysis System](#risk-analysis-system) -->
<!-- - [Database Design](#database-design) -->
- [Author](#author)

---

## Overview

- This repository presents a solution for **automated legal document processing**, focusing on contract drafting, clause refinement, risk assessment, and compliance validation.
- It leverages **Agentic AI**, **NLP**, and **Explainable AI (XAI)** to streamline contract workflows, enhance transparency, and minimize legal and financial risks.
 ### INTRODUCING ALEC
  - `ALEC` - **Autonomous Legal Entity for Contracts** is a solution to the problem of manual, time-consuming, and error-prone legal document analysis and contract drafting in today’s legal industry.
  - A unified platform that integrates contract risk analysis, contract summarization, template-based contract drafting, company background analysis, and autonomous contract generation — all in one system.

----





## Architecture Diagram
  ![](./assets/Images/Architecture.png)

## Services we provide
- Risk Analyzer
- Contract Summerization
- Company Deligence
- AI Contracter
- Autonomous Contract Drafter
- File Storage

## **Risk Analyzer**

### Purpose : 
To perform clause-by-clause risk assessment on contracts using a BERT-large model and Explainable AI (SHAP) for transparency.
### User Interface:
  ![](./assets/Images/Risk_Analyzer_banner.png)
### Usecase : 
A user uploads a contract, and the system identifies risk levels and categories for each clause. High-risk clauses are highlighted and modified. Users can compare original vs. modified contracts, then edit, save, and download the results.
### Advantages :
- Clause-level granularity in risk detection
- Transparent explanations using SHAP
- Automated risk mitigation suggestions


## **Contract Summarization**

### Purpose : 
To condense lengthy legal content while preserving critical information and risk context.
### User Interface
  ![](./assets/Images/Summerizer_banner.png)
### Usecase : 
A user inputs or uploads a contract or a section of it. The system chunks the input, analyzes risks, and summarizes key content, highlighting important terms in bold for quick review.
### **Advantages** :
- Accelerates contract review process
- Retains original legal intent and risk profile
- Highlights critical changes and keywords

## **Company Diligence**

### Purpose :
To generate a legal and financial profile of a company using real-time data aggregation from APIs.
### User Interface
  ![](./assets/Images/Company_Deligence_banner.png)
### Usecase : 
A user provides details such as company name or ticker. The system uses OpenCorporates, SerpAPI, AlphaVantage, and NewsAPI to collect case history, legal news, and financial trends and generates a downloadable diligence report.
### Advantages :
- Centralized due diligence with visual insights
- Supports better negotiation and risk decisions
- Downloads structured reports for documentation


## **Autonomous Contract Drafting**

### User Interface
  ![](./assets/Images/Contract_Drafter_banner.png)

 - ## Template Based Drafting 
    ### Purpose : 
    To enable users to draft contracts from pre-uploaded legal templates using a guided, interactive AI workflow.  
    ### usecase : 
    A user selects a contract template from a library, responds to dynamic AI-generated questions, reviews the generated content, customizes it, and downloads the final contract.
    ### Advantages :
     - Saves time by automating boilerplate drafting
     - Ensures legal standardization across templates
     - Allows editing flexibility before finalization

 - ## Autonomous Drafting
    ### Purpose : 
    To automate the entire contract creation and negotiation process through intelligent agents.
    ### Usecase : 
    A user provides basic instructions on the contract type. The AI interacts to gather required details, drafts the contract, and allows user edits. The system then sends it to the counterparty, negotiates terms, and schedules signing.
    ### Advantages :
     - End-to-end automation of contract lifecycle
     - Reduces manual back-and-forth in negotiations
     - Increases turnaround time with embedded AI agents

## **File Storage**

### Purpose :
To store and manage all files associated with the system, including uploaded legal documents, generated reports, user chat histories, and session data.
### User Interface
  ![](./assets/Images/File_Storage_banner.png)
### Usecase :
When a user uploads a contract for analysis or drafting, the file is securely stored in the system's database. Any reports generated—such as risk assessments, company diligence summaries, or modified contracts—are also saved. User chat sessions and interactions with AI agents are logged to maintain context and allow seamless session continuity. Users can revisit, edit, or download their documents and chat history at any time.
### Advantages :
- Centralized storage for contracts, reports, and session data
- Secure document management with controlled access
- Enables session resumption and audit trail generation
- Easy retrieval and download of processed files and reports
- Facilitates historical analysis and user behavior tracking

----

## Unique Features:
  - **AI-generated contract drafts** based on user inputs and templates
  - **Clause-level risk classification** with legal context awareness
  - **XAI** models to justify predictions and ensure accountability
  - **Autonomous negotiation agents** and final contract builders
  - Real-time data integration for financial, legal, and reputational profiling for **analyzing companies background**

----

## Built with

- ### Frontend:
  - React.js

- ### Backend:
  - FastAPI
  - Python 3.10
  - MongoDB (NoSQL)
  - Pinecone (Vector DB)

- ### AI Models:
  - Risk Analysis BERT 
  - LLaMA 3.3 70B
  - All-MiniLM-v2-L6

- ### Frameworks & Techniques:
  - TensorFlow, PyTorch
  - SHAP (for XAI)
  - Regex, Contrastive Learning
  - RAG (Retrieval-Augmented Generation)

- ### APIs:
  - OpenCorporates
  - SerpAPI
  - AlphaVantage
  - NewsAPI

---



<center><h1>Autonomous Legel Entity for Contracts</h1></center>

![](./assets/Images/image.png)


## Author

- Sabari Vadivelan S (Team Leader) - Contact Gmail [sabari132005@gmail.com]()
- Uvarajan D
- Anto Sam Christ A
- Sarathi S


# ALEC PROMO VIDEO

[Click here to watch the promo video of our project](./assets/alec_promo.mp4)


Ai Powered Legal Document Analysis - https://idea.unisys.com/D8109
