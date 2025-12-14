import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from langchain_ollama import OllamaLLM
import json
from typing import Dict, List
import textwrap
import os
from datetime import datetime
from financial_risk_eveluator import FinancialRiskEvaluator 
from company_data_visualizer import CompanyDataVisualizer 
from crporate_legal_moniter import CorporateLegalMonitor 
from legal_case_analyzer import LegalCaseAnalyzer 
from company_background_analyzer import CompanyBackgroundAnalyzer 




def Financial_analysis_Agent(ticker_symbol,ticker_name):
    analyzer = FinancialRiskEvaluator(finance_api_key="YOUR API KEY",news_api_key="YOUR NEWS API KEY")
    analysis_result = analyzer.evaluate_company(ticker_symbol, ticker_name)
    visualizer = CompanyDataVisualizer('/teamspace/studios/this_studio/Uvarajan/Whole_Server/uploads/pro/Stock_analyses_output/stock_analysis.json')
    visualizer.generate_all_reports()




def Legal_analyisi_Agent(corporation,jurisdiction,period):
    legal_analyzer = CorporateLegalMonitor(api_key="YOUR API KEY")
    analysis_output = legal_analyzer.investigate_legal_issues(corporation,jurisdiction,period)
    if analysis_output["analysis_status"] == "completed":
        print(f"Analysis saved to: {analysis_output['results_path']}")
    case_analyzer = LegalCaseAnalyzer('/teamspace/studios/this_studio/Uvarajan/Whole_Server/uploads/pro/Legal_analyses_output/legal_analysis.json')
    case_analyzer.generate_all_legal_reports()

def Legal_analyisi_Agent(corporation,jurisdiction,period):
    legal_analyzer = CorporateLegalMonitor(api_key="YOUR API KYE")
    analysis_output = legal_analyzer.investigate_legal_issues(corporation,jurisdiction,period)
    if analysis_output["analysis_status"] == "completed":
        print(f"Analysis saved to: {analysis_output['results_path']}")
    case_analyzer = LegalCaseAnalyzer('/teamspace/studios/this_studio/Uvarajan/Whole_Server/uploads/pro/Legal_analyses_output/legal_analysis.json')
    case_analyzer.generate_all_legal_reports()




def company_Background_anslysis_Agent(company_name,region):
    analyzer = CompanyBackgroundAnalyzer("YOUR API KEY")
    companies = analyzer.search_companies(company_name, region, limit=3)
    
    if not companies:
        print("No companies found matching search criteria")
        return
    company = companies[0]['company']
    print(f"Analyzing: {company['name']} ({company['jurisdiction_code']})")
    
    full_profile = analyzer.get_company_full_profile(
        company['jurisdiction_code'],
        company['company_number']
    )
    
    network_analysis = analyzer.analyze_company_network(
        company['jurisdiction_code'],
        company['company_number'],
        depth=2
    )
    
    analyzer.data["company_profile"] = full_profile
    analyzer.data["network_analysis"] = network_analysis
    
    success, filename = analyzer.save_analysis()
    if success:
        print(f"Company background analysis saved to {filename}")
    else:
        print(f"Failed to save analysis: {filename}")



# ticker_symbol = "INFY"
# ticker_name = "Infosys Ltd"
# corporation = "Infosys Limited"
# jurisdiction = "IN"  # India
# period = "2023"  # or use a specific quarter like "Q4 2023"
# region = "in"  # country code for India

# ticker_symbol = "005930.KQ"  # Samsung Electronics' ticker on the Korea Exchange (KOSPI)
# ticker_name = "Samsung Electronics Co Ltd"
# corporation = "Samsung Electronics Co., Ltd."
# jurisdiction = "KR"  # South Korea
# period = "2023"  # or use a specific quarter like "Q4 2023"
# region = "kr"  # country code for South Korea



# Financial_analysis_Agent(ticker_symbol,ticker_name)
# Legal_analyisi_Agent(corporation,jurisdiction,period)
# company_Background_anslysis_Agent(ticker_name,region)





# Initialize Llama model
llama = OllamaLLM(
    model="llama3.3",
    temperature=0.2,
    top_p=0.9,
    repeat_penalty=1.1,
    num_ctx=4096
)

def chunk_data(data: Dict, max_chars: int = 1500) -> List[str]:
    """Smart JSON chunking that preserves structure"""
    if isinstance(data, dict):
        chunks = []
        current_chunk = {}
        current_size = 0
        
        for key, value in data.items():
            item_str = json.dumps({key: value})
            if current_size + len(item_str) > max_chars and current_chunk:
                chunks.append(json.dumps(current_chunk))
                current_chunk = {}
                current_size = 0
                
            current_chunk[key] = value
            current_size += len(item_str)
            
        if current_chunk:
            chunks.append(json.dumps(current_chunk))
        return chunks
    return textwrap.wrap(json.dumps(data), width=max_chars)

def analyze_with_llama(data: Dict, analysis_type: str) -> str:
    """Enhanced analysis that combines chunks into unified analysis"""
    chunks = chunk_data(data)
    context = ""
    full_analysis = ""
    
    system_prompts = {
        "financial": """You are a financial risk analyst. Analyze and combine all chunks to create:
1. Unified financial assessment
2. Consolidated risk score (0-100)
3. Integrated key metrics analysis
4. Complete fraud risk evaluation""",
        
        "corporate": """You are a corporate structure analyst. Synthesize all chunks into:
1. Comprehensive entity status report
2. Complete ownership structure
3. Final data quality assessment
4. Unified complexity score""",
        
        "legal": """You are a legal risk specialist. Combine all chunks to produce:
1. Complete case volume analysis
2. Final severity assessment
3. Integrated jurisdictional risk
4. Consolidated legal exposure score"""
    }
    

    all_chunks = []
    for i, chunk in enumerate(chunks, 1):
        prompt = f"""SYSTEM: {system_prompts[analysis_type]}
        
CONTEXT FROM PREVIOUS ANALYSIS:
{context[-800:] if context else 'No prior context'}

NEW DATA CHUNK ({i}/{len(chunks)}):
{chunk}

INSTRUCTIONS:
- Analyze this chunk in context of complete analysis
- Prepare to combine with other chunks
- Identify connections to previous data"""
        
        response = llama.invoke(prompt)
        all_chunks.append(response)
        context = f"{context}\n{response}"[-1000:]
    
    chunk_separator = "\n\n".join(all_chunks)
    combine_prompt = f"""SYSTEM: Combine all analysis chunks into unified {analysis_type} report:

ANALYSIS CHUNKS:
{chunk_separator}

INSTRUCTIONS:
1. Remove duplicate information
2. Organize by key risk categories
3. Provide final {analysis_type} risk score (0-100)
4. Highlight most critical 3 findings
5. Format in markdown with headings"""
    
    full_analysis = llama.invoke(combine_prompt)
    return full_analysis

def generate_executive_summary(analyses: Dict[str, str]) -> str:
    """Create polished executive summary from unified analyses"""
    summary_prompt = f"""Create executive summary from these complete analyses:
    
FINANCIAL ANALYSIS:
{analyses.get('financial', 'No data')}

CORPORATE ANALYSIS:
{analyses.get('corporate', 'No data')}

LEGAL ANALYSIS:  
{analyses.get('legal', 'No data')}

Structure as:
1. Overall Risk Rating (Low/Medium/High)
2. Key Findings (3-5 bullet points)
3. Critical Risks (Top 3)
4. Recommended Actions
5. Final Risk Score (0-100)"""
    
    return llama.invoke(summary_prompt)

def analyze_all_files(file_paths: Dict[str, str]) -> Dict[str, str]:
    """Process all JSON files with unified analysis"""
    analyses = {}
    print("Starting unified analysis...\n")
    
    for file_type, path in file_paths.items():
        if not os.path.exists(path):
            print(f" File not found: {path}")
            continue
            
        print(f"Analyzing and combining {file_type} data...")
        with open(path, 'r') as f:
            data = json.load(f)
        
        analyses[file_type] = analyze_with_llama(data, file_type)
        print(f" Unified {file_type.upper()} analysis completed\n")
    
    return analyses

# def save_report(content: str, filename: str = None):
#     """Save report with timestamp"""
#     if not filename:
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"Infosys_Unified_Risk_Report_{timestamp}.md"
    
#     with open(filename, 'w') as f:
#         f.write(content)
#     return filename

def save_report(content: str, filename: str = None, file_path: str = "/teamspace/studios/this_studio/Uvarajan/Whole_Server/uploads/pro/"):
    """Save report with timestamp to a specific directory"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Infosys_Unified_Risk_Report_{timestamp}.md"
    
    full_path = os.path.join(file_path, filename)
    
    os.makedirs(file_path, exist_ok=True)  # Ensure the directory exists
    with open(full_path, 'w') as f:
        f.write(content)
    
    return full_path


def get_all_image_paths(folder_path: str, image_extensions=None):
    """Returns a list of all image file paths in the folder (including subfolders)."""
    if image_extensions is None:
        # Common image file extensions (add more if needed)
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp']

    image_paths = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                image_paths.append(os.path.join(root, file))

    return image_paths


def image_markdown_if_exists(image_path: str, title: str = "Image"):
    """Return Markdown string for image only if the file exists."""
    base_url = "https://8000-01jrcj02963afp5r8dxtpsxdqz.cloudspaces.litng.ai"

    if os.path.exists(image_path):
        full_url = f"{base_url}{image_path}"
        return f"## {title}\n\n![{title}]({full_url})\n"
    return ""

def main2(ticker_symbol,ticker_name,corporation,jurisdiction,period,region):
    ticker_symbol = ticker_symbol  # Samsung Electronics' ticker on the Korea Exchange (KOSPI)
    ticker_name = ticker_name
    corporation =corporation
    jurisdiction = jurisdiction  # South Korea
    period = period  # or use a specific quarter like "Q4 2023"
    region = region  # country code for South Korea

    case_categories="/teamspace/studios/this_studio/Uvarajan/Whole_Server/uploads/pro/Legal_analyses_output/case_categories.png"
    case_severity_distribution="/teamspace/studios/this_studio/Uvarajan/Whole_Server/uploads/pro/Legal_analyses_output/case_severity_distribution.png"
    case_timeline="/teamspace/studios/this_studio/Uvarajan/Whole_Server/uploads/pro/Legal_analyses_output/case_timeline.png"
    risk_matrix="/teamspace/studios/this_studio/Uvarajan/Whole_Server/uploads/pro/Legal_analyses_output/risk_matrix.png"
    financial_health_dashboard="/teamspace/studios/this_studio/Uvarajan/Whole_Server/uploads/pro/Stock_analyses_output/financial_health_dashboard.png"
    price_analysis="/teamspace/studios/this_studio/Uvarajan/Whole_Server/uploads/pro/Stock_analyses_output/price_analysis.png"
    risk_analysis="/teamspace/studios/this_studio/Uvarajan/Whole_Server/uploads/pro/Stock_analyses_output/risk_analysis.png"

    Financial_analysis_Agent(ticker_symbol,ticker_name)
    Legal_analyisi_Agent(corporation,jurisdiction,period)
    company_Background_anslysis_Agent(ticker_name,region)
    FILES = {
        "financial": "/teamspace/studios/this_studio/Uvarajan/Whole_Server/uploads/pro/Stock_analyses_output/company_summary_report.json",
        "corporate": "/teamspace/studios/this_studio/Uvarajan/Whole_Server/uploads/pro/Company_Background_analysis_output/company_background_analysis.json", 
        "legal": "/teamspace/studios/this_studio/Uvarajan/Whole_Server/uploads/pro/Legal_analyses_output/legal_analysis.json"
    }
    
    try:
        # Unified analysis
        analyses = analyze_all_files(FILES)
        
        # Generate executive summary
        print("\nGenerating executive summary...")
        summary = generate_executive_summary(analyses)
        
        # Combine into final report
        full_report = f"""# INFOSYS LTD. COMPREHENSIVE RISK ASSESSMENT
**Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
{summary}

## Detailed Analysis

### FINANCIAL RISK ASSESSMENT
{analyses.get('financial', 'No data')}

{image_markdown_if_exists(financial_health_dashboard, "Financial Health Dashboard")}

### CORPORATE STRUCTURE ANALYSIS  
{analyses.get('corporate', 'No data')}

{image_markdown_if_exists(price_analysis, "Price Analysis")}
{image_markdown_if_exists(risk_analysis, "Risk Analysis")}

### LEGAL RISK EVALUATION
{analyses.get('legal', 'No data')}

{image_markdown_if_exists(case_categories, "Case Categories")}
{image_markdown_if_exists(case_severity_distribution, "Case Severity Distribution")}
{image_markdown_if_exists(case_timeline, "Case Timeline")}
{image_markdown_if_exists(risk_matrix, "Risk Matrix")}

"""
        
        # Save and display
        report_file = save_report(full_report)
        print(f"\n Unified report generation complete!")
        print(f" Report saved to: {os.path.abspath(report_file)}")
        report_path=os.path.abspath(report_file)
        # Print summary
        print("\n=== EXECUTIVE SUMMARY ===\n")
        print(summary[:2000])
        folder = "/teamspace/studios/this_studio/Uvarajan/Whole_Server/uploads/pro/"
        all_images = get_all_image_paths(folder)

        for img_path in all_images:
            print(img_path)

        return report_path,all_images
        
    except Exception as e:
        print(f" Error during unified analysis: {str(e)}")
        return None, None

# ticker_symbol = "005930.KQ"  # Samsung Electronics' ticker on the Korea Exchange (KOSPI)
# ticker_name = "Samsung Electronics Co Ltd"
# corporation = "Samsung Electronics Co., Ltd."
# jurisdiction = "KR"  # South Korea
# period = "2023"  # or use a specific quarter like "Q4 2023"
# region = "kr"  # country code for South Korea

# result = main2(ticker_symbol, ticker_name, corporation, jurisdiction, period, region)
# if result is not None:
#     a, c = result
# else:
#     a, c = None, None
# # a,c=main(ticker_symbol,ticker_name,corporation,jurisdiction,period,region)
# print(a,c)
