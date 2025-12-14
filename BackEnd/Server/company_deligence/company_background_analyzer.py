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


class CompanyBackgroundAnalyzer:
    def __init__(self, api_key: str):
        self.base_url = "https://api.opencorporates.com/v0.4"
        self.api_key = api_key
        self.reports_directory = Path("Company_Background_analysis_output")
        self.reports_directory.mkdir(exist_ok=True)
        self.data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "tool_version": "1.2",
                "data_sources": ["OpenCorporates"]
            }
        }
        self.logger = self._setup_logger()
        self.request_delay = 0.7  

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _make_api_call(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        if params is None:
            params = {}
        params['api_token'] = self.api_key
        
        try:
            response = requests.get(
                f"{self.base_url}/{endpoint}",
                params=params,
                timeout=15
            )
            response.raise_for_status()
            
            result = response.json()
            self._store_api_call(endpoint, params, response.status_code, result)
            return result
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed for {endpoint}: {str(e)}")
            self._store_error(endpoint, str(e))
            return {"error": str(e)}

    def _store_api_call(self, endpoint: str, params: Dict, status_code: int, result: Dict):
        if "api_calls" not in self.data:
            self.data["api_calls"] = []
            
        self.data["api_calls"].append({
            "endpoint": endpoint,
            "parameters": params,
            "status_code": status_code,
            "timestamp": datetime.now().isoformat(),
            "response": result
        })

    def _store_error(self, endpoint: str, error_msg: str):
        if "errors" not in self.data:
            self.data["errors"] = []
            
        self.data["errors"].append({
            "endpoint": endpoint,
            "error": error_msg,
            "timestamp": datetime.now().isoformat()
        })

    def search_companies(self, company_name: str, jurisdiction: Optional[str] = None, 
                        limit: int = 5) -> List[Dict]:
        params = {
            'q': company_name,
            'per_page': limit,
            'sparse': 'true'
        }
        if jurisdiction:
            params['jurisdiction_code'] = jurisdiction.lower()
            
        result = self._make_api_call("companies/search", params)
        return result.get('results', {}).get('companies', [])

    def get_company_full_profile(self, jurisdiction_code: str, company_number: str) -> Dict:
        endpoints = [
            ('company_details', f"companies/{jurisdiction_code.lower()}/{company_number}"),
            ('officers', f"companies/{jurisdiction_code.lower()}/{company_number}/officers"),
            ('filings', f"companies/{jurisdiction_code.lower()}/{company_number}/filings"),
            ('ownership', f"companies/{jurisdiction_code.lower()}/{company_number}/owners"),
            ('subsidiaries', f"companies/{jurisdiction_code.lower()}/{company_number}/subsidiaries")
        ]
        
        profile = {}
        for name, endpoint in endpoints:
            profile[name] = self._make_api_call(endpoint)
            time.sleep(self.request_delay)
            
        return profile

    def analyze_company_network(self, jurisdiction_code: str, company_number: str, 
                              depth: int = 1) -> Dict:
        network = {}
        current_depth = 0
        
        while current_depth < depth:
            endpoint = f"companies/{jurisdiction_code.lower()}/{company_number}/network"
            result = self._make_api_call(endpoint)
            
            if "error" in result:
                break
                
            network[f"depth_{current_depth}"] = result.get('results', {})
            current_depth += 1
            time.sleep(self.request_delay)
            
        return network

    def save_analysis(self, filename: str = "Company_Background_analysis_output/company_background_analysis.json") -> Tuple[bool, str]:
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Analysis successfully saved to {filename}")
            return True, filename
        except Exception as e:
            self.logger.error(f"Failed to save analysis: {str(e)}")
            return False, str(e)

    def enhance_with_external_data(self, company_data: Dict) -> Dict:
        company_data["external_data"] = {
            "last_updated": datetime.now().isoformat(),
            "notes": "External data integration not configured"
        }
        return company_data
