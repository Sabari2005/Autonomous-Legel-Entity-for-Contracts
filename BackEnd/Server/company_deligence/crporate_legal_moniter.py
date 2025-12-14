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
import re
class CorporateLegalMonitor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.reports_directory = Path("Legal_analyses_output")
        self.reports_directory.mkdir(exist_ok=True)

        self.jurisdiction_settings = {
            "default": {
                "data_sources": ["reuters.com", "forbes.com", "bloomberg.com"],
                "search_terms": ["legal action", "lawsuit", "litigation"]
            },
            "US": {
                "data_sources": ["justice.gov", "sec.gov", "courtlistener.com"],
                "search_terms": ["class action", "regulatory action", "DOJ case"]
            },
            "EU": {
                "data_sources": ["curia.europa.eu", "europa.eu", "ft.com"],
                "search_terms": ["ECJ ruling", "EU regulation", "GDPR case"]
            }
        }

        self.multilingual_legal_terms = {
            "en": ["lawsuit", "litigation", "legal dispute"],
            "es": ["demanda", "pleito", "disputa legal"],
            "fr": ["procès", "litige", "conflit juridique"]
        }

    def _prepare_search_query(self, corporation: str, jurisdiction: str, period: str) -> str:
        config = self.jurisdiction_settings.get(jurisdiction, self.jurisdiction_settings["default"])
        sources = " OR site:".join(config["data_sources"])
        terms = " OR ".join(config["search_terms"])
        

        language = self._identify_primary_language(jurisdiction)
        if language in self.multilingual_legal_terms:
            terms += " OR " + " OR ".join(self.multilingual_legal_terms[language])
        
        time_constraint = self._calculate_time_filter(period)
        
        return f"{corporation} ({terms}) site:{sources} {time_constraint}"

    def _identify_primary_language(self, jurisdiction: str) -> str:
        language_mapping = {
            "US": "en", "UK": "en", "CA": "en",
            "DE": "de", "FR": "fr", "ES": "es"
        }
        return language_mapping.get(jurisdiction, "en")

    def _calculate_time_filter(self, period: str) -> str:
        period_mapping = {
            "last month": timedelta(days=30),
            "last year": timedelta(days=365),
            "last 5 years": timedelta(days=5*365)
        }
        if period in period_mapping:
            cutoff_date = datetime.now() - period_mapping[period]
            return f"after:{cutoff_date.strftime('%Y-%m-%d')}"
        return ""

    def _identify_legal_authorities(self, content: str) -> List[str]:
        identification_patterns = [
            r"(Supreme Court|High Court|District Court)",
            r"(ECJ|European Court of Justice)",
            r"(SEC|FCA|DOJ|Financial Conduct Authority)"
        ]
        
        found_entities = set()
        for pattern in identification_patterns:
            found_entities.update(re.findall(pattern, content))
        
        return list(found_entities)

    def _categorize_legal_matter(self, content: str) -> str:
        classification_criteria = {
            "Corporate Governance": ["board", "shareholder", "fiduciary"],
            "Regulatory Compliance": ["regulation", "compliance", "violation"],
            "Commercial Dispute": ["contract", "agreement", "breach"],
            "Employment Matter": ["employment", "discrimination", "wage"]
        }
        
        normalized_content = content.lower()
        for category, indicators in classification_criteria.items():
            if any(indicator in normalized_content for indicator in indicators):
                return category
        
        return "Uncategorized"

    def _evaluate_case_severity(self, content: str) -> str:
        normalized_content = content.lower()
        
        if any(term in normalized_content for term in ["criminal", "indictment"]):
            return "Critical"
        elif any(term in normalized_content for term in ["fraud", "penalty"]):
            return "High"
        elif any(term in normalized_content for term in ["dispute", "violation"]):
            return "Medium"
        return "Low"

    def _store_analysis_results(self, analysis_data: Dict) -> str:
        output_file = self.reports_directory / "legal_analysis.json"
        
        with open(output_file, 'w', encoding='utf-8') as output:
            json.dump(analysis_data, output, ensure_ascii=False, indent=2)
        
        return str(output_file)

    def investigate_legal_issues(self, corporation: str, jurisdiction: str = "US",period: str = "last year") -> Dict:

        search_query = self._prepare_search_query(corporation, jurisdiction, period)
        api_endpoint = f"https://serpapi.com/search.json?q={search_query}&api_key={self.api_key}"
        
        try:
            logging.info(f"Analyzing legal issues for {corporation} in {jurisdiction}")
            api_response = requests.get(api_endpoint)
            api_response.raise_for_status()
            response_data = api_response.json()
            
            identified_cases = []
            for result in response_data.get("organic_results", []):
                case_description = result.get("snippet", "")
                case_record = {
                    "case_title": result.get("title"),
                    "source_url": result.get("link"),
                    "origin": result.get("source"),
                    "date_recorded": result.get("date"),
                    "description": case_description,
                    "jurisdiction": jurisdiction,
                    "legal_bodies": self._identify_legal_authorities(case_description),
                    "case_category": self._categorize_legal_matter(case_description),
                    "risk_level": self._evaluate_case_severity(case_description)
                }
                identified_cases.append(case_record)
            
            analysis_results = {
                "case_analysis": {
                    "subject": corporation,
                    "region": jurisdiction,
                    "time_period": period,
                    "analysis_date": datetime.now().isoformat(),
                    "total_cases": len(identified_cases)
                },
                "case_details": identified_cases,
                "risk_breakdown": self._compile_risk_analysis(identified_cases)
            }

            output_path = self._store_analysis_results(analysis_results)
            
            return {
                "analysis_status": "completed",
                "results_path": output_path
            }
            
        except Exception as error:
            logging.error(f"Legal analysis failed: {error}")
            return {
                "analysis_status": "failed",
                "error_details": str(error)
            }

    def _compile_risk_analysis(self, cases: List[Dict]) -> Dict:
        risk_assessment = {
            "severity_distribution": {"Critical": 0, "High": 0, "Medium": 0, "Low": 0},
            "category_analysis": {},
            "authority_references": {}
        }
        
        for legal_case in cases:
            risk_assessment["severity_distribution"][legal_case["risk_level"]] += 1
            case_type = legal_case["case_category"]
            risk_assessment["category_analysis"][case_type] = risk_assessment["category_analysis"].get(case_type, 0) + 1
            for authority in legal_case["legal_bodies"]:
                risk_assessment["authority_references"][authority] = risk_assessment["authority_references"].get(authority, 0) + 1
        
        return risk_assessment
