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
class LegalCaseAnalyzer:
    def __init__(self, file_path):
        self.case_data = self.load_case_file(file_path)
        self.company_name = self.case_data['case_analysis']['subject']
        self.legal_region = self.case_data['case_analysis']['region']
        
    def load_case_file(self, file_path):
        with open(file_path, 'r') as case_file:
            return json.load(case_file)
    
    def create_severity_chart(self):
        severity_levels = self.case_data['risk_breakdown']['severity_distribution']
        
        plt.figure(figsize=(10, 6))
        severity_colors = ['#FF5252', '#FF9800', '#FFEB3B', '#4CAF50']
        case_bars = plt.bar(severity_levels.keys(), severity_levels.values(), color=severity_colors)
        
        plt.title(f'{self.company_name} Legal Case Severity Distribution')
        plt.ylabel('Number of Cases')
        
        for bar in case_bars:
            bar_height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., bar_height,
                    f'{bar_height}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('Legal_analyses_output/case_severity_distribution.png')
        plt.close()
        
    def create_category_chart(self):
        case_types = self.case_data['risk_breakdown']['category_analysis']
        
        plt.figure(figsize=(10, 6))
        highlight_uncategorized = [0.1 if cat == "Uncategorized" else 0 for cat in case_types.keys()]
        plt.pie(case_types.values(), labels=case_types.keys(), 
                autopct='%1.1f%%', startangle=90,
                colors=['#2196F3', '#FFC107', '#9C27B0', '#4CAF50'],
                explode=highlight_uncategorized)
        
        plt.title(f'{self.company_name} Legal Case Categories')
        plt.tight_layout()
        f_path="/teamspace/studios/this_studio/Uvarajan/Whole_Server/uploads/pro/Legal_analyses_output/case_categories.png"
        if os.path.exists(f_path):
            os.remove(f_path)
        try:
            plt.savefig(f_path)
            if not os.path.exists(f_path):
                raise IOError("Save failed, file was not created.")
        except Exception as e:
            if os.path.exists(f_path):
                os.remove(f_path)  # Make sure no partial/broken file remains
            print(f"Error saving plot: {e}")
        # plt.savefig(f_path)
        plt.close()
        
    def create_case_timeline(self):
        all_cases = self.case_data['case_details']
        formatted_dates = []
        
        for case in all_cases:
            try:
                parsed_date = datetime.strptime(case['date_recorded'], '%b %d, %Y')
                formatted_dates.append(parsed_date)
            except:
                continue
        
        if formatted_dates:
            plt.figure(figsize=(12, 4))
            plt.hist(formatted_dates, bins=12, color='#3F51B5', edgecolor='black')
            plt.title(f'{self.company_name} Legal Case Timeline')
            plt.xlabel('Date')
            plt.ylabel('Number of Cases')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            f_path="/teamspace/studios/this_studio/Uvarajan/Whole_Server/uploads/pro/Legal_analyses_output/case_timeline.png"
            if os.path.exists(f_path):
                os.remove(f_path)
            try:
                plt.savefig(f_path)
                if not os.path.exists(f_path):
                    raise IOError("Save failed, file was not created.")
            except Exception as e:
                if os.path.exists(f_path):
                    os.remove(f_path)  
                print(f"Error saving plot: {e}")
            # plt.savefig('')
            plt.close()
    
    def create_risk_matrix(self):
        case_records = self.case_data['case_details']
        risk_details = []
        
        for case in case_records:
            risk_details.append({
                'Title': case['case_title'],
                'Date': case['date_recorded'],
                'Risk': case['risk_level'],
                'Category': case['case_category'],
                'Source': case['origin']
            })
        
        risk_df = pd.DataFrame(risk_details)
        risk_df['Risk_Score'] = risk_df['Risk'].map({'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1})
        
        plt.figure(figsize=(12, 6))
        risk_plot = plt.scatter(
            x=pd.to_datetime(risk_df['Date']),
            y=risk_df['Risk_Score'],
            c=risk_df['Risk_Score'],
            cmap='RdYlGn_r',
            s=100,
            alpha=0.7
        )
        
        plt.title(f'{self.company_name} Legal Risk Matrix')
        plt.xlabel('Date')
        plt.ylabel('Risk Level')
        plt.yticks([1, 2, 3, 4], ['Low', 'Medium', 'High', 'Critical'])
        plt.colorbar(risk_plot, label='Risk Severity')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        f_path="/teamspace/studios/this_studio/Uvarajan/Whole_Server/uploads/pro/Legal_analyses_output/risk_matrix.png"
        if os.path.exists(f_path):
            os.remove(f_path)
        try:
            plt.savefig(f_path)
            if not os.path.exists(f_path):
                raise IOError("Save failed, file was not created.")
        except Exception as e:
            if os.path.exists(f_path):
                os.remove(f_path)  
            print(f"Error saving plot: {e}")
        
        # plt.savefig('')
        plt.close()
        
    def create_case_summary(self):
        report_summary = {
            "Company": self.company_name,
            "Region": self.legal_region,
            "AnalysisPeriod": self.case_data['case_analysis']['time_period'],
            "TotalCases": self.case_data['case_analysis']['total_cases'],
            "RiskBreakdown": self.case_data['risk_breakdown']['severity_distribution'],
            "CaseCategories": self.case_data['risk_breakdown']['category_analysis'],
            "NotableCases": [],
            "KeyFindings": {
                "HighestRiskCategory": max(
                    self.case_data['risk_breakdown']['category_analysis'].items(),
                    key=lambda x: x[1]
                )[0],
                "RiskConcentration": max(
                    self.case_data['risk_breakdown']['severity_distribution'].items(),
                    key=lambda x: x[1]
                )[0],
                "RecentHighRiskCases": []
            }
        }
        
        for case in self.case_data['case_details']:
            if case['risk_level'] in ['High', 'Critical']:
                report_summary['NotableCases'].append({
                    'Title': case['case_title'],
                    'Date': case['date_recorded'],
                    'Risk': case['risk_level'],
                    'Source': case['origin']
                })
                
                try:
                    case_date = datetime.strptime(case['date_recorded'], '%b %d, %Y')
                    if (datetime.now() - case_date).days < 180:
                        report_summary['KeyFindings']['RecentHighRiskCases'].append({
                            'Title': case['case_title'],
                            'Date': case['date_recorded']
                        })
                except:
                    continue
            
        return report_summary
    
    def generate_all_legal_reports(self):
        self.create_severity_chart()
        self.create_category_chart()
        self.create_case_timeline()
        self.create_risk_matrix()
        final_report = self.create_case_summary()
        
        print("Generated legal analysis files:")
        print("- case_severity_distribution.png")
        print("- case_categories.png")
        print("- case_timeline.png")
        print("- risk_matrix.png")
        
        return final_report
